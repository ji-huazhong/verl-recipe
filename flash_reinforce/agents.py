# Copyright 2026 verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Small agent adapters using verl's existing token alignment and rollout client."""

import asyncio
import os
import re
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


class PythonToolAgentLoop(ToolAgentLoop):
    """Reuse tool parsing/execution while enforcing separate context and generation budgets."""

    def __init__(self, *args, max_generated_tokens, max_tokens_per_turn, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_generated_tokens = max_generated_tokens
        self.max_tokens_per_turn = max_tokens_per_turn

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        remaining = self.rollout_config.max_model_len - len(agent_data.prompt_ids)
        if self.max_generated_tokens is not None:
            remaining = min(remaining, self.max_generated_tokens - sum(agent_data.response_mask))
        if remaining <= 0:
            return AgentState.TERMINATED
        params = {**sampling_params, "max_tokens": min(remaining, self.max_tokens_per_turn)}
        state = await super()._handle_generating_state(agent_data, params, ignore_termination)
        tools = [tool.tool_schema for tool in agent_data._active_tools.values()]
        content, calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        # Only the last assistant answer is graded; sandbox output is never a candidate.
        agent_data.extra_fields["final_answer"] = content if state == AgentState.TERMINATED and not calls else ""
        return state

    async def _handle_processing_tools_state(self, agent_data):
        previous = (agent_data.prompt_ids, agent_data.response_mask, agent_data.response_logprobs)
        state = await super()._handle_processing_tools_state(agent_data)
        if len(agent_data.prompt_ids) > self.rollout_config.max_model_len:
            # Do not store an over-budget observation or retokenize earlier policy actions.
            agent_data.prompt_ids, agent_data.response_mask, agent_data.response_logprobs = previous
            return AgentState.TERMINATED
        return state


def make_alfworld_env(game_file, max_actions):
    """Open precisely one game with ALFWorld's standard TextWorld wrappers."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    game_file = Path(game_file)
    if not game_file.is_absolute():
        game_file = Path(os.environ["ALFWORLD_DATA"]) / game_file
    if not game_file.is_file():
        raise FileNotFoundError(game_file)
    env_id = textworld.gym.register_games(
        [str(game_file)],
        textworld.EnvInfos(won=True, extras=["gamefile"]),
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_actions,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)


class ALFWorldAgentLoop(AgentLoopBase):
    """Keep one environment alive per rollout and emit one trajectory with binary terminal reward."""

    def __init__(self, *args, max_actions=50, max_generated_tokens=8192, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_actions = max_actions
        self.max_generated_tokens = max_generated_tokens

    async def run(self, sampling_params, **kwargs):
        env = await asyncio.to_thread(make_alfworld_env, kwargs["extra_info"]["game_file"], self.max_actions)
        try:
            return await self._run_episode(env, sampling_params, **kwargs)
        finally:
            # Also close failed/cancelled episodes; state must not leak to another sample.
            await asyncio.to_thread(env.close)

    async def _run_episode(self, env, sampling_params, **kwargs):
        observations, _ = await asyncio.to_thread(env.reset)
        messages = [dict(message) for message in kwargs["raw_prompt"]]
        messages[-1]["content"] += "\n\n" + observations[0]
        prompt_ids = await self.ct_build_initial_tokens(messages)
        response_mask, log_probs = [], []
        extra = {}
        request_id = uuid4().hex
        won = False
        actions = 0
        generated = 0
        for _ in range(self.max_actions):
            remaining = min(
                self.max_generated_tokens - generated,
                self.rollout_config.max_model_len - len(prompt_ids),
            )
            if remaining <= 0:
                break
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params={**sampling_params, "max_tokens": remaining},
                priority=int(kwargs.get("priority", 0)),
            )
            if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
                raise ValueError("ALFWorld requires stored behavior log probabilities for every generated token")
            if not output.token_ids:
                break
            generated += len(output.token_ids)
            merged, response_mask, log_probs = await self.ct_merge_assistant_token(
                prompt_ids, output.token_ids, response_mask, log_probs, assistant_logprobs=output.log_probs
            )
            prompt_ids = merged.token_ids
            for key, value in output.extra_fields.items():
                if key == "min_global_steps" and key in extra:
                    extra[key] = min(extra[key], value)
                elif key == "max_global_steps" and key in extra:
                    extra[key] = max(extra[key], value)
                else:
                    extra[key] = value
            text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            matches = re.findall(r"<action>\s*(.*?)\s*</action>", text, flags=re.DOTALL)
            action = matches[-1].strip() if matches else ""
            actions += 1
            if not action or "\n" in action:
                observation, done = "Invalid action. Return one command inside <action>...</action>.", False
            else:
                observations, _, dones, infos = await asyncio.to_thread(env.step, [action])
                observation, done = observations[0], bool(dones[0])
                won = bool(infos["won"][0])
            if done or won or actions == self.max_actions:
                break
            updated = messages + [{"role": "user", "content": observation}]
            merged, new_mask, new_log_probs = await self.ct_merge_context_msg(
                messages, updated, prompt_ids, response_mask, log_probs
            )
            if len(merged.token_ids) >= self.rollout_config.max_model_len:
                break
            messages, prompt_ids = updated, merged.token_ids
            response_mask, log_probs = new_mask, new_log_probs
        if not any(response_mask):
            raise ValueError("ALFWorld episode produced no policy tokens")
        extra.update(
            {
                "reward_extra_info": {"acc": float(won), "success": float(won), "actions": actions},
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        initial_prompt_len = len(prompt_ids) - len(response_mask)
        return AgentLoopOutput(
            prompt_ids=prompt_ids[:initial_prompt_len],
            response_ids=prompt_ids[initial_prompt_len:],
            response_mask=response_mask,
            response_logprobs=log_probs,
            reward_score=float(won),
            num_turns=2 * actions,
            metrics={},
            extra_fields=extra,
        )
