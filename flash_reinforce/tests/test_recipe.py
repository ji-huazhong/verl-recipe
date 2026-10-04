# Copyright 2026 verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks against the pinned verl config and agent interfaces."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flash_reinforce import agents
from flash_reinforce.prepare_data import alfworld_rows, math_row, unique_math_rows
from flash_reinforce.reward import compute_score
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.utils.config import validate_config

RECIPE = Path(__file__).resolve().parents[1]
SETTINGS = {
    "r1_math": (128, 9216, 0.003, 32),
    "qwen_math": (128, 6144, 0.003, 4),
    "qwen7b_tool": (128, 8192, 0.001, 4),
    "qwen3_tool": (128, 18432, 0.01, 4),
    "qwen3_tool_20turn": (128, 16384, 0.003, 4),
    "alfworld": (64, 16384, 0.001, 3),
}


@pytest.mark.parametrize("name", SETTINGS)
def test_resolved_config_uses_native_one_pass_training(name, monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_RECIPE_DIR", str(RECIPE))
    monkeypatch.setenv("FLASH_DATA_DIR", str(tmp_path))
    with initialize_config_dir(config_dir=str(RECIPE / "config"), version_base=None):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    batch, context, threshold, eval_n = SETTINGS[name]
    actor, rollout = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    assert cfg.data.train_batch_size == actor.ppo_mini_batch_size == batch
    assert actor.ppo_epochs == rollout.n == cfg.trainer.v1.separate_async.parameter_sync_step == 1
    assert rollout.max_model_len == context
    assert actor.policy_loss.flash_reinforce_kl_threshold == threshold
    assert rollout.val_kwargs.n == eval_n
    assert not cfg.actor_rollout_ref.hybrid_engine
    assert cfg.trainer.v1.sampler.max_off_policy_threshold is None
    assert cfg.trainer.v1.separate_async.num_warmup_batches * batch >= cfg.data.gen_batch_size
    assert actor.policy_loss.flash_reinforce_neg_topq == (0.9 if name == "alfworld" else 1.0)
    assert actor.optim.lr_scheduler_type == ("constant" if name == "alfworld" else "cosine")
    assert Path(cfg.reward.custom_reward_function.path).is_file()
    if rollout.agent.agent_loop_config_path:
        definitions = OmegaConf.load(rollout.agent.agent_loop_config_path)
        selected = next(entry for entry in definitions if entry.name == rollout.agent.default_agent_loop)
        assert selected._target_.startswith("flash_reinforce.agents.")
    if "tool" in name:
        assert Path(rollout.multi_turn.function_tool_path).is_file()
        assert cfg.data.apply_chat_template_kwargs.enable_thinking is False


def test_math_data_preserves_answers_and_deduplicates():
    row = {"problem": "What is 1+1?", "answer": "2"}
    other = {"problem": "What is 2+2?", "answer": "4"}
    rows = unique_math_rows([row, row, other], 2, tool=True)
    assert len(rows) == 2
    assert rows[0]["reward_model"]["ground_truth"] == "2"
    assert "python tool" in rows[0]["prompt"][-1]["content"]
    assert math_row(row, 0, "val", benchmark="minerva")["data_source"] == "minerva"
    with pytest.raises(ValueError, match="found 1"):
        unique_math_rows([row, row], 2)
    with pytest.raises(ValueError, match="answer/Answer"):
        math_row({"problem": "x", "solution": "A reference solution is not a short ground truth"}, 0, "train")


def test_reward_is_binary_and_uses_only_committed_answer():
    correct = r"The answer is \boxed{2}."
    incorrect = r"The answer is \boxed{3}."
    assert compute_score("amc23", correct, "2")["score"] == 1.0
    assert compute_score("amc23", incorrect, "2")["score"] == 0.0
    assert compute_score("amc23", correct, "2", {"final_answer": incorrect})["score"] == 0.0
    assert compute_score("amc23", correct, "2", {"final_answer": ""})["score"] == 0.0
    with pytest.raises(ValueError, match="environment"):
        compute_score("alfworld_valid_seen", "success", "1")


def test_alfworld_selection_excludes_unsolvable_games_and_wrong_splits(tmp_path):
    import json

    split = tmp_path / "json_2.1.1/train"
    for name, solvable in [("a", True), ("b", False)]:
        directory = split / name
        directory.mkdir(parents=True)
        (directory / "game.tw-pddl").write_text(json.dumps({"solvable": solvable}))
        (directory / "traj_data.json").write_text(json.dumps({"task_type": "pick_and_place_simple"}))
    rows = alfworld_rows(tmp_path, "train")
    assert len(rows) == 1
    assert rows[0]["extra_info"]["game_file"] == "json_2.1.1/train/a/game.tw-pddl"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(rows[0]["extra_info"]["game_file"] + "\n")
    with pytest.raises(ValueError, match="does not belong"):
        alfworld_rows(tmp_path, "valid_seen", manifest=manifest)


def _alf_loop(outputs, max_actions=50, max_tokens=8192, context=16384):
    loop = agents.ALFWorldAgentLoop.__new__(agents.ALFWorldAgentLoop)
    loop.max_actions, loop.max_generated_tokens = max_actions, max_tokens
    loop.rollout_config = SimpleNamespace(max_model_len=context)
    loop.server_manager = SimpleNamespace(generate=AsyncMock(side_effect=outputs))
    loop.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "<action>look</action>" if ids[0] != 9 else "invalid")
    loop.ct_build_initial_tokens = AsyncMock(return_value=[101, 102])

    async def merge_assistant(prompt, tokens, mask, probs, assistant_logprobs):
        return SimpleNamespace(token_ids=prompt + tokens), mask + [1] * len(tokens), probs + assistant_logprobs

    async def merge_context(previous, updated, prompt, mask, probs):
        return SimpleNamespace(token_ids=prompt + [99]), mask + [0], probs + [0.0]

    loop.ct_merge_assistant_token = merge_assistant
    loop.ct_merge_context_msg = merge_context
    return loop


def _tokens(ids, version=0):
    return SimpleNamespace(
        token_ids=ids,
        log_probs=[-0.5] * len(ids),
        extra_fields={"min_global_steps": version, "max_global_steps": version},
    )


def _env(step_results):
    return SimpleNamespace(
        reset=Mock(return_value=(["Find the apple"], {})), step=Mock(side_effect=step_results), close=Mock()
    )


@pytest.mark.parametrize("success", [True, False])
def test_alfworld_full_trajectory_masks_observations_and_preserves_behavior(monkeypatch, success):
    env = _env([(["Continue"], [999], [False], {"won": [False]}), (["Done"], [999], [True], {"won": [success]})])
    monkeypatch.setattr(agents, "make_alfworld_env", lambda *args: env)
    loop = _alf_loop([_tokens([1, 2], 3), _tokens([3], 5)])
    out = asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "Play"}], extra_info={"game_file": "game"}))
    assert out.response_ids == [1, 2, 99, 3]
    assert out.response_mask == [1, 1, 0, 1]
    assert out.response_logprobs == [-0.5, -0.5, 0.0, -0.5]
    assert out.reward_score == float(success)  # Never the shaped 999 returned by env.step.
    assert out.extra_fields["min_global_steps"] == 3
    assert out.extra_fields["max_global_steps"] == 5
    assert out.extra_fields["reward_extra_info"]["actions"] == 2
    assert out.as_dict()["rm_scores"].sum() == float(success)
    env.close.assert_called_once()


def test_alfworld_executes_last_allowed_action_and_closes_on_error(monkeypatch):
    env = _env([(["Done"], [0], [True], {"won": [True]})])
    monkeypatch.setattr(agents, "make_alfworld_env", lambda *args: env)
    loop = _alf_loop([_tokens([1])], max_actions=1)
    out = asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "Play"}], extra_info={"game_file": "game"}))
    assert out.reward_score == 1.0
    env.close.assert_called_once()
    env.close.reset_mock()
    loop = _alf_loop([RuntimeError("generation failed")])
    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "Play"}], extra_info={"game_file": "game"}))
    env.close.assert_called_once()


def test_alfworld_generation_budget_and_invalid_actions(monkeypatch):
    env = _env([])
    monkeypatch.setattr(agents, "make_alfworld_env", lambda *args: env)
    loop = _alf_loop([_tokens([9, 9])], max_tokens=2)
    out = asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "Play"}], extra_info={"game_file": "game"}))
    assert out.reward_score == 0.0
    assert sum(out.response_mask) == 2
    assert loop.server_manager.generate.await_count == 1
    env.step.assert_not_called()
    env.close.assert_called_once()


def test_alfworld_rejects_missing_behavior_logprobs(monkeypatch):
    env = _env([])
    monkeypatch.setattr(agents, "make_alfworld_env", lambda *args: env)
    token_output = _tokens([1])
    token_output.log_probs = None
    loop = _alf_loop([token_output])
    with pytest.raises(ValueError, match="behavior log probabilities"):
        asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "Play"}], extra_info={"game_file": "game"}))
    env.close.assert_called_once()


@pytest.mark.parametrize("has_tool_call", [False, True])
def test_python_final_answer_and_separate_token_budget(monkeypatch, has_tool_call):
    parent = AsyncMock(return_value=AgentState.TERMINATED)
    monkeypatch.setattr(ToolAgentLoop, "_handle_generating_state", parent)
    loop = agents.PythonToolAgentLoop.__new__(agents.PythonToolAgentLoop)
    loop.rollout_config = SimpleNamespace(max_model_len=10)
    loop.max_generated_tokens, loop.max_tokens_per_turn = 5, 4
    loop.tool_parser = SimpleNamespace(
        extract_tool_calls=AsyncMock(return_value=(r"\boxed{2}", [1] if has_tool_call else []))
    )
    data = SimpleNamespace(
        prompt_ids=[1] * 8, response_mask=[1, 1, 0, 1], response_ids=[1], extra_fields={}, _active_tools={}
    )
    state = asyncio.run(loop._handle_generating_state(data, {"temperature": 1.0}))
    assert state == AgentState.TERMINATED
    assert parent.call_args.args[1]["max_tokens"] == 2
    assert data.extra_fields["final_answer"] == ("" if has_tool_call else r"\boxed{2}")
