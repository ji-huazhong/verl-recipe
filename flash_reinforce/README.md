# FlashREINFORCE

Experiment recipes for the task families in [FlashREINFORCE](https://yifanzhang-pro.github.io/FlashREINFORCE/): long-CoT reasoning, mathematical reasoning, Python-tool reasoning (dense and MoE), and ALFWorld. The algorithm lives in verl; this directory adds configuration, data preparation, a Sandbox Fusion function tool, and two small agent adapters. There is no custom trainer and no `uni-agent` dependency.

## Required verl version

Install the fork revision in [REQUIRED_VERL.txt](REQUIRED_VERL.txt). It includes the FlashREINFORCE objective and optional negative-token filtering needed by ALFWorld. Stock verl releases do not provide this combination yet.

Use a Linux CUDA environment with the pinned verl revision and its supported PyTorch/vLLM dependencies. Manage the environment with `uv`; follow that revision's installation instructions for the CUDA stack. From this repository root, inspect the exact core dependency with:

```bash
bash install_verl.sh --recipe flash_reinforce --show
uv pip install "verl@git+https://github.com/ji-huazhong/verl.git@f78c319e77e4895f769c0368e398ca91137788d2"
uv pip install wandb
```

ALFWorld additionally needs its text environment and game files, accessible at the same path on every rollout worker:

```bash
uv pip install alfworld
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download
```

The text-only package is sufficient; visual THOR dependencies are unnecessary. See the [official ALFWorld installation instructions](https://github.com/alfworld/alfworld#quickstart). Python-tool experiments require a running [Sandbox Fusion](https://github.com/bytedance/SandboxFusion) service:

```bash
export SANDBOX_FUSION_URL=http://sandbox-host:8080/run_code
```

The tool reuses verl's Sandbox Fusion client. Each call runs in a fresh process, with a 30-second execution limit and a 1-GiB memory limit. These are recipe choices; they are not claimed to match the original executor. Export these environment variables before starting Ray, and make this repository, model files, data, and dependencies available to every node.

## Configurations

All training configurations use one rollout per prompt, the full-batch mean reward baseline, a single full-batch optimizer step, sample-mean loss, token importance sampling, and sequence-level trust. PPO clipping, reference KL, a critic, advantage standardization, and external rollout correction are disabled.

| Launch name | Model | Train size / batch | Token limits | Trust δ | Epochs | Evaluation |
| --- | --- | --- | --- | --- | --- | --- |
| `r1_math` | DeepSeek-R1-Distill-Qwen-1.5B | 1,460 / 128 | prompt 1,024; generated 8,192; total 9,216 | .003 | 350 | AIME24 + AIME25, avg@32, every 128 updates |
| `qwen_math` | Qwen2.5-Math-1.5B | 7,500 / 128 | prompt 2,048; generated 4,096; total 6,144 | .003 | 68 | AMC23 + Minerva + AIME25, avg@4, every 100 updates |
| `qwen7b_tool` | Qwen2.5-7B-Instruct | 7,500 / 128 | 10 assistant turns; generated 6,144 across turns; total 8,192 | .001 | 68 | Same three benchmarks, avg@4, every 100 updates |
| `qwen3_tool` | Qwen3-30B-A3B | 7,500 / 128 | 10 assistant turns; generated 8,192 per turn; total 18,432 | .01 | 17* | Same three benchmarks, avg@4, every 50 updates |
| `qwen3_tool_20turn` | Qwen3-30B-A3B | 7,500 / 128 | 20 assistant turns; generated 14,336 across turns; total 16,384 | .003 | 17 | Same three benchmarks, avg@4, every 100 updates |
| `alfworld` | Qwen2.5-7B-Instruct | 1,024 / 64 | 50 action attempts; generated 8,192 across turns; total 16,384 | .001 | 20 | 140 seen + 134 unseen games, avg@3, every 40 updates |

Parameters come from [paper Tables 9, 11–13, 17](https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf) and the [released settings](https://github.com/yifanzhang-pro/FlashREINFORCE/tree/master/examples/settings). `qwen3_tool` follows the paper's main comparison; `qwen3_tool_20turn` follows the separately released longer-turn setting. *The main comparison's epoch horizon is not specified in Table 12; 17 epochs is a launch default borrowed from the released setting, not a verified paper value.*

Shared optimization is AdamW, learning rate 1e-6, weight decay .1, gradient clipping 1, cosine decay with 3% warmup and minimum learning-rate ratio .1. ALFWorld uses constant 1e-6 without warmup and retains the highest-entropy `ceil(.9 * T)` policy tokens on failed trajectories. Failure means uncentered reward ≤ 0. Successes retain every token; filtering never changes the original sequence-length or batch-size denominator. Other tasks disable this filter.

Training sampling uses temperature 1 and top-p 1. R1 evaluation uses .6/.95, ALFWorld uses 1/.7, and the other evaluations use .7/.7. MoE thinking and routing replay are disabled. Checkpoints are saved every 100 updates as a local I/O default (released configurations use 10).

## Prepare data

Run these commands from the repository root. Output parquet uses verl's ordinary `prompt`, `data_source`, `reward_model`, and `extra_info` columns. Preparation records row counts and output SHA-256 hashes in `manifest.json`.

```bash
export FLASH_DATA_DIR="$HOME/data/flash_reinforce"
python -m flash_reinforce.prepare_data r1 --output-dir "$FLASH_DATA_DIR/r1"
```

This uses `sail/Sanity-Test-R1D-1.5B` for training and its test split for AIME24/AIME25. Evaluation requires 30 problems per benchmark and distinct source labels. You can instead supply explicit `--eval-source aime24=/path/aime24.parquet --eval-source aime25=/path/aime25.parquet`.

For DAPO tasks, provide the evaluation splits explicitly so training/evaluation provenance is inspectable:

```bash
python -m flash_reinforce.prepare_data math \
  --output-dir "$FLASH_DATA_DIR/math" \
  --eval-source amc23=/data/eval/amc23.parquet \
  --eval-source minerva=/data/eval/minerva.parquet \
  --eval-source aime25=/data/eval/aime25.parquet

python -m flash_reinforce.prepare_data tool \
  --output-dir "$FLASH_DATA_DIR/tool" \
  --eval-source amc23=/data/eval/amc23.parquet \
  --eval-source minerva=/data/eval/minerva.parquet \
  --eval-source aime25=/data/eval/aime25.parquet
```

Each input can use existing verl chat prompts and `reward_model.ground_truth`, or raw `problem`/`question`/`Problem` and `answer`/`Answer` fields. Full worked `solution` text is deliberately not accepted as the answer label. For Hugging Face evaluation IDs, the loader reads the `train` split; materialize other splits as parquet first. The training default is `BytedTsinghua-SIA/DAPO-Math-17k`; `--train-source` accepts parquet, a `datasets.save_to_disk` directory, or a dataset ID, and `--train-revision` pins the HF revision. The first 7,500 unique complete prompts are retained. Use the authors' exact prepared parquet if available: deduplication alone does not establish identical sample IDs or ordering.

The paper's five-benchmark Qwen-Math final comparison uses avg@16. Add `--eval-source math500=/data/eval/math500.parquet --eval-source olympiadbench=/data/eval/olympiadbench.parquet` when preparing math data, and evaluate the selected checkpoint with:

```bash
bash flash_reinforce/run.sh qwen_math \
  trainer.val_only=true actor_rollout_ref.model.path=/path/to/hf-checkpoint \
  actor_rollout_ref.rollout.val_kwargs.n=16
```

Use an exported Hugging Face checkpoint, not an FSDP shard directory. verl reports benchmark metrics separately; avg@N means the mean of N independent scores per problem, not pass@N. Report the equal-weight benchmark mean separately.

ALFWorld preparation selects solvable games from all six supported task types:

```bash
python -m flash_reinforce.prepare_data alfworld --output-dir "$FLASH_DATA_DIR/alfworld"
# To use an exact training subset, add --train-manifest /data/train_games.txt.
```

Manifest lines are game paths relative to `ALFWORLD_DATA`, such as `json_2.1.1/train/.../game.tw-pddl`. Without a manifest, preparation takes the first 1,024 eligible games in sorted order. It validates 140 seen and 134 unseen evaluation games, rejects cross-split manifest entries, and writes the selected paths to `train_games.txt` / `val_games.txt`. The original paper's exact 1,024 IDs are not supplied here; do not label this default subset an exact reproduction.

The environment adapter opens one official TextWorld game per trajectory, with ALFWorld's object-name wrappers. The model emits one `<action>...</action>` command; observations are appended as user messages with policy mask zero. Reward is the terminal `won` flag, including success on the final allowed action. Invalid formats consume an action attempt. Environments close on normal completion and errors. This prompt/action protocol is a local adapter, not an assertion that the authors used identical formatting.

## Launch

```bash
# Resolve configuration without model loading or GPUs.
bash flash_reinforce/run.sh alfworld --cfg job --resolve

# Each name from the configuration table is a runnable entry point.
TRAIN_GPUS=2 ROLLOUT_GPUS=6 bash flash_reinforce/run.sh r1_math
TRAIN_GPUS=2 ROLLOUT_GPUS=6 bash flash_reinforce/run.sh qwen_math
TRAIN_GPUS=2 ROLLOUT_GPUS=6 bash flash_reinforce/run.sh qwen7b_tool
TRAIN_GPUS=8 ROLLOUT_GPUS=8 ROLLOUT_TP=2 bash flash_reinforce/run.sh qwen3_tool
TRAIN_GPUS=8 ROLLOUT_GPUS=8 ROLLOUT_TP=2 bash flash_reinforce/run.sh qwen3_tool_20turn
TRAIN_GPUS=2 ROLLOUT_GPUS=6 bash flash_reinforce/run.sh alfworld
```

GPU counts are allocation examples, not measured memory recommendations. Actor and rollout pools are separate; their GPU requirements add together. Adapt nodes, sharding, tensor parallelism, and microbatch/token limits to your hardware. The 30B examples need 16 GPUs with the defaults; for two 8-GPU nodes, override `trainer.nnodes=1`, `trainer.n_gpus_per_node=8`, `actor_rollout_ref.rollout.nnodes=1`, and `actor_rollout_ref.rollout.n_gpus_per_node=8` within the same Ray cluster. Both roles remain one-node pools.

The entry point uses native verl V1 `separate_async`, FSDP2, vLLM, stored behavior log probabilities, and NCCL weight synchronization after every optimizer step. It performs no Ray cluster setup. `PYTHON` chooses the executable; `FLASH_OUTPUT_DIR` chooses the checkpoint root. Normal Hydra overrides are supported, e.g. `trainer.logger=[console]`, `trainer.total_training_steps=2`, or `actor_rollout_ref.model.path=/models/local-model`.

For R1's paper threshold cap, override `actor_rollout_ref.actor.policy_loss.flash_reinforce_kl_threshold=0.005`; the default .003 comes from the released base configuration. Set ALFWorld's `flash_reinforce_neg_topq=1.0` to disable filtering for an ablation. In both cases use the complete `actor_rollout_ref.actor.policy_loss.<field>` key.

## Interpretation and validation limits

This is a port of all task families, not a claim to reproduce published scores or every ablation. Important differences are explicit:

- Native asynchronous sampling uses 16-prompt requests, uncapped policy staleness, and 1 warmup learner batch (4 for R1/MoE). This does not reproduce labs-molt's 512-prompt requests, queue-depth controls, or a fixed lag of 4/8. Warmup batches and queue depth are different quantities. Inspect measured staleness when comparing throughput and learning curves.
- The R1 step-5,000 continuation requires the corresponding weights, fresh AdamW state, a specified continuation learning rate, 50-update warmup, and cumulative-step evaluation accounting. The default fresh run and an ordinary `resume_mode` override do not reconstruct that protocol.
- Mathematical grading reuses verl's strict boxed-answer verifier and converts its result to binary 0/1. Tool tasks grade only the committed final assistant answer, never sandbox output. Verifier, prompt, data-order, truncation, executor, and runtime differences can affect results.
- Routing-replay stability runs, GRPO baselines, geometric gates, token-mean objectives, and all no-IS/no-trust variants are not separate recipes here. They require their own validated algorithms/configurations.

Local verification covers all six Hydra configurations, core config validation, data selection, terminal environment rewards, policy/observation masks, behavior-log-probability alignment, action and generation budgets, and environment cleanup using test doubles:

```bash
python -m pytest flash_reinforce/tests -q
bash -n flash_reinforce/run.sh
ruff check flash_reinforce
ruff format --check flash_reinforce
```

GPU training, real Sandbox Fusion execution, and an installed ALFWorld game rollout have not been exercised in this local CPU environment. Validate those integrations on the target cluster before a full experiment.
