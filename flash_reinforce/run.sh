#!/usr/bin/env bash
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
  r1_math|qwen_math|qwen7b_tool|qwen3_tool|qwen3_tool_20turn|alfworld) setting=$1; shift ;;
  *) echo "Usage: $0 {r1_math|qwen_math|qwen7b_tool|qwen3_tool|qwen3_tool_20turn|alfworld} [Hydra arguments...]" >&2; exit 2 ;;
esac
export FLASH_RECIPE_DIR="$recipe_dir"
export FLASH_DATA_DIR="${FLASH_DATA_DIR:-$HOME/data/flash_reinforce}"
export PYTHONPATH="$(dirname -- "$recipe_dir")${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${PYTHON:-python3}

"$python_bin" - <<'CHECK'
from verl.workers.config import PolicyLossConfig
from verl.trainer.ppo.core_algos import get_policy_loss_fn
get_policy_loss_fn("flash_reinforce")
if "flash_reinforce_neg_topq" not in PolicyLossConfig.__dataclass_fields__:
    raise SystemExit("Install the verl revision in flash_reinforce/REQUIRED_VERL.txt (negative-token filtering is required).")
CHECK

exec "$python_bin" -m verl.trainer.main_ppo \
  --config-path "$recipe_dir/config" --config-name "$setting" "$@"
