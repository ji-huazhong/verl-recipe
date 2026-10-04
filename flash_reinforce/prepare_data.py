# Copyright 2026 verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare explicit benchmark splits and one-prompt-per-trajectory parquet data."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

MATH_INSTRUCTION = r"Reason step by step and put the final answer in \boxed{}."
TOOL_INSTRUCTION = (
    r"Solve the math problem. You may use the python tool to check calculations. "
    r"Each call runs independently; print the result. Commit your final answer in \boxed{} in your final reply."
)
ALFWORLD_INSTRUCTION = (
    "Complete the household task described by the environment. You may reason briefly, then issue exactly one "
    "TextWorld command inside <action>...</action>. For example: <action>go to countertop 1</action>. "
    "Use commands such as look, inventory, go to, open, close, take ... from ..., put ... in/on ..., "
    "clean ... with ..., heat ... with ..., cool ... with ..., or use .... Do not invent observations."
)


def load_rows(source, split="train", revision=None):
    """Read existing parquet / datasets.save_to_disk output, or stream a Hugging Face split."""
    from datasets import DatasetDict, load_dataset, load_from_disk

    path = Path(source).expanduser()
    if path.is_file():
        return load_dataset("parquet", data_files=str(path), split="train", streaming=True)
    if path.is_dir():
        loaded = load_from_disk(str(path))
        return loaded[split] if isinstance(loaded, DatasetDict) else loaded
    return load_dataset(source, split=split, revision=revision, streaming=True)


def math_row(row, index, split, benchmark=None, tool=False):
    """Normalize the existing verl schema or explicit problem/answer pairs without using solutions as labels."""
    prompt = row.get("prompt")
    if prompt is None:
        question = row.get("problem", row.get("question", row.get("Problem")))
        if not isinstance(question, str):
            raise ValueError("Expected chat-style prompt or a problem/question/Problem string")
        prompt = [{"role": "user", "content": question + "\n\n" + MATH_INSTRUCTION}]
    if not isinstance(prompt, list) or not prompt or any(m["role"] not in {"user", "system"} for m in prompt):
        raise ValueError("Prompts must contain only system/user messages, without reference solutions")
    prompt = [dict(message) for message in prompt]
    if tool:
        # Keep the original question and system instructions, and append an explicit tool contract.
        prompt[-1]["content"] += "\n\n" + TOOL_INSTRUCTION
    reward_model = row.get("reward_model") or {}
    answer = reward_model.get("ground_truth", row.get("answer", row.get("Answer")))
    if answer is None:
        raise ValueError("Expected reward_model.ground_truth or an answer/Answer field")
    source = benchmark or row.get("data_source", "math")
    return {
        "prompt": prompt,
        "data_source": source,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"index": index, "split": split, "source": row.get("data_source", source)},
    }


def unique_math_rows(rows, limit, tool=False):
    """Retain first occurrences of complete prompts, matching the public settings' deduplication intent."""
    seen, result = set(), []
    for row in rows:
        item = math_row(row, len(result), "train", tool=tool)
        key = json.dumps(item["prompt"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) == limit:
            return result
    raise ValueError(f"Expected {limit} unique training prompts, found {len(result)}")


def alfworld_rows(data_root, split, limit=None, manifest=None):
    """Select solvable games from an explicit manifest, or a deterministic sorted subset."""
    root = Path(data_root).expanduser().resolve()
    allowed = {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
    if manifest:
        files = [root / line.strip() for line in Path(manifest).read_text().splitlines() if line.strip()]
    else:
        files = sorted((root / "json_2.1.1" / split).rglob("game.tw-pddl"))
    result = []
    for game in files:
        relative = game.resolve().relative_to(root)
        if relative.parts[:2] != ("json_2.1.1", split):
            raise ValueError(f"Game {relative} does not belong to split {split}")
        if "movable" in str(game) or "Sliced" in str(game):
            continue
        task = json.loads(game.with_name("traj_data.json").read_text())["task_type"]
        if task not in allowed or not json.loads(game.read_text()).get("solvable", False):
            continue
        if any(item["extra_info"]["game_file"] == str(relative) for item in result):
            raise ValueError(f"Duplicate game in manifest: {relative}")
        result.append(
            {
                "prompt": [{"role": "user", "content": ALFWORLD_INSTRUCTION}],
                "data_source": f"alfworld_{split}",
                "ability": "household",
                "reward_model": {"style": "rule", "ground_truth": "1"},
                "extra_info": {"index": len(result), "split": split, "game_file": str(relative)},
            }
        )
        if limit is not None and len(result) == limit:
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["r1", "math", "tool", "alfworld"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-source", help="Prepared parquet, save_to_disk directory, or HF dataset ID")
    parser.add_argument("--train-revision", help="Optional HF revision for the training source")
    parser.add_argument(
        "--eval-source",
        action="append",
        default=[],
        metavar="BENCHMARK=PARQUET",
        help="Repeat for AMC23, Minerva, AIME25; optional math500 and olympiadbench",
    )
    parser.add_argument("--alfworld-data", default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument("--train-manifest", help="ALFWorld game paths relative to ALFWORLD_DATA, one per line")
    args = parser.parse_args()
    if args.task == "alfworld":
        if not args.alfworld_data:
            parser.error("ALFWORLD_DATA or --alfworld-data is required")
        train = alfworld_rows(args.alfworld_data, "train", 1024, args.train_manifest)
        seen = alfworld_rows(args.alfworld_data, "valid_seen")
        unseen = alfworld_rows(args.alfworld_data, "valid_unseen")
        if (len(train), len(seen), len(unseen)) != (1024, 140, 134):
            raise ValueError(f"Expected 1024/140/134 games; got {len(train)}/{len(seen)}/{len(unseen)}")
        val = seen + unseen
    else:
        source = args.train_source or (
            "sail/Sanity-Test-R1D-1.5B" if args.task == "r1" else "BytedTsinghua-SIA/DAPO-Math-17k"
        )
        train = unique_math_rows(
            load_rows(source, revision=args.train_revision),
            1460 if args.task == "r1" else 7500,
            tool=args.task == "tool",
        )
        if args.task == "r1" and not args.eval_source:
            val = [math_row(row, i, "val") for i, row in enumerate(load_rows("sail/Sanity-Test-R1D-1.5B", "test"))]
            if len(val) != 60 or sorted(Counter(row["data_source"] for row in val).values()) != [30, 30]:
                raise ValueError(
                    "R1 evaluation requires 30 AIME24 and 30 AIME25 problems with distinct data_source labels"
                )
        else:
            sources = dict(item.split("=", 1) for item in args.eval_source)
            required = {"amc23", "minerva", "aime25"} if args.task != "r1" else {"aime24", "aime25"}
            if not required <= sources.keys():
                parser.error(f"Provide --eval-source entries for {sorted(required)}")
            val = [
                math_row(row, i, "val", benchmark=name, tool=args.task == "tool")
                for name, source in sources.items()
                for i, row in enumerate(load_rows(source))
            ]
    from datasets import Dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": args.task, "arguments": {key: str(value) for key, value in vars(args).items()}, "files": {}}
    for split, rows in (("train", train), ("val", val)):
        path = args.output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(path)
        manifest["files"][split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.task == "alfworld":
        for name, rows in (("train", train), ("val", val)):
            (args.output_dir / f"{name}_games.txt").write_text(
                "\n".join(row["extra_info"]["game_file"] for row in rows) + "\n"
            )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
