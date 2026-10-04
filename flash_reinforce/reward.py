# Copyright 2026 verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Binary math rewards using verl's existing boxed-answer verifier."""

from verl.utils.reward_score.math_dapo import compute_score as math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Grade the committed final answer; never grade a tool observation or shaping bonus."""
    if data_source.startswith("alfworld"):
        raise ValueError("ALFWorld reward must come from the environment, not a text reward function")
    extra_info = extra_info or {}
    answer = extra_info.get("final_answer", solution_str)
    result = math_score(answer, str(ground_truth), strict_box_verify=True)
    return {"score": float(result["acc"]), "acc": float(result["acc"]), "pred": result["pred"] or ""}
