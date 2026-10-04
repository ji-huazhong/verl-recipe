# Copyright 2026 verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Stateless Python execution through verl's existing Sandbox Fusion client."""

import os

from verl.tools.function_tool import function_tool
from verl.utils.reward_score.sandbox_fusion.utils import call_sandbox_api


@function_tool
def python(code: str) -> str:
    """Execute Python in an isolated sandbox. Each call starts a fresh process; print the result.

    Args:
        code: Python source code, without Markdown fences.
    """
    result, error = call_sandbox_api(
        sandbox_fusion_url=os.environ["SANDBOX_FUSION_URL"],
        code=code,
        stdin=None,
        compile_timeout=10,
        run_timeout=30,
        memory_limit_mb=1024,
        language="python",
    )
    if error:
        return "Sandbox execution failed: " + error
    run = result.get("run_result") or result.get("compile_result") or {}
    return str(run.get("stdout", "")) + str(run.get("stderr", "")) or str(result.get("status", "No output"))
