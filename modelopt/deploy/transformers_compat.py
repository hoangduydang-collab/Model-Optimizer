# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime fallback when TRT-LLM was not patched for transformers 5.x at install time.

Prefer :mod:`trtllm_transformers5_patch` (install-time, permanent). This module only
registers aliases on the live ``transformers`` package if import would still fail.
"""

from __future__ import annotations


def apply_transformers_compat() -> list[str]:
    """Register transformers 5.x aliases if still missing. Idempotent."""
    from modelopt.deploy.trtllm_transformers5_patch import trtllm_transformers5_patch_applied

    if trtllm_transformers5_patch_applied():
        return []

    import transformers

    applied: list[str] = []

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        from transformers import AutoModelForImageTextToText

        transformers.AutoModelForVision2Seq = AutoModelForImageTextToText
        applied.append("runtime: AutoModelForVision2Seq -> AutoModelForImageTextToText")

    if not hasattr(transformers, "HybridCache"):
        try:
            from transformers.cache_utils import HybridCache
        except ImportError:
            pass
        else:
            transformers.HybridCache = HybridCache
            applied.append("runtime: HybridCache from transformers.cache_utils")

    return applied
