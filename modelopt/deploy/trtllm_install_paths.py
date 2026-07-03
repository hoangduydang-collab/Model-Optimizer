# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate installed TensorRT-LLM on disk without importing the package.

Importing ``tensorrt_llm`` during install-time patching can fail (circular init,
``bindings`` not ready). Patch scripts should edit site-packages files directly.
"""

from __future__ import annotations

import site
from pathlib import Path


def find_tensorrt_llm_root() -> Path:
    """Return the ``tensorrt_llm`` package directory under the active venv."""
    search_roots: list[str] = []
    search_roots.extend(site.getsitepackages())
    user = site.getusersitepackages()
    if user:
        search_roots.append(user)

    for root_dir in search_roots:
        candidate = Path(root_dir) / "tensorrt_llm"
        if (candidate / "__init__.py").is_file():
            return candidate

    raise RuntimeError(
        "tensorrt_llm package not found in site-packages. "
        "Install with: uv pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
    )
