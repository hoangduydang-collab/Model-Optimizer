# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install-time patches so TensorRT-LLM 1.2.x imports with transformers 5.x.

Upstream TRT-LLM merged compat on main (e.g. 58f7ccb, d6dedbd); stable PyPI 1.2.1
still hard-imports removed symbols. ModelOpt Qwen3 MoE needs transformers >= 5.0.

Edits site-packages files directly — does not ``import tensorrt_llm`` during patching.
"""

from __future__ import annotations

import re
from pathlib import Path

from modelopt.deploy.trtllm_install_paths import find_tensorrt_llm_root

TRANSFORMERS5_PATCH_MARKER = "# modelopt: transformers5 compat"

_CONVERT_OLD = (
    "from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,\n"
    "                          AutoTokenizer)"
)

_CONVERT_NEW = f"""{TRANSFORMERS5_PATCH_MARKER} AutoModelForVision2Seq
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

from transformers import AutoModelForCausalLM, AutoTokenizer"""

_MULTIMODAL_OLD = (
    "from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,\n"
    "                          AutoModelForVision2Seq, AutoProcessor,\n"
    "                          Blip2ForConditionalGeneration, Blip2Processor,\n"
    "                          FuyuForCausalLM, FuyuProcessor,\n"
    "                          LlavaForConditionalGeneration, NougatProcessor,\n"
    "                          Pix2StructForConditionalGeneration,\n"
    "                          VisionEncoderDecoderModel, CLIPVisionModel)"
)

_MULTIMODAL_NEW = f"""{TRANSFORMERS5_PATCH_MARKER} AutoModelForVision2Seq
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoProcessor, Blip2ForConditionalGeneration,
                          Blip2Processor, FuyuForCausalLM, FuyuProcessor,
                          LlavaForConditionalGeneration, NougatProcessor,
                          Pix2StructForConditionalGeneration,
                          VisionEncoderDecoderModel, CLIPVisionModel)"""

_GET_PARAMETER_DEVICE_BLOCK = f"""{TRANSFORMERS5_PATCH_MARKER} get_parameter_device
try:
    from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
except ImportError:
    def get_parameter_device(module):
        return next(module.parameters()).device

    def get_parameter_dtype(module):
        return next(module.parameters()).dtype"""

# TRT-LLM 1.2.1 modeling_clip.py layout (parens + newline).
_GET_PARAMETER_DEVICE_PAREN_RE = re.compile(
    r"from transformers\.modeling_utils import\s*\(\s*get_parameter_device,\s*\n\s*get_parameter_dtype\s*\)"
)
_GET_PARAMETER_DEVICE_LINE_RE = re.compile(
    r"from transformers\.modeling_utils import get_parameter_device, get_parameter_dtype"
)

_AUTOCFG_REGISTER_LINE_RE = re.compile(
    r"^(\s*)AutoConfig\.register\(([^)]+)\)\s*$",
    re.MULTILINE,
)


def _patch_file(path: Path, old: str, new: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _patch_get_parameter_device_imports(root: Path) -> list[str]:
    changed: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if TRANSFORMERS5_PATCH_MARKER + " get_parameter_device" in text:
            continue
        if "get_parameter_device" not in text or "transformers.modeling_utils" not in text:
            continue
        new_text, n_paren = _GET_PARAMETER_DEVICE_PAREN_RE.subn(_GET_PARAMETER_DEVICE_BLOCK, text, count=1)
        if n_paren:
            path.write_text(new_text, encoding="utf-8")
            changed.append(str(path))
            continue
        new_text, n_line = _GET_PARAMETER_DEVICE_LINE_RE.subn(_GET_PARAMETER_DEVICE_BLOCK, text, count=1)
        if n_line:
            path.write_text(new_text, encoding="utf-8")
            changed.append(str(path))
    return changed


def _patch_autoconfig_register_exist_ok(root: Path) -> list[str]:
    """Transformers 5.x ships configs TRT-LLM re-registers; need exist_ok=True."""
    marker = TRANSFORMERS5_PATCH_MARKER + " AutoConfig.register exist_ok"
    changed: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if marker in text or "AutoConfig.register(" not in text:
            continue

        def _repl(match: re.Match[str]) -> str:
            indent, args = match.group(1), match.group(2)
            if "exist_ok" in args:
                return match.group(0)
            return (
                f"{indent}{marker}\n"
                f"{indent}AutoConfig.register({args}, exist_ok=True)"
            )

        new_text, count = _AUTOCFG_REGISTER_LINE_RE.subn(_repl, text)
        if count:
            path.write_text(new_text, encoding="utf-8")
            changed.append(str(path))
    return changed


def trtllm_transformers5_patch_applied() -> bool:
    """Return True if convert.py already contains the install-time compat patch."""
    try:
        convert_path = find_tensorrt_llm_root() / "models" / "gpt" / "convert.py"
    except RuntimeError:
        return False
    if not convert_path.is_file():
        return False
    return TRANSFORMERS5_PATCH_MARKER in convert_path.read_text(encoding="utf-8")


def apply_trtllm_transformers5_compat_patch() -> list[str]:
    """Patch installed TensorRT-LLM for transformers 5.x import compatibility."""
    root = find_tensorrt_llm_root()
    changed: list[str] = []

    convert_path = root / "models" / "gpt" / "convert.py"
    if _patch_file(convert_path, _CONVERT_OLD, _CONVERT_NEW):
        changed.append(str(convert_path))

    multimodal_path = root / "tools" / "multimodal_builder.py"
    if _patch_file(multimodal_path, _MULTIMODAL_OLD, _MULTIMODAL_NEW):
        changed.append(str(multimodal_path))

    changed.extend(_patch_get_parameter_device_imports(root))
    changed.extend(_patch_autoconfig_register_exist_ok(root))

    return changed


def main() -> int:
    changed = apply_trtllm_transformers5_compat_patch()
    if changed:
        print("Patched TensorRT-LLM for transformers 5.x:")
        for path in changed:
            print(f"  {path}")
    else:
        print("TensorRT-LLM transformers 5.x compat already applied or anchors not found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
