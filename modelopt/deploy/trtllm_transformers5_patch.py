# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install-time patches so TensorRT-LLM 1.2.x imports with transformers 5.x.

Upstream TRT-LLM merged this pattern on main (commit 58f7ccb); stable PyPI wheels
(1.2.1, 1.3.0rc*) still hard-import ``AutoModelForVision2Seq``. ModelOpt Qwen3 MoE
requires transformers >= 5.0 (``Qwen3MoeExperts``).

Prefer patching installed ``tensorrt_llm`` at setup time over a runtime alias in
ModelOpt — same fix NVIDIA ships, applied once per venv.
"""

from __future__ import annotations

from pathlib import Path

TRANSFORMERS5_PATCH_MARKER = "# modelopt: transformers5 AutoModelForVision2Seq compat"

_CONVERT_OLD = (
    "from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,\n"
    "                          AutoTokenizer)"
)

_CONVERT_NEW = f"""{TRANSFORMERS5_PATCH_MARKER}
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

_MULTIMODAL_NEW = f"""{TRANSFORMERS5_PATCH_MARKER}
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


def _patch_file(path: Path, old: str, new: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if TRANSFORMERS5_PATCH_MARKER in text or new in text:
        return False
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def apply_trtllm_transformers5_compat_patch() -> list[str]:
    """Patch installed TensorRT-LLM for transformers 5.x import compatibility."""
    from modelopt.deploy.transformers_compat import apply_transformers_compat

    apply_transformers_compat()
    import tensorrt_llm

    root = Path(tensorrt_llm.__file__).resolve().parent
    changed: list[str] = []

    convert_path = root / "models" / "gpt" / "convert.py"
    if _patch_file(convert_path, _CONVERT_OLD, _CONVERT_NEW):
        changed.append(str(convert_path))

    multimodal_path = root / "tools" / "multimodal_builder.py"
    if _patch_file(multimodal_path, _MULTIMODAL_OLD, _MULTIMODAL_NEW):
        changed.append(str(multimodal_path))

    return changed


def trtllm_transformers5_patch_applied() -> bool:
    """Return True if convert.py already contains the install-time compat patch."""
    try:
        import tensorrt_llm
    except ImportError:
        return False
    convert_path = Path(tensorrt_llm.__file__).resolve().parent / "models" / "gpt" / "convert.py"
    if not convert_path.is_file():
        return False
    return TRANSFORMERS5_PATCH_MARKER in convert_path.read_text(encoding="utf-8")


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
