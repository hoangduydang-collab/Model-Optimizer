#!/usr/bin/env python3
"""Gate C: load a ModelOpt unified HF checkpoint in TensorRT-LLM and sample-generate.

Wraps ``modelopt.deploy.llm.LLM`` (PyTorch backend). Exits 0 on sane output, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TensorRT-LLM deploy smoke test for ModelOpt checkpoints")
    p.add_argument("--checkpoint_dir", required=True, help="exported unified HF checkpoint dir")
    p.add_argument("--tp", type=int, default=2, help="tensor parallel size (0 = all GPUs)")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--no-trust_remote_code", dest="trust_remote_code", action="store_false")
    p.add_argument("--max_batch_size", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=4096, help="cap context for smoke test")
    return p.parse_args()


def _load_quant_summary(ckpt: Path) -> dict:
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return {"error": f"missing {cfg_path}"}
    with cfg_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    qc = data.get("quantization_config")
    if isinstance(qc, str):
        qc = json.loads(qc)
    if not isinstance(qc, dict):
        return {"quantization_config": None}
    return {
        "quant_method": qc.get("quant_method"),
        "quant_algo": qc.get("quant_algo"),
        "ignore_count": len(qc.get("ignore") or []),
        "config_group_keys": sorted((qc.get("config_groups") or {}).keys()),
    }


def main() -> int:
    args = _parse_args()
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.is_dir():
        print(f"ERROR: checkpoint_dir not found: {ckpt}", file=sys.stderr)
        return 1

    print("=== Gate C: TensorRT-LLM deploy ===")
    print(f"checkpoint: {ckpt}")
    print(f"tp={args.tp} prompt={args.prompt!r}")
    print("quant summary:", json.dumps(_load_quant_summary(ckpt), indent=2))

    from trtllm_w4a8_moe_custom import (
        apply_trtllm_w4a8_custom_patches,
        prepare_checkpoint_and_runtime,
    )

    try:
        prepare_checkpoint_and_runtime(ckpt)
    except Exception as exc:
        print(f"FAIL: checkpoint prep raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        import torch
    except ImportError as exc:
        print(f"ERROR: torch not installed: {exc}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for TensorRT-LLM deploy", file=sys.stderr)
        return 1

    try:
        from modelopt.deploy.llm import LLM
    except ImportError as exc:
        print(
            "ERROR: cannot import modelopt.deploy.llm (need tensorrt_llm + mpi4py):",
            exc,
            file=sys.stderr,
        )
        print(
            "Hint: source modelopt-test/_env.sh (sets LD_LIBRARY_PATH + Open MPI localhost).",
            file=sys.stderr,
        )
        return 1

    try:
        apply_trtllm_w4a8_custom_patches()
        print("TRT-LLM W4A8_CUSTOM MoE runtime patch applied")
    except ImportError as exc:
        print(f"FAIL: W4A8_CUSTOM runtime patch: {exc}", file=sys.stderr)
        return 1

    try:
        llm = LLM(
            str(ckpt),
            tp=args.tp,
            trust_remote_code=args.trust_remote_code,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            enable_kv_cache_reuse=False,
        )
    except Exception as exc:
        print(f"FAIL: LLM init raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        outputs = llm.generate_text([args.prompt], max_new_tokens=args.max_new_tokens, temperature=0.0)
    except Exception as exc:
        print(f"FAIL: generate_text raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    text = outputs[0] if outputs else ""
    full = args.prompt + text
    sane = bool(text and text.strip())

    print("\n========== TRT-LLM OUTPUT ==========")
    print(f"prompt: {args.prompt!r}")
    print(f"completion: {text!r}")
    print(f"full: {full!r}")
    print(f"sane_output: {sane}")
    print("====================================\n")

    if not sane:
        print("FAIL: empty or whitespace-only completion", file=sys.stderr)
        return 1

    print("PASS: TensorRT-LLM loaded checkpoint and returned sane text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
