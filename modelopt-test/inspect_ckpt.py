#!/usr/bin/env python3
"""Gate A sanity: summarize ModelOpt export quantization metadata (no GPU)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _load_quant_config(ckpt: Path) -> dict | None:
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: missing {cfg_path}", file=sys.stderr)
        return None
    with cfg_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    qc = data.get("quantization_config")
    if isinstance(qc, str):
        qc = json.loads(qc)
    return qc if isinstance(qc, dict) else None


def _bits_summary(qc: dict) -> list[str]:
    lines: list[str] = []
    groups = qc.get("config_groups") or {}
    for name, group in groups.items():
        w = group.get("weights") or {}
        a = group.get("input_activations") or {}
        w_bits = w.get("num_bits")
        a_bits = a.get("num_bits")
        gs = w.get("group_size") or a.get("group_size")
        lines.append(
            f"  {name}: weights={w_bits}-bit {w.get('type')} "
            f"acts={a_bits}-bit {a.get('type')} group_size={gs}"
        )
    return lines


def _algo_implies_bits(algo: str) -> tuple[bool, bool]:
    """Return (w4, fp8_activations) implied by top-level quant_algo when config_groups absent."""
    w4_algos = {
        "W4A8_AWQ",
        "NVFP4_AWQ",
        "W4A16_AWQ",
        "W4A16_NVFP4",
        "W4A8_NVFP4_FP8",
        "W4A8_MXFP4_FP8",
    }
    fp8_act_algos = {"W4A8_AWQ", "NVFP4_AWQ", "W4A8_NVFP4_FP8", "W4A8_MXFP4_FP8"}
    return algo in w4_algos, algo in fp8_act_algos


def _weight_shards_present(ckpt: Path) -> bool:
    if (ckpt / "model.safetensors").exists():
        return True
    index = ckpt / "model.safetensors.index.json"
    if index.exists():
        return True
    return any(ckpt.glob("model-*.safetensors"))

    patterns = {
        "lm_head": r"lm_head",
        "moe_gate": r"mlp\.gate|router|block_sparse_moe\.gate",
    }
    joined = "\n".join(str(x) for x in ignore)
    return {k: bool(re.search(p, joined)) for k, p in patterns.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ModelOpt exported checkpoint")
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()

    ckpt = args.checkpoint_dir
    if not ckpt.is_dir():
        print(f"ERROR: not a directory: {ckpt}", file=sys.stderr)
        return 1

    qc = _load_quant_config(ckpt)
    if qc is None:
        print("ERROR: no quantization_config in config.json", file=sys.stderr)
        return 1

    ignore = list(qc.get("ignore") or [])
    hits = _ignore_hits(ignore)

    print("=== Checkpoint quant inspection ===")
    print(f"path: {ckpt}")
    print(f"quant_method: {qc.get('quant_method')}")
    print(f"quant_algo:   {qc.get('quant_algo')}")
    print(f"ignore entries: {len(ignore)}")
    print(f"expected skips present: {hits}")
    print("config_groups:")
    group_lines = _bits_summary(qc)
    if group_lines:
        for line in group_lines:
            print(line)
    else:
        print("  (empty — ModelOpt omits config_groups for uniform W4A8_AWQ; see quant_algo)")

    algo = str(qc.get("quant_algo", ""))
    w4 = any("weights=4-bit" in ln for ln in group_lines)
    fp8_act = any("acts=8-bit float" in ln for ln in group_lines)
    if not group_lines:
        w4_algo, fp8_algo = _algo_implies_bits(algo)
        w4 = w4 or w4_algo
        fp8_act = fp8_act or fp8_algo

    has_weights = _weight_shards_present(ckpt)
    print(f"weight_shards: {has_weights}")

    if algo in ("W4A8_AWQ", "NVFP4_AWQ"):
        ok = w4 and fp8_act and has_weights
    elif algo in ("W4A16_AWQ",):
        ok = w4 and has_weights
    else:
        ok = w4 and has_weights

    if not hits.get("lm_head"):
        print("WARN: lm_head not found in ignore list")
        ok = False
    if not hits.get("moe_gate"):
        print("WARN: MoE gate/router not found in ignore list")
        ok = False

    print(f"\ninspect_ok: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
