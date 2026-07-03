#!/usr/bin/env python3
"""TRT-LLM W4A8_CUSTOM helpers for ModelOpt folded-AWQ MoE checkpoints.

ModelOpt W4A8_AWQ export folds AWQ ``pre_quant_scale`` into LayerNorm and writes
HF expert keys (``gate_proj`` / ``up_proj`` / ``down_proj``). TRT-LLM's MoE
W4A8 VANILLA loader still expects separate ``pre_quant_scale`` tensors.

The CUSTOM_W4A8 MoE path (DeepSeek V3 / quantize_mixed_precision_moe.py) matches
folded AWQ math: ``fc31_act_scale = 1 / input_scale`` and no ``pre_quant_scale``.

This module:
1. Patches TRT-LLM to select ``MoEWeightLoadingMode.W4A8_CUSTOM`` for Qwen MoE
   when the model quant config is INT4 block + FP8 (W4A8_AWQ).
2. Rewrites exported safetensors so CUSTOM loaders find ``weight_scale_inv`` and
   fused ``input_scale`` (``input_scale * weight_scale_2``, matching VANILLA alpha).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_MARKER = ".trtllm_w4a8_custom_prepared.json"
_EXPERT_PROJ_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj))\.(?P<suffix>.+)$"
)
_EXPERT_RE = re.compile(r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts\.\d+)\.")


def _is_w4a8_awq_checkpoint(ckpt: Path) -> bool:
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return False
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config")
    if isinstance(qc, str):
        qc = json.loads(qc)
    return isinstance(qc, dict) and qc.get("quant_algo") == "W4A8_AWQ"


def _marker_path(ckpt: Path) -> Path:
    return ckpt / _MARKER


def checkpoint_needs_rewrite(ckpt: Path) -> bool:
    if not _is_w4a8_awq_checkpoint(ckpt):
        return False
    if _marker_path(ckpt).exists():
        return False
    shard_paths = _list_weight_shards(ckpt)
    if not shard_paths:
        return False
    with safe_open(str(shard_paths[0]), framework="pt") as fh:
        keys = list(fh.keys())
    return any("mlp.experts" in k and k.endswith(".input_scale") for k in keys)


def _list_weight_shards(ckpt: Path) -> list[Path]:
    single = ckpt / "model.safetensors"
    if single.exists():
        return [single]
    index_path = ckpt / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as fh:
            index = json.load(fh)
        names = sorted(set(index.get("weight_map", {}).values()))
        return [ckpt / name for name in names if (ckpt / name).exists()]
    return sorted(ckpt.glob("model-*.safetensors"))


def _fuse_input_scale(input_scale: torch.Tensor, weight_scale_2: torch.Tensor) -> torch.Tensor:
    fused = input_scale.float() * weight_scale_2.float()
    return fused.to(dtype=input_scale.dtype)


def _max_tensors(*tensors: torch.Tensor) -> torch.Tensor:
    result = tensors[0]
    for tensor in tensors[1:]:
        result = torch.max(result.float(), tensor.float()).to(dtype=result.dtype)
    return result


def _canonical_gate_up_input_scale(parts: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Fuse gate/up input scales for TRT CUSTOM_W4A8 (matches VANILLA alpha math)."""
    inputs = [parts[k] for k in ("gate_proj.input_scale", "up_proj.input_scale") if k in parts]
    if not inputs:
        return None
    shared_input = _max_tensors(*inputs)
    w2s = [parts[k] for k in ("gate_proj.weight_scale_2", "up_proj.weight_scale_2") if k in parts]
    if not w2s:
        return shared_input
    return _fuse_input_scale(shared_input, _max_tensors(*w2s))


def _canonical_down_input_scale(parts: dict[str, torch.Tensor]) -> torch.Tensor | None:
    down_in = parts.get("down_proj.input_scale")
    if down_in is None:
        return None
    down_w2 = parts.get("down_proj.weight_scale_2")
    if down_w2 is None:
        return down_in
    return _fuse_input_scale(down_in, down_w2)


def _gather_expert_scales(shard_paths: list[Path]) -> dict[str, dict[str, torch.Tensor]]:
    """Collect per-expert scale tensors across all shards (scales are tiny)."""
    tracked_suffixes = {
        "gate_proj.input_scale",
        "up_proj.input_scale",
        "gate_proj.weight_scale_2",
        "up_proj.weight_scale_2",
        "down_proj.input_scale",
        "down_proj.weight_scale_2",
    }
    gathered: dict[str, dict[str, torch.Tensor]] = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as fh:
            for key in fh.keys():  # noqa: SIM118
                match = _EXPERT_RE.match(key)
                if match is None:
                    continue
                suffix = key[len(match.group("prefix")) + 1 :]
                if suffix not in tracked_suffixes:
                    continue
                gathered.setdefault(match.group("prefix"), {})[suffix] = fh.get_tensor(key).clone()
    return gathered


def _build_canonical_input_scales(
    gathered: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    canonical: dict[str, dict[str, torch.Tensor]] = {}
    for expert_prefix, parts in gathered.items():
        entry: dict[str, torch.Tensor] = {}
        gate_up = _canonical_gate_up_input_scale(parts)
        if gate_up is not None:
            entry["gate_proj.input_scale"] = gate_up
            entry["up_proj.input_scale"] = gate_up
        down = _canonical_down_input_scale(parts)
        if down is not None:
            entry["down_proj.input_scale"] = down
        if entry:
            canonical[expert_prefix] = entry
    return canonical


def _rewrite_state_dict(
    sd: dict[str, torch.Tensor],
    canonical_input_scales: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], int]:
    proj_prefixes: set[str] = set()
    for key in sd:
        match = _EXPERT_PROJ_RE.match(key)
        if match:
            proj_prefixes.add(match.group("prefix"))

    changed = 0
    out = dict(sd)
    drop_keys: list[str] = []
    add_keys: dict[str, torch.Tensor] = {}

    for expert_prefix, canon_parts in canonical_input_scales.items():
        for suffix, value in canon_parts.items():
            key = f"{expert_prefix}.{suffix}"
            if key not in out:
                continue
            canonical = value.clone()
            if not torch.equal(out[key], canonical):
                out[key] = canonical
                changed += 1
        for w2_suffix in ("gate_proj.weight_scale_2", "up_proj.weight_scale_2", "down_proj.weight_scale_2"):
            w2_key = f"{expert_prefix}.{w2_suffix}"
            if w2_key in out:
                drop_keys.append(w2_key)

    for prefix in sorted(proj_prefixes):
        ws_key = f"{prefix}.weight_scale"
        ws_inv_key = f"{prefix}.weight_scale_inv"
        pqs_key = f"{prefix}.pre_quant_scale"

        if ws_key in out and ws_inv_key not in out:
            add_keys[ws_inv_key] = out[ws_key].clone()
            changed += 1

        if pqs_key in out:
            drop_keys.append(pqs_key)
            changed += 1

    for key in drop_keys:
        if key in out:
            del out[key]
            changed += 1

    out.update(add_keys)
    return out, changed


def rewrite_checkpoint_for_trtllm_w4a8_custom(
    ckpt: Path,
    *,
    force: bool = False,
) -> bool:
    """Rewrite MoE expert scale tensors for TRT-LLM CUSTOM_W4A8 loading.

    Returns True if the checkpoint was modified (or already prepared).
    """
    ckpt = ckpt.resolve()
    if not _is_w4a8_awq_checkpoint(ckpt):
        return False

    marker = _marker_path(ckpt)
    if marker.exists() and not force:
        return True

    shard_paths = _list_weight_shards(ckpt)
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors weights found under {ckpt}")

    index_path = ckpt / "model.safetensors.index.json"
    index: dict | None = None
    if index_path.exists():
        with index_path.open(encoding="utf-8") as fh:
            index = json.load(fh)
    weight_map: dict[str, str] = dict((index or {}).get("weight_map") or {})

    gathered = _gather_expert_scales(shard_paths)
    canonical_input_scales = _build_canonical_input_scales(gathered)

    total_changed = 0
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as fh:
            metadata = dict(fh.metadata() or {})
            sd = {k: fh.get_tensor(k).clone() for k in fh.keys()}  # noqa: SIM118

        new_sd, changed = _rewrite_state_dict(sd, canonical_input_scales)
        if changed:
            save_file(new_sd, str(shard_path), metadata=metadata)
            total_changed += changed

            if weight_map:
                shard_name = shard_path.name
                for key in new_sd:
                    if key not in sd and key.endswith(".weight_scale_inv"):
                        weight_map[key] = shard_name
                for key in list(sd):
                    if key not in new_sd and key.endswith(".weight_scale_2"):
                        weight_map.pop(key, None)
                    if key not in new_sd and key.endswith(".pre_quant_scale"):
                        weight_map.pop(key, None)

    if index is not None:
        index["weight_map"] = weight_map
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)

    marker.write_text(
        json.dumps(
            {
                "format": "trtllm_w4a8_custom",
                "changed_tensors": total_changed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def apply_trtllm_w4a8_custom_patches() -> None:
    """Select W4A8_CUSTOM MoE loading for Qwen MoE + ModelOpt W4A8_AWQ."""
    try:
        import tensorrt_llm._torch.modules.fused_moe as fused_moe_mod
        from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
    except ImportError as exc:
        raise ImportError(
            "tensorrt_llm is required for W4A8_CUSTOM patches"
        ) from exc

    if getattr(fused_moe_mod, "_modelopt_w4a8_custom_patched", False):
        return

    _orig_create_moe = fused_moe_mod.create_moe

    def create_moe(*args, model_config=None, weight_loading_mode=MoEWeightLoadingMode.VANILLA, **kwargs):
        if weight_loading_mode == MoEWeightLoadingMode.VANILLA and model_config is not None:
            cfg = getattr(model_config, "pretrained_config", None)
            model_type = getattr(cfg, "model_type", None) if cfg is not None else None
            if model_type in ("qwen3_moe", "qwen2_moe"):
                qc = getattr(model_config, "quant_config", None)
                if (
                    qc is not None
                    and hasattr(qc.layer_quant_mode, "is_int4_weight_only_per_group")
                    and qc.layer_quant_mode.is_int4_weight_only_per_group()
                ):
                    weight_loading_mode = MoEWeightLoadingMode.W4A8_CUSTOM
        return _orig_create_moe(
            *args,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            **kwargs,
        )

    fused_moe_mod.create_moe = create_moe
    fused_moe_mod._modelopt_w4a8_custom_patched = True


def prepare_checkpoint_and_runtime(ckpt: Path, *, force_rewrite: bool = False) -> None:
    """Rewrite checkpoint if needed. Call apply_trtllm_w4a8_custom_patches() after tensorrt_llm imports."""
    if _is_w4a8_awq_checkpoint(ckpt):
        rewrite_checkpoint_for_trtllm_w4a8_custom(ckpt, force=force_rewrite)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Rewrite ModelOpt W4A8_AWQ MoE checkpoint for TRT-LLM CUSTOM_W4A8"
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Rewrite even if marker exists")
    args = parser.parse_args(argv)

    if not _is_w4a8_awq_checkpoint(args.checkpoint_dir):
        print(f"SKIP: not a W4A8_AWQ checkpoint: {args.checkpoint_dir}", file=sys.stderr)
        return 1

    rewrite_checkpoint_for_trtllm_w4a8_custom(args.checkpoint_dir, force=args.force)
    print(f"Prepared {args.checkpoint_dir} for TRT-LLM W4A8_CUSTOM MoE loading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
