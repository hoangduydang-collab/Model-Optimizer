# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM W4A8_CUSTOM layout for folded ModelOpt AWQ MoE checkpoints.

ModelOpt W4A8_AWQ export folds AWQ ``pre_quant_scale`` into LayerNorm. TRT-LLM's
``MoEWeightLoadingMode.W4A8_CUSTOM`` path expects ``weight_scale_inv``, tied
gate/up ``input_scale`` (activation only), and down ``input_scale`` with
``weight_scale_2`` folded into it. Gate/up ``weight_scale_2`` is unused by CUSTOM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_MARKER = ".trtllm_w4a8_custom_prepared.json"
_MARKER_VERSION = "trtllm_w4a8_custom_v2"
_EXPERT_PROJ_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj))\.(?P<suffix>.+)$"
)
_EXPERT_RE = re.compile(r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts\.\d+)\.")


def state_dict_has_moe_experts(state_dict: dict[str, Any]) -> bool:
    return any("mlp.experts" in key for key in state_dict)


def _fuse_input_scale(input_scale: torch.Tensor, weight_scale_2: torch.Tensor) -> torch.Tensor:
    fused = input_scale.float() * weight_scale_2.float()
    return fused.to(dtype=input_scale.dtype)


def _max_tensors(*tensors: torch.Tensor) -> torch.Tensor:
    result = tensors[0]
    for tensor in tensors[1:]:
        result = torch.max(result.float(), tensor.float()).to(dtype=result.dtype)
    return result


def _canonical_gate_up_input_scale(parts: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Tie gate/up activation scales only (CUSTOM fc31_alpha = input_scale, not * weight_scale_2)."""
    inputs = [parts[k] for k in ("gate_proj.input_scale", "up_proj.input_scale") if k in parts]
    if not inputs:
        return None
    return _max_tensors(*inputs)


def _canonical_down_input_scale(parts: dict[str, torch.Tensor]) -> torch.Tensor | None:
    down_in = parts.get("down_proj.input_scale")
    if down_in is None:
        return None
    down_w2 = parts.get("down_proj.weight_scale_2")
    if down_w2 is None:
        return down_in
    return _fuse_input_scale(down_in, down_w2)


def _gather_expert_scales_from_mapping(
    tensors: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    tracked_suffixes = {
        "gate_proj.input_scale",
        "up_proj.input_scale",
        "gate_proj.weight_scale_2",
        "up_proj.weight_scale_2",
        "down_proj.input_scale",
        "down_proj.weight_scale_2",
    }
    gathered: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        match = _EXPERT_RE.match(key)
        if match is None:
            continue
        suffix = key[len(match.group("prefix")) + 1 :]
        if suffix not in tracked_suffixes:
            continue
        gathered.setdefault(match.group("prefix"), {})[suffix] = value.clone()
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


def postprocess_state_dict_for_trtllm_w4a8_moe(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rewrite MoE expert scales in-memory for TRT-LLM ``W4A8_CUSTOM`` loading."""
    if not state_dict_has_moe_experts(state_dict):
        return state_dict

    gathered = _gather_expert_scales_from_mapping(state_dict)
    canonical_input_scales = _build_canonical_input_scales(gathered)
    new_sd, _ = _rewrite_state_dict(state_dict, canonical_input_scales)
    return new_sd


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
        for w2_suffix in (
            "gate_proj.weight_scale_2",
            "up_proj.weight_scale_2",
            "down_proj.weight_scale_2",
        ):
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


def rewrite_checkpoint_for_trtllm_w4a8_custom(
    ckpt: Path,
    *,
    force: bool = False,
) -> bool:
    """Rewrite on-disk safetensors for legacy exports missing TRT-LLM scale layout."""
    ckpt = ckpt.resolve()
    if not _is_w4a8_awq_checkpoint(ckpt):
        return False

    marker = ckpt / _MARKER
    if marker.exists() and not force:
        try:
            with marker.open(encoding="utf-8") as fh:
                if json.load(fh).get("format") == _MARKER_VERSION:
                    return True
        except (json.JSONDecodeError, OSError):
            pass

    shard_paths = _list_weight_shards(ckpt)
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors weights found under {ckpt}")

    index_path = ckpt / "model.safetensors.index.json"
    index: dict | None = None
    if index_path.exists():
        with index_path.open(encoding="utf-8") as fh:
            index = json.load(fh)
    weight_map: dict[str, str] = dict((index or {}).get("weight_map") or {})

    gathered: dict[str, dict[str, torch.Tensor]] = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as fh:
            for key in fh.keys():  # noqa: SIM118
                match = _EXPERT_RE.match(key)
                if match is None:
                    continue
                suffix = key[len(match.group("prefix")) + 1 :]
                if suffix not in {
                    "gate_proj.input_scale",
                    "up_proj.input_scale",
                    "gate_proj.weight_scale_2",
                    "up_proj.weight_scale_2",
                    "down_proj.input_scale",
                    "down_proj.weight_scale_2",
                }:
                    continue
                gathered.setdefault(match.group("prefix"), {})[suffix] = fh.get_tensor(key).clone()

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
                "format": _MARKER_VERSION,
                "changed_tensors": total_changed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return True
