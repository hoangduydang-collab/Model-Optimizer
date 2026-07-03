# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install-time patches for TensorRT-LLM Qwen MoE W4A8_CUSTOM loading.

Mirrors ``modeling_deepseekv3.py`` until TRT-LLM selects ``W4A8_CUSTOM`` for
ModelOpt folded-AWQ Qwen MoE checkpoints natively. Idempotent and safe to re-run.
"""

from __future__ import annotations

from pathlib import Path

PATCH_MARKER = "# modelopt: W4A8_CUSTOM qwen moe"
WEIGHT_PATCH_MARKER = "# modelopt: W4A8_CUSTOM hf export uses VANILLA weight layout"


def _replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(
            f"TRT-LLM patch anchor not found in {path}. "
            "Installed tensorrt_llm version may differ from the supported 1.2.x layout."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _qwen3_helper_block() -> str:
    return f'''
    {PATCH_MARKER}
    @staticmethod
    def _get_experts_quant_config(model_config, layer_idx: int | None):
        if getattr(model_config, "quant_config_dict", None) is None:
            return model_config.quant_config
        if layer_idx is None:
            return model_config.quant_config
        return model_config.quant_config_dict.get(
            f"model.layers.{{layer_idx}}.mlp.experts", model_config.quant_config
        )
'''


def _qwen2_helper_block() -> str:
    return _qwen3_helper_block()


def _patch_qwen3_moe(path: Path) -> bool:
    changed = False
    changed |= _replace_once(
        path,
        (
            "            weight_loading_mode=self.weight_loading_mode,\n"
            "        )\n"
            "\n"
            "    def forward("
        ),
        (
            "            weight_loading_mode=self.weight_loading_mode,\n"
            "        )\n"
            + _qwen3_helper_block()
            + "\n"
            "    def forward("
        ),
    )
    changed |= _replace_once(
        path,
        (
            "        self.weight_loading_mode = MoEWeightLoadingMode.FUSED_GATE_UP_PROJ if config.model_type == "
            '"qwen3_vl_moe_text" else MoEWeightLoadingMode.VANILLA\n'
            "        self.experts = create_moe("
        ),
        (
            "        if config.model_type == \"qwen3_vl_moe_text\":\n"
            "            self.weight_loading_mode = MoEWeightLoadingMode.FUSED_GATE_UP_PROJ\n"
            "        elif self._get_experts_quant_config(\n"
            "            model_config, layer_idx\n"
            "        ).layer_quant_mode.is_int4_weight_only_per_group():\n"
            "            self.weight_loading_mode = MoEWeightLoadingMode.W4A8_CUSTOM\n"
            "        else:\n"
            "            self.weight_loading_mode = MoEWeightLoadingMode.VANILLA\n"
            "        self.experts = create_moe("
        ),
    )
    return changed


def _patch_qwen2_moe(path: Path) -> bool:
    changed = False
    changed |= _replace_once(
        path,
        "from ..modules.fused_moe import DefaultMoeRoutingMethod, create_moe\n",
        (
            "from ..modules.fused_moe import DefaultMoeRoutingMethod, create_moe\n"
            "from ..modules.fused_moe.interface import MoEWeightLoadingMode\n"
        ),
    )
    changed |= _replace_once(
        path,
        (
            "            quant_config=None)\n"
            "\n"
            "    def forward("
        ),
        (
            "            quant_config=None)\n"
            + _qwen2_helper_block()
            + "\n"
            "    def forward("
        ),
    )
    changed |= _replace_once(
        path,
        (
            "        self.experts = create_moe(\n"
            "            num_experts=self.num_experts,\n"
            "            routing_method=DefaultMoeRoutingMethod(top_k=self.top_k),\n"
            "            hidden_size=self.hidden_dim,\n"
            "            intermediate_size=self.moe_intermediate_size,\n"
            "            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},\n"
            "            dtype=config.torch_dtype,\n"
            "            reduce_results=reduce_results,\n"
            "            model_config=model_config,\n"
            "            layer_idx=layer_idx)"
        ),
        (
            "        weight_loading_mode = (\n"
            "            MoEWeightLoadingMode.W4A8_CUSTOM\n"
            "            if self._get_experts_quant_config(\n"
            "                model_config, layer_idx\n"
            "            ).layer_quant_mode.is_int4_weight_only_per_group()\n"
            "            else MoEWeightLoadingMode.VANILLA\n"
            "        )\n"
            "        self.experts = create_moe(\n"
            "            num_experts=self.num_experts,\n"
            "            routing_method=DefaultMoeRoutingMethod(top_k=self.top_k),\n"
            "            hidden_size=self.hidden_dim,\n"
            "            intermediate_size=self.moe_intermediate_size,\n"
            "            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},\n"
            "            dtype=config.torch_dtype,\n"
            "            reduce_results=reduce_results,\n"
            "            model_config=model_config,\n"
            "            layer_idx=layer_idx,\n"
            "            weight_loading_mode=weight_loading_mode,\n"
            "        )"
        ),
    )
    return changed


def _patch_w4a8_custom_weight_layout(path: Path) -> bool:
    """ModelOpt HF export needs VANILLA int4 repack; DeepSeek CUSTOM uses pre-packed weights."""
    text = path.read_text(encoding="utf-8")
    if WEIGHT_PATCH_MARKER in text:
        return False

    w31_old = (
        "        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:\n"
        "            pass\n"
        "        else:\n"
        "            raise NotImplementedError(\n"
        "                f\"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}.\"\n"
        "            )\n"
        "\n"
        "        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),\n"
    )
    w31_new = (
        "        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:\n"
        f"            {WEIGHT_PATCH_MARKER}\n"
        "            transposed = w31_weight_shard.cpu().T.contiguous()\n"
        "            unpacked = unpacker(transposed.view(torch.int8))\n"
        "            transposed = unpacked.T.contiguous()\n"
        "            w31_weight_shard = packer(transposed)\n"
        "        else:\n"
        "            raise NotImplementedError(\n"
        "                f\"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}.\"\n"
        "            )\n"
        "\n"
        "        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),\n"
    )
    w2_old = (
        "        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:\n"
        "            pass\n"
        "        else:\n"
        "            raise NotImplementedError(\n"
        "                f\"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}.\"\n"
        "            )\n"
        "        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),\n"
    )
    w2_new = (
        "        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:\n"
        f"            {WEIGHT_PATCH_MARKER}\n"
        "            transposed = w2_weight_shard.cpu().T.contiguous()\n"
        "            unpacked = unpacker(transposed.view(torch.int8))\n"
        "            transposed = unpacked.T.contiguous()\n"
        "            w2_weight_shard = packer(transposed)\n"
        "        else:\n"
        "            raise NotImplementedError(\n"
        "                f\"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}.\"\n"
        "            )\n"
        "        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),\n"
    )
    if w31_old not in text:
        raise RuntimeError(
            f"TRT-LLM W4A8_CUSTOM weight-layout anchor not found in {path}."
        )
    if w2_old not in text:
        raise RuntimeError(
            f"TRT-LLM W4A8_CUSTOM w2 weight-layout anchor not found in {path}."
        )
    text = text.replace(w31_old, w31_new, 1)
    text = text.replace(w2_old, w2_new, 1)
    path.write_text(text, encoding="utf-8")
    return True


def apply_trtllm_qwen_moe_patches() -> list[str]:
    """Patch installed TensorRT-LLM to use W4A8_CUSTOM for folded-AWQ Qwen MoE."""
    from modelopt.deploy.trtllm_install_paths import find_tensorrt_llm_root

    root = find_tensorrt_llm_root()
    changed: list[str] = []

    qwen3_path = root / "_torch" / "models" / "modeling_qwen3_moe.py"
    if qwen3_path.exists() and _patch_qwen3_moe(qwen3_path):
        changed.append(str(qwen3_path))

    qwen2_path = root / "_torch" / "models" / "modeling_qwen_moe.py"
    if qwen2_path.exists() and _patch_qwen2_moe(qwen2_path):
        changed.append(str(qwen2_path))

    quant_path = root / "_torch" / "modules" / "fused_moe" / "quantization.py"
    if quant_path.exists() and _patch_w4a8_custom_weight_layout(quant_path):
        changed.append(str(quant_path))

    return changed


def main() -> int:
    changed = apply_trtllm_qwen_moe_patches()
    if changed:
        print("Patched TensorRT-LLM Qwen MoE W4A8_CUSTOM support:")
        for path in changed:
            print(f"  {path}")
    else:
        print("TensorRT-LLM Qwen MoE W4A8_CUSTOM patches already applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
