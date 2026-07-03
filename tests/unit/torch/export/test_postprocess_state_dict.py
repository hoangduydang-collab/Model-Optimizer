# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from modelopt.torch.export.quant_utils import (
    is_internal_quantizer_export_key,
    postprocess_state_dict,
)


def test_is_internal_quantizer_export_key_matches_fused_moe_quantizers():
    assert is_internal_quantizer_export_key(
        "model.layers.0.mlp.experts.gate_up_proj_weight_quantizers.58.1._amax"
    )
    assert is_internal_quantizer_export_key(
        "model.layers.44.mlp.experts.down_proj_weight_quantizers.92.1.scale"
    )
    assert is_internal_quantizer_export_key(
        "model.layers.0.mlp.experts.gate_up_proj_input_quantizer._amax"
    )
    assert not is_internal_quantizer_export_key(
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale"
    )


def test_postprocess_state_dict_drops_fused_moe_quantizer_keys():
    sd = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(2, 2),
        "model.layers.0.mlp.experts.gate_up_proj_weight_quantizers.58.1._amax": torch.randn(1),
        "model.layers.0.mlp.experts.down_proj_weight_quantizers.4.0._amax": torch.randn(1),
    }

    out = postprocess_state_dict(sd, 448, None)

    assert list(out.keys()) == ["model.layers.0.mlp.experts.0.gate_proj.weight"]
