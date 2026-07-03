# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from modelopt.torch.export.trtllm_w4a8_moe import postprocess_state_dict_for_trtllm_w4a8_moe


def test_postprocess_state_dict_for_trtllm_w4a8_moe_fuses_gate_up_scales():
    prefix = "model.layers.0.mlp.experts.0"
    sd = {
        f"{prefix}.gate_proj.weight_scale": torch.tensor([1.0]),
        f"{prefix}.gate_proj.weight_scale_2": torch.tensor(2.0),
        f"{prefix}.gate_proj.input_scale": torch.tensor(0.5),
        f"{prefix}.up_proj.weight_scale": torch.tensor([1.0]),
        f"{prefix}.up_proj.weight_scale_2": torch.tensor(4.0),
        f"{prefix}.up_proj.input_scale": torch.tensor(0.25),
        f"{prefix}.down_proj.weight_scale": torch.tensor([1.0]),
        f"{prefix}.down_proj.weight_scale_2": torch.tensor(3.0),
        f"{prefix}.down_proj.input_scale": torch.tensor(0.1),
    }

    out = postprocess_state_dict_for_trtllm_w4a8_moe(sd)

    expected_gate_up = torch.tensor(2.0)  # max(0.5, 0.25) * max(2, 4)
    assert torch.equal(out[f"{prefix}.gate_proj.input_scale"], expected_gate_up)
    assert torch.equal(out[f"{prefix}.up_proj.input_scale"], expected_gate_up)
    assert torch.equal(out[f"{prefix}.down_proj.input_scale"], torch.tensor(0.3))
    assert f"{prefix}.gate_proj.weight_scale_inv" in out
    assert f"{prefix}.gate_proj.weight_scale_2" not in out
    assert f"{prefix}.up_proj.weight_scale_2" not in out
    assert f"{prefix}.down_proj.weight_scale_2" not in out
