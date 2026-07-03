# ModelOpt Qwen3-30B-A3B W4A8_AWQ — Bugs & Fixes Log

**Project:** Migrate Qwen3-30B-A3B MoE from llm-compressor → NVIDIA Model-Optimizer (`w4a8_awq`) → TensorRT-LLM (PyTorch backend, tp=2)

**Repo:** `/mnt/nfs/hoangduy/projects/Model-Optimizer` (branch `duy-branch`)

**Artifacts:** `/mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq`

**Last updated:** 2026-07-03

---

## Quick reference — environment invariants

| Stage | Env | `transformers` | Setup |
|-------|-----|----------------|-------|
| Gate A (quant/export) | `.venv-quant` | **≥ 5.0, < 5.13** | `bash setup_env_quant.sh` |
| Gate C (TRT-LLM deploy) | `.venv-deploy` | **4.57.3** | `bash setup_env_deploy.sh` |
| **Contract** | checkpoint dir on NFS | — | Artifact crosses stages |

Activate: `source modelopt-test/_env_quant.sh` or `_env_deploy.sh`.

Legacy single `.venv` + transformers 5.x patches on TRT-LLM is **deprecated** — use dual venv.

### MoE layout sanity check (quant venv only)

```bash
source modelopt-test/_env_quant.sh
cd ~/projects/Model-Optimizer

python -c "
import transformers
from transformers import AutoModelForCausalLM
from modelopt.torch.quantization.plugins.huggingface import (
    moe_block_uses_fused_experts, _is_sparse_sequaential_moe_block
)
print('transformers:', transformers.__version__)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-30B-A3B', trust_remote_code=True)
block = m.model.layers[0].mlp
print('experts type:', type(block.experts).__name__)
print('fused?', moe_block_uses_fused_experts(block))
print('sparse?', _is_sparse_sequaential_moe_block(block))
"
```

**Pass:** `transformers` ≥ 5.0, `Qwen3MoeExperts`, `fused? True`, `sparse? False`, ~13158 quantizers on restore.

**Fail:** `ModuleList`, `sparse? True`, ~56214 quantizers → transformers too old or wrong layout.

---

## Bug #1 — Restore fails: sparse path vs fused calib checkpoint (PRIMARY)

### Symptoms

```
Detected MOE module 'model.layers.0.mlp' of type Qwen3MoeSparseMoeBlock,
  registering with _QuantSparseSequentialMoe.
Inserted 56214 quantizers
...
ApplyModeError: Unmatched keys in quantizer state_dict
```

Calib on h104 had:

```
Detected fused MoE experts 'model.layers.0.mlp.experts' of type Qwen3MoeExperts
Inserted 13158 quantizers
```

### Root cause

**`transformers` version mismatch**, not primarily a detector bug.

| `transformers` | `block.experts` type | Quant path | Quantizer count |
|----------------|----------------------|------------|-----------------|
| **4.57.3** (broken) | `ModuleList` | `_QuantSparseSequentialMoe` | ~56214 |
| **≥ 5.0** (correct) | `Qwen3MoeExperts` | `_QuantFusedExperts` | ~13158 |

Calib `.pth` (58G at `.modelopt_calib_checkpoint.pth`) was built with transformers 5.x fused layout. Restore on h100 with 4.57.3 loaded a different architecture; sparse detection was **correct for ModuleList** but **wrong for the checkpoint**.

Likely trigger: shared single `.venv` on NFS with conflicting `transformers` pins between quant and deploy.

### Fix

```bash
source modelopt-test/_env_quant.sh
cd ~/projects/Model-Optimizer

"$UV" pip install 'transformers>=5.0,<5.13'
"$UV" pip uninstall -y nvidia-modelopt 2>/dev/null || true
"$UV" pip install -e '.[hf]'
```

Re-run MoE sanity check, then `bash modelopt-test/run_export_only.sh`.

### Would the original detector have worked on transformers 5.x?

**Yes, likely.** Upstream already had `if _is_fused_experts_module(module.experts): return False` before sparse registration, and `register_fused_experts_on_the_fly` runs before sparse in plugin order. The incident was loading **ModuleList** (v4 layout), not misclassifying `Qwen3MoeExperts`.

---

## Bug #2 — MoE detector hardening (SECONDARY / defense in depth)

### Symptoms (theoretical / edge cases)

Fused `Qwen3MoeExperts` misclassified as sparse when heuristics rely on `hasattr(experts, "__iter__")` — always true for `nn.Module`.

### Root cause

`nn.Module` is iterable over child submodules. Comment in old code assumed fused experts would not be iterable; transformers 5.x can make them iterable for indexing.

### Fix (branch `duy-branch`, commits `aaf3f2d6`, `59936ce5`)

**File:** `modelopt/torch/quantization/plugins/huggingface.py`

1. `_looks_like_fused_moe_experts()` — structural checks, not `__iter__`.
2. `isinstance(experts, nn.ModuleList)` required for sparse path.
3. `moe_block_uses_fused_experts()` — early exit in `register_sparse_moe_on_the_fly`.
4. `_FUSED_MOE_EXPERTS_TYPE_NAMES` — explicit registry (`Qwen3MoeExperts`, `DeepseekV3NaiveMoe`, `MixtralExperts`, …), mirroring export `layer_utils.get_expert_linear_names()`.
5. `_wrapper_class_for_known_fused_moe_experts()` — fallback when structural `_fused_experts_wrapper_class()` returns None.

**Tests:** `tests/unit/torch/quantization/plugins/test_fused_experts.py`

---

## Bug #3 — TRT-LLM garbage generation (repetitive tokens)

### Symptoms

Gate C loads OK (~24s, 873 modules) but generation is garbage: `"organisers organisers organisers …"`

### Root cause

TRT-LLM `W4A8_CUSTOM` scale math:

- **gate/up (fc31):** `fc31_alpha = input_scale` only (activation scale)
- **down (fc2):** `fc2_alpha = input_scale` with `weight_scale_2` folded in

Our rewriter incorrectly fused `input_scale * weight_scale_2` into **gate/up** `input_scale` (VANILLA path behavior).

### Fix

**File:** `modelopt/torch/export/trtllm_w4a8_moe.py`

- Gate/up: `max(gate.input_scale, up.input_scale)` only
- Down: fuse `input_scale * weight_scale_2`
- Marker bumped to `trtllm_w4a8_custom_v2`

**Tests:** `tests/unit/torch/export/test_trtllm_w4a8_moe.py`

**Important:** Existing on-disk checkpoint may have wrong v1 gate/up scales. `--force` rewrite alone cannot recover; need **re-export** from calib checkpoint after fix.

---

## Bug #4 — HF export fails: quantizer keys in state_dict

### Symptoms

Export terminal flooded with `gate_up_proj_weight_quantizers.*` keys; `save_pretrained` fails.

### Fix

**Files:** `modelopt/torch/export/quant_utils.py`, `unified_export_hf.py`

- `is_internal_quantizer_export_key()` / `filter_internal_quantizer_keys_from_export_state_dict()`
- Applied before `save_pretrained`

---

## Bug #5 — HF export fails: AWQ resmooth on fused experts

### Symptoms

```
len(module.experts)  # QuantQwen3MoeExperts has no __len__
```

During `requantize_resmooth_fused_llm_layers` → `get_experts_list()`.

### Root cause

AWQ resmooth assumed ModuleList experts; fused `_QuantFusedExperts` uses shared input quantizers per projection.

### Fix

**File:** `modelopt/torch/export/unified_export_hf.py`

Skip per-expert ModuleList resmooth when fused experts have `{first_proj_attr}_weight_quantizers` (structural check). Fused path uses `_export_fused_experts`.

---

## Bug #6 — Dual modelopt distribution warning

### Symptoms

```
Multiple distributions found for package modelopt. Picked distribution: nvidia-modelopt
```

Editable checkout exists at repo `modelopt/__init__.py` but imports may hit PyPI.

### Fix

```bash
"$UV" pip uninstall -y nvidia-modelopt
"$UV" pip install -e '.[hf]'
python -c "import modelopt; print(modelopt.__file__)"
# Expect: /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt/__init__.py
```

`setup_env_deploy.sh` re-pins editable install after `tensorrt-llm` install for this reason.

---

## TRT-LLM patches (working)

**File:** `modelopt/deploy/trtllm_qwen_moe_patch.py`

- Qwen MoE selects `W4A8_CUSTOM` when `is_int4_weight_only_per_group()`
- SM90 CUSTOM uses VANILLA weight repack (ModelOpt HF weights, not DeepSeek pre-packed)

```bash
python -c "from modelopt.deploy.trtllm_qwen_moe_patch import apply_trtllm_qwen_moe_patches; print(apply_trtllm_qwen_moe_patches())"
```

Applied automatically by `modelopt-test/setup_env_deploy.sh`.

---

## Key files

| File | Role |
|------|------|
| `modelopt/torch/quantization/plugins/huggingface.py` | Fused vs sparse MoE registration |
| `modelopt/torch/quantization/plugins/custom.py` | Plugin order (fused before sparse) |
| `modelopt/torch/export/trtllm_w4a8_moe.py` | W4A8_CUSTOM scale rewrite (v2) |
| `modelopt/torch/export/quant_utils.py` | Export quantizer key filter |
| `modelopt/torch/export/unified_export_hf.py` | Export pipeline + fused MoE resmooth skip |
| `modelopt/torch/export/layer_utils.py` | Export-side MoE class names (reference) |
| `modelopt/deploy/trtllm_qwen_moe_patch.py` | TRT-LLM install patches |
| `examples/hf_ptq/hf_ptq.py` | Calib save/restore, `--export_only` |
| `modelopt-test/run_export_only.sh` | Re-export without re-calib |
| `modelopt-test/deploy_trtllm.py` | Gate C deploy |

---

## Gate status (as of 2026-07-03)

| Gate | Status | Notes |
|------|--------|-------|
| A: AWQ calib + save `.pth` | **Done** | h104, ~1.5h, 13158 quantizers, `Qwen3MoeExperts` |
| A: HF export | **Blocked → unblocked** | Export bugs fixed; blocked on transformers align + re-export |
| C: TRT-LLM load | **Pass** | 873 modules, ~24s |
| C: Generation quality | **Fail → pending** | Scale fusion fix needs re-export + redeploy |

---

## Cluster commands

```bash
cd ~/projects/Model-Optimizer

# Gate A (quant venv)
source modelopt-test/_env_quant.sh
bash modelopt-test/run_export_only.sh

# Gate C (deploy venv)
source modelopt-test/_env_deploy.sh
python modelopt-test/deploy_trtllm.py \
  --checkpoint_dir /mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq \
  --tp 2 --prompt "The capital of France is"
```

---

## Bug #7 — TRT-LLM import fails on transformers 5.x

### Symptoms

```
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

When running `apply_trtllm_qwen_moe_patches()` or `deploy_trtllm.py` after upgrading to `transformers>=5.0` for fused MoE export.

### Root cause

**Conflicting requirements:**

- ModelOpt Qwen3 MoE calib/export/restore needs **transformers ≥ 5.0** (`Qwen3MoeExperts`)
- TensorRT-LLM **1.2.x** still imports `AutoModelForVision2Seq`, removed in transformers 5.0 (renamed to `AutoModelForImageTextToText`)

### Fix

**Long-term (team):** **Dual venv** — `.venv-quant` (transformers 5.x) + `.venv-deploy` (TRT-LLM + transformers 4.57.3). See `setup_env_quant.sh`, `setup_env_deploy.sh`, `_env_quant.sh`, `_env_deploy.sh`.

**Deprecated:** single `.venv` with install-time TRT-LLM transformers-5 patches (`trtllm_transformers5_patch.py`) — only for legacy; do not extend.

**Immediate workaround** (before pulling fix):

```bash
python -c "
from modelopt.deploy.transformers_compat import apply_transformers_compat
print(apply_transformers_compat())
from modelopt.deploy.trtllm_qwen_moe_patch import apply_trtllm_qwen_moe_patches
print(apply_trtllm_qwen_moe_patches())
"
```

Or manual shim:

```bash
python -c "
import transformers
from transformers import AutoModelForImageTextToText
transformers.AutoModelForVision2Seq = AutoModelForImageTextToText
"
```

---

## Related docs

- `modelopt_migration_assessment.md` — strategic migration assessment (llm-compressor → ModelOpt)
- `.cursor/rules/team-and-env-strategy.mdc` — team context + dual-venv / container strategy
- `.cursor/rules/modelopt-qwen3-moe-migration.mdc` — agent memory rule for this workstream
- `.cursor/rules/cluster-python-uv.mdc` — `$UV` / venv conventions on cluster

## Environment strategy (team)

See `.cursor/rules/team-and-env-strategy.mdc`. **Dual venv** (quant vs deploy) is the recommended local/cluster pattern when TRT-LLM and ModelOpt require incompatible `transformers` pins. **NGC TRT-LLM containers** are NVIDIA’s recommended production deploy path ([installation guide](https://nvidia.github.io/TensorRT-LLM/installation/installation-guide.html)).
