# ModelOpt W4A8 Qwen3 → TensorRT-LLM feasibility test

Binary pass/fail: can ModelOpt `w4a8_awq` quantize **Qwen3-30B-A3B** and can **TensorRT-LLM** (PyTorch backend) load it and return sane text?

Lives inside the **Model-Optimizer** repo (`modelopt-test/`) so you can `git push` / `git pull` on the cluster.

## Layout

```
Model-Optimizer/modelopt-test/
  setup_env.sh
  run_quant.sh
  run_export_only.sh   # retry export without re-calib (needs .modelopt_calib_checkpoint.pth)
  inspect_ckpt.py
  deploy_trtllm.py
  run_control_int4.sh
  slurm/quant.sbatch
  slurm/deploy.sbatch
```

## Prerequisites

- This repo on the cluster, e.g. `/mnt/nfs/hoangduy/projects/Model-Optimizer` (duy-branch)
- `Qwen/Qwen3-30B-A3B` in HF cache (from the llm-compressor run)

**Cluster note:** bare `pip` is not on PATH. Use `bash setup_env.sh` (runs `uv pip`) or after `source modelopt-test/_env.sh` use `uv pip` / `python -m pip` — never `pip install` alone.

Scripts resolve the repo root automatically (`modelopt-test/..`). The venv is **project-local** at `Model-Optimizer/.venv` (gitignored) so it does not collide with `venvs/main` from `env.sh`.

## Step 1 — Environment (you)

```bash
cd /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test
bash setup_env.sh
```

Creates or reuses `<repo>/.venv`. **Default setup is deploy-only** (TensorRT-LLM + patches; skips `hf_ptq` / `flash-attn` so torch is not upgraded to an incompatible CUDA 13 stack).

```bash
bash setup_env.sh
```

For Gate A quantization (adds `hf_ptq` deps without upgrading torch):

```bash
INSTALL_HF_PTQ=1 bash setup_env.sh
```

If your venv is broken (wrong torch/CUDA), recreate:

```bash
RECREATE_VENV=1 bash setup_env.sh
```

For later sessions:

```bash
source /mnt/nfs/hoangduy/projects/Model-Optimizer/.venv/bin/activate
```

Paste back the printed `modelopt` / `tensorrt_llm` versions or any install error.

## Step 2 — Gate A: quantize (GPU)

Uses public `cnn_dailymail` calibration data (ModelOpt's default `cnn_nemotron_v2_mix` requires gated Nemotron v2 access on HuggingFace).

`run_quant.sh` passes `--moe_calib_experts_ratio 1.0` by default so **all MoE experts** see calibration tokens (same intent as llm-compressor `moe_calibrate_all_experts: true`). Without this, Qwen3's sparse top-k routing leaves many experts with zero `amax`; export may still succeed via weight fallbacks, but quantization quality on cold experts is poor.

```bash
cd /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test
bash run_quant.sh
```

Override dataset if needed: `CALIB_DATASET=wikitext bash run_quant.sh`

**Current run (sparse routing):** If you already finished calibration without `moe_calib_experts_ratio`, pass Gate A export with the existing calib cache — accuracy is not trusted, but the binary checkpoint test is still valid:

```bash
# Sync repo (export patches in modelopt/torch/export/*.py), then:
bash run_export_only.sh
```

After calibration (before export), `hf_ptq.py` saves
`<export_path>/.modelopt_calib_checkpoint.pth` (~full model state). If export fails,
retry export only (minutes, not hours):

```bash
bash run_export_only.sh
```

Re-quant with full expert coverage (for a fair accuracy comparison vs llm-compressor):

```bash
bash run_quant.sh   # now includes moe_calib_experts_ratio=1.0 by default
```

Or Slurm:

```bash
sbatch /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test/slurm/quant.sbatch
```

## Step 3 — Gate A sanity (no GPU)

```bash
python /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test/inspect_ckpt.py \
  /mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq
```

## Step 4 — Gate C: TensorRT-LLM deploy (2 GPUs)

W4A8_AWQ MoE checkpoints are exported with TRT-LLM ``W4A8_CUSTOM`` scale layout
(``weight_scale_inv``, fused ``input_scale``). ``setup_env.sh`` patches installed
TensorRT-LLM so Qwen MoE selects ``W4A8_CUSTOM`` (same as DeepSeek V3).

```bash
python /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test/deploy_trtllm.py \
  --checkpoint_dir /mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq \
  --tp 2 \
  --prompt "The capital of France is"
```

For checkpoints exported **before** this integration, rewrite scales once:

```bash
python modelopt-test/trtllm_w4a8_moe_custom.py /mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq --force
```

Or: `sbatch modelopt-test/slurm/deploy.sbatch` (from repo root).

## Step 5 — Triage if Gate C fails

```bash
bash /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test/run_control_int4.sh
```

## Notes

- Deploy target is **TensorRT-LLM only** (not vLLM).
- TRT-LLM matrix lists **W4A8 AWQ for Qwen-2/2.5 but not Qwen-3**; this test checks empirically.
- **MoE calibration:** `MOE_CALIB_EXPERTS_RATIO=1.0` (default) matches llm-compressor's calibrate-all-experts behavior. Lower values mirror inference top-k and can leave expert quantizers uncalibrated.
