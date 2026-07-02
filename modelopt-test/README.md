# ModelOpt W4A8 Qwen3 → TensorRT-LLM feasibility test

Binary pass/fail: can ModelOpt `w4a8_awq` quantize **Qwen3-30B-A3B** and can **TensorRT-LLM** (PyTorch backend) load it and return sane text?

Lives inside the **Model-Optimizer** repo (`modelopt-test/`) so you can `git push` / `git pull` on the cluster.

## Layout

```
Model-Optimizer/modelopt-test/
  setup_env.sh
  run_quant.sh
  inspect_ckpt.py
  deploy_trtllm.py
  run_control_int4.sh
  slurm/quant.sbatch
  slurm/deploy.sbatch
```

## Prerequisites

- This repo on the cluster, e.g. `/mnt/nfs/hoangduy/projects/Model-Optimizer` (duy-branch)
- `Qwen/Qwen3-30B-A3B` in HF cache (from the llm-compressor run)

Scripts resolve the repo root automatically (`modelopt-test/..`). The venv is **project-local** at `Model-Optimizer/.venv` (gitignored) so it does not collide with `venvs/main` from `env.sh`.

## Step 1 — Environment (you)

```bash
cd /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test
bash setup_env.sh
```

Creates or reuses `<repo>/.venv`. To force a clean rebuild:

```bash
RECREATE_VENV=1 bash setup_env.sh
```

For later sessions:

```bash
source /mnt/nfs/hoangduy/projects/Model-Optimizer/.venv/bin/activate
```

Paste back the printed `modelopt` / `tensorrt_llm` versions or any install error.

## Step 2 — Gate A: quantize (GPU)

```bash
cd /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test
bash run_quant.sh
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

```bash
python /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test/deploy_trtllm.py \
  --checkpoint_dir /mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq \
  --tp 2 \
  --prompt "The capital of France is"
```

Or: `sbatch modelopt-test/slurm/deploy.sbatch` (from repo root).

## Step 5 — Triage if Gate C fails

```bash
bash /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test/run_control_int4.sh
```

## Notes

- Deploy target is **TensorRT-LLM only** (not vLLM).
- TRT-LLM matrix lists **W4A8 AWQ for Qwen-2/2.5 but not Qwen-3**; this test checks empirically.
