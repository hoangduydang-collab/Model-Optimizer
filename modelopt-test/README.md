# ModelOpt W4A8 Qwen3 → TensorRT-LLM feasibility test

Binary pass/fail: can ModelOpt `w4a8_awq` quantize **Qwen3-30B-A3B** and can **TensorRT-LLM** (PyTorch backend) load it and return sane text?

Lives inside the **Model-Optimizer** repo (`modelopt-test/`) so the team can `git push` / `git pull` on the cluster.

## Layout

```
Model-Optimizer/modelopt-test/
  setup_env.sh           # creates both venvs
  setup_env_quant.sh     # Gate A only (.venv-quant)
  setup_env_deploy.sh    # Gate C only (.venv-deploy)
  _env_quant.sh          # activate quant venv
  _env_deploy.sh         # activate deploy venv
  run_quant.sh
  run_export_only.sh
  inspect_ckpt.py
  deploy_trtllm.py
  upgrade_deploy_env.sh  # refresh deploy venv only
  slurm/quant.sbatch
  slurm/deploy.sbatch
```

## Dual venv (recommended)

ModelOpt quant/export needs **transformers 5.x** (fused `Qwen3MoeExperts`). TRT-LLM 1.2.1 pins **transformers 4.57.3**. Use **two venvs**; the **exported checkpoint** is the contract between them.

| Venv | Path | Used for | `transformers` |
|------|------|----------|----------------|
| Quant | `.venv-quant` | Gate A: calib, export, restore | **≥ 5.0** |
| Deploy | `.venv-deploy` | Gate C: TRT-LLM load + generate | **4.57.3** |

## Prerequisites

- Repo on cluster: `/mnt/nfs/hoangduy/projects/Model-Optimizer` (`duy-branch`)
- `Qwen/Qwen3-30B-A3B` in HF cache

**Cluster note:** use `"$UV" pip` after `source /mnt/nfs/hoangduy/env.sh` — bare `pip` is not on PATH.

## Step 1 — Environment

```bash
cd /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test
bash setup_env.sh
```

Or separately:

```bash
bash setup_env_quant.sh    # Gate A
bash setup_env_deploy.sh   # Gate C
```

Recreate from scratch:

```bash
RECREATE_VENV=1 bash setup_env.sh
```

**Activate for manual work:**

```bash
source modelopt-test/_env_quant.sh   # quant / export
source modelopt-test/_env_deploy.sh  # TRT-LLM deploy
```

Legacy single `.venv` is deprecated (`MODELOPT_PROFILE=legacy source _env.sh`).

## Step 2 — Gate A: quantize (GPU, quant venv)

`run_quant.sh` sources `_env_quant.sh` automatically.

```bash
cd /mnt/nfs/hoangduy/projects/Model-Optimizer/modelopt-test
bash run_quant.sh
```

`moe_calib_experts_ratio=1.0` by default (all experts see calib tokens).

Export-only retry (uses quant venv):

```bash
bash run_export_only.sh
```

Slurm: `sbatch modelopt-test/slurm/quant.sbatch`

## Step 3 — Gate A sanity (no GPU)

```bash
source modelopt-test/_env_quant.sh
python modelopt-test/inspect_ckpt.py /mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq
```

## Step 4 — Gate C: TensorRT-LLM deploy (deploy venv, 2 GPUs)

```bash
source modelopt-test/_env_deploy.sh
python modelopt-test/deploy_trtllm.py \
  --checkpoint_dir /mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq \
  --tp 2 \
  --prompt "The capital of France is"
```

Slurm: `sbatch modelopt-test/slurm/deploy.sbatch`

W4A8_CUSTOM scale layout is applied at export; `setup_env_deploy.sh` patches TRT-LLM for Qwen MoE.

## Troubleshooting

See `BUGS_AND_FIXES.md` for known issues (MoE detection, scale fusion, restore mismatches).

- **Restore/export MoE mismatch:** use **quant venv** only; verify `Qwen3MoeExperts` + transformers 5.x.
- **TRT-LLM import errors:** use **deploy venv** only; run `bash upgrade_deploy_env.sh`.
- **Garbage generation:** re-export from calib cache (`run_export_only.sh`).

## Step 5 — Triage if Gate C fails

```bash
bash modelopt-test/run_control_int4.sh
```

## Notes

- Deploy target is **TensorRT-LLM only** (not vLLM).
- TRT-LLM matrix lists **W4A8 AWQ for Qwen-2/2.5 but not Qwen-3**; this test checks empirically.
