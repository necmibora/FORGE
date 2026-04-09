# FORGE

**F**unction-calling **O**pen-source **R**untime for **G**rounded **E**valuation.

On-prem framework for serving and benchmarking open-source LLMs (with vLLM) inside enterprise HPC environments where data cannot leave the datacenter.

## Status — v0.1 (minimum viable backend)

- FastAPI backend with `/system`, `/models`, `/models/load|unload|loaded`, `/chat` (SSE).
- vLLM `AsyncLLMEngine` wrapper (lazy import — server starts even without vLLM installed).
- Model registry scans `MODELS_DIR` (base) and `QUANT_DIR` (pre-quantized, e.g. AWQ).
- Slurm sbatch script to launch the server on a GPU compute node.

Not yet: frontend, benchmark suites (BFCL/lm-eval/perf), in-framework quantization, custom dataset upload.

## Quick start (local, no GPU)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
FORGE_MODELS_DIR=/tmp/models FORGE_QUANT_DIR=/tmp/quant \
    uvicorn app.main:app --reload
curl http://localhost:8007/system
curl http://localhost:8007/models
```

Without `vllm` installed, `/system` and `/models` work; `/models/load` will return 503.

## On HPC

```bash
sbatch backend/slurm/start_backend.sbatch
squeue -u $USER          # find the assigned compute node
# from your laptop:
ssh -L 8007:<compute-node>:8007 <user>@<login-node>
curl http://localhost:8007/system
```

## Config (env vars, prefix `FORGE_`)

| Var | Default |
|---|---|
| `FORGE_MODELS_DIR` | `/cta/users/quantization/base_models` |
| `FORGE_QUANT_DIR`  | `/cta/users/quantization/quantized_models` |
| `FORGE_DATASETS_DIR` | `/cta/users/quantization/datasets` |
| `FORGE_PORT` (slurm script only) | `8007` |
