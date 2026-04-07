#!/bin/bash
#SBATCH --job-name=forge-backend
#SBATCH --output=forge-backend-%j.out
#SBATCH --error=forge-backend-%j.err
#SBATCH --partition=cuda
#SBATCH --account=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail


module load cuda/12.8
module load python/3.11.9-5lbnnno

source ~/.venv/bin/activate

export FORGE_MODELS_DIR="${FORGE_MODELS_DIR:-/cta/users/quantization/base_models}"
export FORGE_QUANT_DIR="${FORGE_QUANT_DIR:-/cta/users/quantization/quantized_models}"
export FORGE_DATASETS_DIR="${FORGE_DATASETS_DIR:-/cta/users/quantization/datasets}"
export FORGE_PORT="${FORGE_PORT:-8007}"

cd "${FORGE_HOME:-$HOME/FORGE}/backend"

echo "Node:        $(hostname)"
echo "Slurm job:   ${SLURM_JOB_ID:-n/a}"
echo "Port:        ${FORGE_PORT}"
echo "Models dir:  ${FORGE_MODELS_DIR}"
echo "Quant dir:   ${FORGE_QUANT_DIR}"
echo
echo "From your laptop, tunnel with:"
echo "  ssh -L ${FORGE_PORT}:$(hostname):${FORGE_PORT} <user>@<login-node>"
echo

exec uvicorn app.main:app --host 0.0.0.0 --port "${FORGE_PORT}"
