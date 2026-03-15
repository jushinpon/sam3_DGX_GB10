#!/bin/bash
#SBATCH --job-name=sam3_lean
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16 #need to check further if this is necessary for SAM3 or if we can reduce it
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sam3.log

set -euo pipefail

# === 1. Basic paths ===
WORKDIR="./"
PYTHON_ENV_BASE="/home/opt_arm/anaconda3"
PYTHON_BIN="/home/opt_arm/anaconda3/envs/sam3/bin/python"
OUTPUT_DIR="${WORKDIR}/output"

# === 2. Move to working directory ===
cd "${WORKDIR}"

# === 3. Delete old output folder first ===
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing old output folder: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

# === 4. Activate conda environment from NFS ===
source "${PYTHON_ENV_BASE}/etc/profile.d/conda.sh"
conda activate sam3

# === 5. Essential GB10 settings ===
export TORCH_CUDA_ARCH_LIST="12.1"
export CUDA_MODULE_LOADING=LAZY

# PyTorch shared library path
export TORCH_LIB_DIR=$(${PYTHON_BIN} -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# === 6. Diagnostics ===
echo "============================================================"
echo "Job started at : $(date)"
echo "Node           : $(hostname)"
echo "Workdir        : $(pwd)"
echo "Python         : $(command -v python)"
echo "Python bin     : ${PYTHON_BIN}"
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_GPUS : ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS     : ${SLURM_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "============================================================"

echo "=== scontrol show job ==="
scontrol show job "${SLURM_JOB_ID}" || true

echo "=== GPU check ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo "=== PyTorch CUDA check ==="
${PYTHON_BIN} - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.randn(8, device="cuda")
    print("tensor device:", x.device)
    print("sum:", (x * x).sum().item())
PY

# === 7. Fail early if no GPU ===
if ! ${PYTHON_BIN} - <<'PY'
import sys, torch
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
then
    echo "ERROR: This Slurm job did not receive a usable GPU."
    echo "Check the lines above for:"
    echo "  - SLURM_JOB_GPUS"
    echo "  - SLURM_GPUS"
    echo "  - CUDA_VISIBLE_DEVICES"
    echo "  - scontrol show job ${SLURM_JOB_ID}"
    exit 1
fi

echo "============================================================"
echo "Running SAM3 ..."
echo "============================================================"

# === 8. Run SAM3 example ===
${PYTHON_BIN} example.py

echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"