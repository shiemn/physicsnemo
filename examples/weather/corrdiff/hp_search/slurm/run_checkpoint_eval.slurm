#!/bin/bash -l
#SBATCH -J HP_CheckpointEval
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --ntasks-per-node=1
#SBATCH --output=hp_search/slurm/logs/checkpoint_eval-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"
# NOTE: --nodes and --gres must be passed via sbatch command line:
#   sbatch --nodes=1 --gres=gpu:h100:4 ...

# =============================================================================
# Checkpoint Evaluation Script
# Evaluates all unfinished diffusion/regression checkpoint combinations
# =============================================================================

echo "=================================================="
echo "Checkpoint Evaluation Job"
echo "=================================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "Started:    $(date)"
echo "=================================================="

# Configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoints}"
CONFIG_DIR="${CONFIG_DIR:-hp_search_configs}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/hp_search/checkpoint_eval}"
BASE_GEN_CONFIG="${BASE_GEN_CONFIG:-conf/hp_base_gen.yaml}"
GEN_CONFIGS="${GEN_CONFIGS:-}"  # Uses defaults in evaluate_checkpoints.py if empty
MIN_DIFF_STEPS="${MIN_DIFF_STEPS:-50000}"
MIN_REG_STEPS="${MIN_REG_STEPS:-50000}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
GRES="${GRES:-gpu:h100:4}"

# Host paths (for directories that need to exist before container starts)
HOST_BASE="/hnvme/workspace/b214cb11-helma-ecodata/downscaling"
HOST_OUTPUT_DIR="${HOST_BASE}/outputs/corrdiff/hp_search/checkpoint_eval"

# Create output directories on the host filesystem
mkdir -p "$HOST_OUTPUT_DIR"
mkdir -p hp_search/slurm/logs

# Setup environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16

# Multi-node setup: set MASTER_PORT to avoid conflicts between jobs
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 1000))
export APPTAINERENV_MASTER_PORT=${MASTER_PORT}
echo "Master port: ${MASTER_PORT}"
echo "Node list: ${SLURM_NODELIST}"
echo "Node ID: ${SLURM_NODEID:-0}"

# Copy data to each node's local scratch (srun ensures all nodes complete before continuing)
srun --ntasks-per-node=1 /bin/bash $HOME/bjerknes/corrdiff/jobs/helma/copy.sh

# Container setup
CONTAINER="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_infini.sif"

# Generate the evaluation script inside the container
echo ""
echo "Generating evaluation script..."
echo ""

apptainer exec --nv \
    --bind /scratch/:/data/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
    ${CONTAINER} \
    python hp_search/evaluate_checkpoints.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --config-dir "${CONFIG_DIR}" \
        --base-gen-config "${BASE_GEN_CONFIG}" \
        --gen-configs "${GEN_CONFIGS}" \
        --output-dir "${OUTPUT_DIR}" \
        --min-diff-steps ${MIN_DIFF_STEPS} \
        --min-reg-steps ${MIN_REG_STEPS} \
        --num-gpus ${NUM_GPUS} \
        --output-script hp_search/slurm/eval_checkpoints_generated.sh

echo ""
echo "Running evaluations..."
echo ""

# Run the generated script inside the container
srun apptainer exec --nv \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling:/data/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
    ${CONTAINER} \
    bash hp_search/slurm/eval_checkpoints_generated.sh

echo ""
echo "=================================================="
echo "Checkpoint Evaluation Complete"
echo "Finished: $(date)"
echo "=================================================="

# Summarize results
echo ""
echo "Results saved to: ${OUTPUT_DIR} (container path)"
echo "Host path: ${HOST_OUTPUT_DIR}"
echo ""
ls -la "${HOST_OUTPUT_DIR}"/ 2>/dev/null | head -20
