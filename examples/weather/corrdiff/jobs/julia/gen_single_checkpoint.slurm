#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1


export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export PYTHONNOUSERSITE=1

echo ${CUDA_VISIBLE_DEVICES}

export HYDRA_FULL_ERROR=1

# Use MASTER_PORT from environment if set, otherwise default
if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=12345
fi
export MASTER_PORT=${MASTER_PORT}
export APPTAINERENV_MASTER_PORT=${MASTER_PORT}

# Set distributed training environment variables for single GPU
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export APPTAINERENV_RANK=0
export APPTAINERENV_WORLD_SIZE=1
export APPTAINERENV_LOCAL_RANK=0

echo "Using MASTER_PORT: ${MASTER_PORT}"

# Get parameters
CONFIG_NAME=$1
CHECKPOINT_PATH=${2:-""}
OUTPUT_FILE=${3:-"/outputs/output.nc"}
YEAR=${4:-2005}

# Build checkpoint override if provided
if [ -n "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_OVERRIDE="++generation.io.reg_ckpt_filename=${CHECKPOINT_PATH}"
else
    CHECKPOINT_OVERRIDE=""
fi

START_DATE="${YEAR}-01-01T00:00:00"
END_DATE="${YEAR}-12-31T23:59:59"

echo "Processing full year: $START_DATE to $END_DATE"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_FILE}"

# Create output directory on the host filesystem (before container runs)
# OUTPUT_FILE is the path inside container (e.g., /outputs/multi_checkpoint_results/...)
# We need to create it on the host at ~/data/outputs/norway/...
HOST_OUTPUT_FILE="${OUTPUT_FILE#/outputs/}"
HOST_OUTPUT_DIR=$(dirname "${HOST_OUTPUT_FILE}")
mkdir -p ~/data/outputs/norway/"${HOST_OUTPUT_DIR}"
echo "Created host output directory: ~/data/outputs/norway/${HOST_OUTPUT_DIR}"

srun apptainer run --nv --bind ~/data/downscaling:/data \
                        --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ \
                        --bind ~/data/outputs/norway:/outputs/ \
                        ~/new_corrdiff.sif python -u ~/bjerknes/corrdiff/generate.py --config-name $CONFIG_NAME \
                        ++generation.times_range="[${START_DATE}, ${END_DATE}, 3]" \
                        ++generation.io.output_filename=${OUTPUT_FILE} \
                        ${CHECKPOINT_OVERRIDE}


