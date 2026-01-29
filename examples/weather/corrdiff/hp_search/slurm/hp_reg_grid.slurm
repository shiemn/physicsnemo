#!/bin/bash -l
#SBATCH -J HPRegGrid
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=hp_search/slurm/logs/hp_reg_grid-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"
#SBATCH --array=1-10%4

# Array of config names (index 1-10)
CONFIGS=(
    "hp_reg_grid/reg_01_baseline"
    "hp_reg_grid/reg_02_low_lr"
    "hp_reg_grid/reg_03_high_lr_decay"
    "hp_reg_grid/reg_04_small_batch"
    "hp_reg_grid/reg_05_large_batch"
    "hp_reg_grid/reg_06_flexi_p15"
    "hp_reg_grid/reg_07_flexi_p10"
    "hp_reg_grid/reg_08_flexi_p20"
    "hp_reg_grid/reg_09_more_attn"
    "hp_reg_grid/reg_10_combined"
)

# Get config for this array task (SLURM_ARRAY_TASK_ID is 1-indexed)
CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID-1]}

echo "=================================================="
echo "HP Regression Grid - Task ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_NAME}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Started: $(date)"
echo "=================================================="

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))  # Unique port per array job
export APPTAINERENV_MASTER_PORT=${MASTER_PORT}

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

# Copy data to all nodes
srun --ntasks-per-node=1 /bin/bash $HOME/bjerknes/corrdiff/jobs/helma/copy.sh

# Create checkpoint directory
mkdir -p /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_reg_grid

# Timeout handling - create signal file 5 min before job ends
SIGNAL_FILE_HOST="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/.timeout_signal_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SIGNAL_FILE_CONTAINER="/checkpoints/.timeout_signal_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TIMEOUT_SIGNAL_FILE="$SIGNAL_FILE_CONTAINER"
rm -f "$SIGNAL_FILE_HOST"

if [ -n "$SLURM_JOB_END_TIME" ]; then
    SIGNAL_DELAY=$(( SLURM_JOB_END_TIME - $(date +%s) - 300 ))
    if [ $SIGNAL_DELAY -gt 0 ]; then
        echo "Will create timeout signal in ${SIGNAL_DELAY} seconds"
        (sleep $SIGNAL_DELAY; touch "$SIGNAL_FILE_HOST") &
    fi
fi

# Run distributed training: 2 nodes x 4 GPUs = 8 GPUs total
srun --export=ALL,TIMEOUT_SIGNAL_FILE apptainer run --nv \
                        --env TIMEOUT_SIGNAL_FILE="$SIGNAL_FILE_CONTAINER" \
                        --bind /scratch/:/data/ \
                        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
                        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
                        /hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_infini.sif \
                        torchrun --nnodes=2 \
                                 --nproc_per_node=4 \
                                 --rdzv_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
                                 --rdzv_backend=c10d \
                                 --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                                 $HOME/bjerknes/corrdiff/train.py --config-name ${CONFIG_NAME}

EXIT_CODE=$?
rm -f "$SIGNAL_FILE_HOST"

echo "=================================================="
echo "Task ${SLURM_ARRAY_TASK_ID} Complete (exit code: ${EXIT_CODE})"
echo "Finished: $(date)"
echo "=================================================="

exit ${EXIT_CODE}
