#!/bin/bash -l
#SBATCH -J Train-CorrDiff
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

set -euo pipefail

echo "=================================================="
echo "Training"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
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
# Use the same job-based port as the other launchers to reduce node-local collisions.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export APPTAINERENV_MASTER_PORT=${MASTER_PORT}

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
CONFIG=${1:?"Usage: sbatch multi_node.slurm <config-name> [copy.sh|taiwan|europa] [hydra overrides...]"}
shift
DATA_MODE=${1:-copy.sh}
if [ $# -gt 0 ]; then
    shift
fi
USER_OVERRIDES=("$@")
HYDRA_OVERRIDES=()

# Copy data to all nodes (default: copy.sh; named zarr modes stage zipped zarrs).
if [ "$DATA_MODE" = "taiwan" ]; then
    echo "Data mode: taiwan zipped zarr staging"
    srun --ntasks-per-node=1 /bin/bash $HOME/bjerknes/corrdiff/jobs/helma/stage_zarr_zip.sh \
        ${HNVME}/CorrDiff/cwa_dataset.zarr.zip \
        /scratch/cwa_dataset/cwa_dataset.zarr
    HYDRA_OVERRIDES+=("dataset.data_path=/data/cwa_dataset/cwa_dataset.zarr")
elif [ "$DATA_MODE" = "europa" ]; then
    echo "Data mode: europa zipped zarr staging"
    srun --ntasks-per-node=1 /bin/bash $HOME/bjerknes/corrdiff/jobs/helma/stage_zarr_zip.sh \
        ${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip \
        /scratch/Europe/wuerzburg450_corrdiff_v2.zarr
    HYDRA_OVERRIDES+=("dataset.data_path=/data/Europe/wuerzburg450_corrdiff_v2.zarr")
else
    COPY_SCRIPT=$DATA_MODE
    echo "Data mode: copy script ${COPY_SCRIPT}"
    srun --ntasks-per-node=1 /bin/bash $HOME/bjerknes/corrdiff/jobs/helma/${COPY_SCRIPT}
fi

# Run distributed training: $SLURM_NNODES nodes x 4 GPUs each
echo "Launching torchrun: ${SLURM_NNODES} nodes x 4 GPUs = $((SLURM_NNODES * 4)) GPUs total"
set +e
srun apptainer run --nv \
                        --bind /scratch/:/data/ \
                        --bind ${HNVME}/checkpoints/corrdiff:/checkpoints/ \
                        --bind ${HNVME}/outputs/corrdiff:/outputs/ \
                        ${HNVME}/apptainer/corrdiff_ngc.sif \
                        torchrun --nnodes=${SLURM_NNODES} \
                                 --nproc_per_node=4 \
                                 --rdzv_id=${SLURM_JOB_ID} \
                                 --rdzv_backend=c10d \
                                 --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                                 $HOME/bjerknes/corrdiff/train.py --config-name $CONFIG "${HYDRA_OVERRIDES[@]}" "${USER_OVERRIDES[@]}"
status=$?
set -e

echo "=================================================="
echo "Training Complete (exit code: ${status})"
echo "Finished: $(date)"
echo "=================================================="
exit ${status}
