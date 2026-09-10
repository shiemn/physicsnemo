#!/bin/bash -l
#SBATCH -J CorrDiff
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/daniel/outputs/ablation_1_and_2_models/slurm/job-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user=daniel.weggenmann@uni-wuerzburg.de

# Usage: sbatch train.slurm <config-name> [--direct|scratch|taiwan|europa|<tar-file>] [hydra overrides...]
#   --direct  bind hnvme storage directly instead of copying to /scratch
#   taiwan    extract cwa_dataset.zarr.zip to /scratch and override dataset.data_path
#   europa    extract wuerzburg450_corrdiff_v2.zarr.zip to /scratch and override dataset.data_path
# Override GPU count: sbatch --gres=gpu:h100:1 train.slurm <config>

set -eo pipefail

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16
echo ${CUDA_VISIBLE_DEVICES}

CONFIG=${1:?"Usage: sbatch train.slurm <config-name> [--direct|scratch|taiwan|europa|<tar-file>] [hydra overrides...]"}
shift
DATA_MODE=${1:-scratch}
if [ $# -gt 0 ]; then
    shift
fi
USER_OVERRIDES=("$@")

HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
CORRDIFF_DIR="${HOME}/corrdiffProjektSimon/code/examples/weather/corrdiff"
RUN_ROOT="${HNVME}/daniel"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints/ablation_1_and_2_models"
OUTPUT_ROOT="${RUN_ROOT}/outputs/ablation_1_and_2_models"
NGPUS=$(nvidia-smi -L | wc -l)
# Use the same job-based port as the other launchers to reduce node-local collisions.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
HYDRA_OVERRIDES=()

echo "MASTER_PORT: ${MASTER_PORT}"

if [ "$DATA_MODE" = "--direct" ]; then
    echo "Data mode: direct (binding hnvme)"
    DATA_BIND="$HNVME/"
elif [ "$DATA_MODE" = "taiwan" ]; then
    echo "Data mode: taiwan zipped zarr staging"
    bash "${CORRDIFF_DIR}/jobs/helma/stage_zarr_zip.sh" \
        ${HNVME}/CorrDiff/cwa_dataset.zarr.zip \
        /scratch/cwa_dataset/cwa_dataset.zarr
    DATA_BIND="/scratch/"
    HYDRA_OVERRIDES+=("dataset.data_path=/data/cwa_dataset/cwa_dataset.zarr")
elif [ "$DATA_MODE" = "europa" ]; then
    echo "Data mode: europa zipped zarr staging"
    bash "${CORRDIFF_DIR}/jobs/helma/stage_zarr_zip.sh" \
        ${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip \
        /scratch/Europe/wuerzburg450_corrdiff_v2.zarr
    DATA_BIND="/scratch/"
    HYDRA_OVERRIDES+=("dataset.data_path=/data/Europe/wuerzburg450_corrdiff_v2.zarr")
elif [ -f "$DATA_MODE" ]; then
    echo "Data mode: tar extraction ($(basename $DATA_MODE))"
    bash "${CORRDIFF_DIR}/jobs/helma/copy_taiwan.sh"
    DATA_BIND="/scratch/"
else
    echo "Data mode: scratch (copying to /scratch)"
    bash "${CORRDIFF_DIR}/jobs/helma/copy.sh"
    DATA_BIND="/scratch/"
fi

srun --ntasks=1 apptainer run --nv \
    --bind ${DATA_BIND}:/data/ \
    --bind "${CHECKPOINT_ROOT}:/checkpoints/" \
    --bind "${OUTPUT_ROOT}:/outputs/" \
    ${HNVME}/apptainer/corrdiff_ngc.sif \
    torchrun --nproc_per_node=$NGPUS --master_port=${MASTER_PORT} \
        "${CORRDIFF_DIR}/train.py" --config-name "$CONFIG" "${HYDRA_OVERRIDES[@]}" "${USER_OVERRIDES[@]}"
