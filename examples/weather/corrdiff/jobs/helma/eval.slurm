#!/bin/bash -l
#SBATCH -J Eval
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --output=output/eval-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# =============================================================================
# Evaluate a CorrDiff model
#
# Usage:
#   sbatch jobs/helma/eval.slurm <config_name> [run_tag] [--direct|taiwan|europa] [extra hydra overrides...]
#
# Examples:
#   sbatch jobs/helma/eval.slurm eval_edm2_v2_norwayW
#   sbatch jobs/helma/eval.slurm eval_edm2_v2_norwayW my_run_tag
#   sbatch jobs/helma/eval.slurm eval_edm2_v2_norwayW my_run_tag generation.num_ensembles=20
#   sbatch jobs/helma/eval.slurm eval/europa/diffusion europe_2021 europa
#
# The config is expected to have all checkpoint paths set (no ??? fields).
# =============================================================================

set -e

CONFIG=${1:?"Usage: sbatch eval.slurm <config_name> [run_tag] [overrides...]"}
RUN_TAG=${2:-${CONFIG}_${SLURM_JOB_ID}}
shift 2 2>/dev/null || shift $#   # remaining args are extra hydra overrides

DATA_MODE=--direct
if [ "${1:-}" = "--direct" ] || [ "${1:-}" = "taiwan" ] || [ "${1:-}" = "europa" ]; then
    DATA_MODE=$1
    shift
fi
USER_OVERRIDES=("$@")

OUTPUT_DIR="/outputs/eval/${RUN_TAG}"

# =============================================================================
# Environment
# =============================================================================

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16

CONTAINER="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_ngc.sif"
CORRDIFF_DIR="$HOME/bjerknes/corrdiff"
HNVME="/hnvme/workspace/b214cb11-helma-ecodata/downscaling"

HYDRA_OVERRIDES=()
if [ "$DATA_MODE" = "taiwan" ]; then
    echo "Data mode: taiwan zipped zarr staging"
    bash "$CORRDIFF_DIR/jobs/helma/stage_zarr_zip.sh" \
        "$HNVME/CorrDiff/cwa_dataset.zarr.zip" \
        /scratch/cwa_dataset/cwa_dataset.zarr
    DATA_BIND=/scratch/
    HYDRA_OVERRIDES+=("dataset.data_path=/data/cwa_dataset/cwa_dataset.zarr")
elif [ "$DATA_MODE" = "europa" ]; then
    echo "Data mode: europa zipped zarr staging"
    bash "$CORRDIFF_DIR/jobs/helma/stage_zarr_zip.sh" \
        "$HNVME/Europe/wuerzburg450_corrdiff_v2.zarr.zip" \
        /scratch/Europe/wuerzburg450_corrdiff_v2.zarr
    DATA_BIND=/scratch/
    HYDRA_OVERRIDES+=("dataset.data_path=/data/Europe/wuerzburg450_corrdiff_v2.zarr")
else
    echo "Data mode: direct (binding hnvme)"
    DATA_BIND="$HNVME/"
fi

NGPUS=$(nvidia-smi -L | wc -l)

echo "=================================================="
echo "Config:   ${CONFIG}"
echo "Run tag:  ${RUN_TAG}"
echo "Output:   ${OUTPUT_DIR}"
echo "GPUs:     ${NGPUS}"
echo "Data mode:${DATA_MODE}"
echo "=================================================="

# =============================================================================
# Run
# =============================================================================

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

srun apptainer run --nv \
    --bind ${DATA_BIND}:/data/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
    ${CONTAINER} \
    torchrun --nproc_per_node=$NGPUS --master_port=$MASTER_PORT \
        ${CORRDIFF_DIR}/evaluate.py \
        --config-name ${CONFIG} \
        run_tag=${RUN_TAG} \
        hydra.run.dir=${OUTPUT_DIR} \
        wandb.results_dir=${OUTPUT_DIR}/wandb \
        eval.output_json=${OUTPUT_DIR}/eval_results.json \
        eval.predictions_file=${OUTPUT_DIR}/predictions.nc \
        "${HYDRA_OVERRIDES[@]}" \
        "${USER_OVERRIDES[@]}"
