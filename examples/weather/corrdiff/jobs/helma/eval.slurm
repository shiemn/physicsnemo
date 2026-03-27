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
#   sbatch jobs/helma/eval.slurm <config_name> [run_tag] [extra hydra overrides...]
#
# Examples:
#   sbatch jobs/helma/eval.slurm eval_edm2_v2_norwayW
#   sbatch jobs/helma/eval.slurm eval_edm2_v2_norwayW my_run_tag
#   sbatch jobs/helma/eval.slurm eval_edm2_v2_norwayW my_run_tag generation.num_ensembles=20
#
# The config is expected to have all checkpoint paths set (no ??? fields).
# =============================================================================

set -e

CONFIG=${1:?"Usage: sbatch eval.slurm <config_name> [run_tag] [overrides...]"}
RUN_TAG=${2:-${CONFIG}_${SLURM_JOB_ID}}
shift 2 2>/dev/null || shift $#   # remaining args are extra hydra overrides

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

NGPUS=$(nvidia-smi -L | wc -l)

echo "=================================================="
echo "Config:   ${CONFIG}"
echo "Run tag:  ${RUN_TAG}"
echo "Output:   ${OUTPUT_DIR}"
echo "GPUs:     ${NGPUS}"
echo "=================================================="

# =============================================================================
# Run
# =============================================================================

MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 10000 ))

srun apptainer run --nv \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/:/data/ \
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
        "$@"

