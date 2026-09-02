#!/bin/bash -l
#SBATCH -J ClimateEval
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --output=output/climate-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# One climate chunk per job. The evaluation writes annual sufficient
# statistics only; it never writes predictions.nc.
#
# Usage:
#   sbatch jobs/helma/temporal/climate_eval.slurm <config_name> [run_tag] [overrides...]

set -euo pipefail

CONFIG=${1:?"Usage: sbatch jobs/helma/temporal/climate_eval.slurm <config_name> [run_tag] [overrides...]"}
RUN_TAG=${2:-${CONFIG//\//_}_${SLURM_JOB_ID}}
shift 2 2>/dev/null || shift "$#"
USER_OVERRIDES=("$@")

HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
CONTAINER=${HNVME}/apptainer/corrdiff_ngc.sif
CORRDIFF_DIR=${HOME}/bjerknes/corrdiff
OUTPUT_DIR=/outputs/climate_signal/${RUN_TAG}

export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16

NGPUS=$(nvidia-smi -L | wc -l)
if [ "${NGPUS}" -ne 1 ]; then
    echo "Climate aggregation requires exactly one GPU; allocation exposed ${NGPUS}." >&2
    exit 2
fi

echo "Config: ${CONFIG}"
echo "Run tag: ${RUN_TAG}"
echo "Output: ${OUTPUT_DIR}/climate_chunk.nc"

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
srun apptainer run --nv \
    --bind ${HNVME}:/data/ \
    --bind ${HNVME}/checkpoints/corrdiff:/checkpoints/ \
    --bind ${HNVME}/outputs/corrdiff:/outputs/ \
    ${CONTAINER} \
    torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} \
        ${CORRDIFF_DIR}/evaluate.py \
        --config-name ${CONFIG} \
        run_tag=${RUN_TAG} \
        hydra.run.dir=${OUTPUT_DIR} \
        eval.climate.output_file=${OUTPUT_DIR}/climate_chunk.nc \
        "${USER_OVERRIDES[@]}"
