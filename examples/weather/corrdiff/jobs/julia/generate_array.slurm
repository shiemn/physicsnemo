#!/bin/bash
#SBATCH -J CorrDiff-Gen
#SBATCH -c 4
#SBATCH -p standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%A_%a.out

# Usage: sbatch generate_array.slurm <config-name> [year]
# Runs one job per month (array indices 0-11 = months 1-12)

export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

CONFIG_NAME=${1:?"Usage: sbatch generate_array.slurm <config-name> [year]"}
YEAR=${2:-2005}

# One task per month
MONTH=$((SLURM_ARRAY_TASK_ID + 1))
MONTH_STR=$(printf "%02d" $MONTH)

case $MONTH in
    1|3|5|7|8|10|12) LAST_DAY=31 ;;
    4|6|9|11)         LAST_DAY=30 ;;
    2)                LAST_DAY=28 ;;
esac

START_DATE="${YEAR}-${MONTH_STR}-01T00:00:00"
END_DATE="${YEAR}-${MONTH_STR}-${LAST_DAY}T23:59:59"
OUTPUT_FILE=/outputs/${CONFIG_NAME}/month_${MONTH_STR}.nc

echo "Processing ${YEAR}-${MONTH_STR}: ${START_DATE} to ${END_DATE}"
mkdir -p ~/data/outputs/norway/${CONFIG_NAME}

# Unique port per array task to avoid collisions
export MASTER_PORT=$((12345 + SLURM_ARRAY_TASK_ID))
export APPTAINERENV_MASTER_PORT=${MASTER_PORT}

srun apptainer run --nv \
    --bind ~/data/downscaling:/data \
    --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ \
    --bind ~/data/outputs/norway:/outputs/ \
    ~/corrdiff_ngc.sif \
    python -u ~/bjerknes/corrdiff/generate.py --config-name ${CONFIG_NAME} \
        ++generation.times_range="[${START_DATE}, ${END_DATE}, 3]" \
        ++generation.io.output_filename=${OUTPUT_FILE}
