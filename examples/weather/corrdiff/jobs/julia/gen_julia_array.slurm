#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 4
#SBATCH -p standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%A_%a.out

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}


echo ${CUDA_VISIBLE_DEVICES}

export HYDRA_FULL_ERROR=1

# Define month ranges (array index 0-11 corresponds to months 1-12)
MONTH=$((SLURM_ARRAY_TASK_ID + 1))
YEAR=2005

# Use ARRAY_BASE_PORT from environment if set, otherwise default
if [ -z "$ARRAY_BASE_PORT" ]; then
    ARRAY_BASE_PORT=12345
fi
export MASTER_PORT=$((ARRAY_BASE_PORT + SLURM_ARRAY_TASK_ID))
export APPTAINERENV_MASTER_PORT=$((ARRAY_BASE_PORT + SLURM_ARRAY_TASK_ID))

# Set distributed training environment variables for single GPU
export APPTAINERENV_RANK=0
export APPTAINERENV_WORLD_SIZE=1
export APPTAINERENV_LOCAL_RANK=0
export APPTAINERENV_PYTHONNOUSERSITE=1

echo "Using MASTER_PORT: ${MASTER_PORT} (array task ${SLURM_ARRAY_TASK_ID})"

# Calculate the last day of each month
case $MONTH in
    1|3|5|7|8|10|12) LAST_DAY=31 ;;
    4|6|9|11) LAST_DAY=30 ;;
    2) LAST_DAY=28 ;;  # 2005 is not a leap year
esac

# Format month with leading zero
MONTH_STR=$(printf "%02d" $MONTH)

START_DATE="${YEAR}-${MONTH_STR}-01T00:00:00"
END_DATE="${YEAR}-${MONTH_STR}-${LAST_DAY}T23:59:59"

echo "Processing month $MONTH: $START_DATE to $END_DATE"

#create an output directory on the host filesystem using the nome of the config file
CONFIG_NAME=$1
OUTPUT_DIR=~/data/outputs/norway/${CONFIG_NAME}
mkdir -p ${OUTPUT_DIR}

OUTPUT_FILE=/outputs/${CONFIG_NAME}/month_${MONTH_STR}.nc

srun apptainer run --nv --bind ~/data/downscaling:/data \
                        --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ \
                        --bind ~/data/outputs/norway:/outputs/ \
                        ~/new_corrdiff.sif python -u ~/bjerknes/corrdiff/generate.py --config-name $1 \
                        ++generation.times_range="[${START_DATE}, ${END_DATE}, 3]" \
                        ++generation.io.output_filename=${OUTPUT_FILE}

