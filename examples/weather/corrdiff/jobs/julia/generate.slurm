#!/bin/bash
#SBATCH -J CorrDiff-Gen
#SBATCH -c 16
#SBATCH -p h100
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out

# Usage: sbatch generate.slurm <config-name> [extra hydra overrides...]
# Override GPU count at submission: sbatch --gres=gpu:1 generate.slurm <config>

export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONNOUSERSITE=1
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

CONFIG_NAME=${1:?"Usage: sbatch generate.slurm <config-name> [overrides...]"}
shift

srun apptainer run --nv \
    --bind ~/data/downscaling:/data \
    --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ \
    --bind ~/data/outputs/norway:/outputs/ \
    ~/corrdiff_ngc.sif \
    torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE ~/bjerknes/corrdiff/generate.py \
        --config-name ${CONFIG_NAME} "$@"
