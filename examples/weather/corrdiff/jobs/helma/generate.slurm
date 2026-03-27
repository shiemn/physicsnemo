#!/bin/bash -l
#SBATCH -J CorrDiff-Gen
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Usage: sbatch generate.slurm <config-name> [extra hydra overrides...]
# Override GPU count: sbatch --gres=gpu:h100:1 generate.slurm <config>

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16
echo ${CUDA_VISIBLE_DEVICES}

CONFIG_NAME=${1:?"Usage: sbatch generate.slurm <config-name> [overrides...]"}
shift

HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
NGPUS=$(nvidia-smi -L | wc -l)

echo "Config: ${CONFIG_NAME}, GPUs: ${NGPUS}"

srun apptainer run --nv \
    --bind ${HNVME}/:/data/ \
    --bind ${HNVME}/checkpoints/corrdiff:/checkpoints/ \
    --bind ${HNVME}/outputs/corrdiff:/outputs/ \
    ${HNVME}/apptainer/new_corrdiff.sif \
    torchrun --nproc_per_node=$NGPUS \
        $HOME/bjerknes/corrdiff/generate.py --config-name ${CONFIG_NAME} "$@"
