#!/bin/bash -l
#SBATCH -J CorrDiff
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Usage: sbatch train.slurm <config-name> [--direct]
#   --direct  bind hnvme storage directly instead of copying to /scratch
# Override GPU count: sbatch --gres=gpu:h100:1 train.slurm <config>

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16
echo ${CUDA_VISIBLE_DEVICES}

CONFIG=${1:?"Usage: sbatch train.slurm <config-name> [--direct|<tar-file>]"}
DATA_MODE=${2:-scratch}

HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
NGPUS=$(nvidia-smi -L | wc -l)

if [ "$DATA_MODE" = "--direct" ]; then
    echo "Data mode: direct (binding hnvme)"
    DATA_BIND="$HNVME/"
elif [ -f "$DATA_MODE" ]; then
    echo "Data mode: tar extraction ($(basename $DATA_MODE))"
    bash $HOME/bjerknes/corrdiff/jobs/helma/copy_taiwan.sh
    DATA_BIND="/scratch/"
else
    echo "Data mode: scratch (copying to /scratch)"
    bash $HOME/bjerknes/corrdiff/jobs/helma/copy.sh
    DATA_BIND="/scratch/"
fi

srun apptainer run --nv \
    --bind ${DATA_BIND}:/data/ \
    --bind ${HNVME}/checkpoints/corrdiff:/checkpoints/ \
    --bind ${HNVME}/outputs/corrdiff:/outputs/ \
    ${HNVME}/apptainer/corrdiff_ngc.sif \
    torchrun --nproc_per_node=$NGPUS \
        $HOME/bjerknes/corrdiff/train.py --config-name $CONFIG
