#!/bin/bash -l
#SBATCH -J EDM2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=output/train_edm2-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16

echo ${CUDA_VISIBLE_DEVICES}

bash $HOME/bjerknes/corrdiff/jobs/helma/copy.sh

NGPUS=$(nvidia-smi -L | wc -l)

srun apptainer run --nv \
    --bind /scratch/:/data/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
    /hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_ngc.sif \
    torchrun --nproc_per_node=$NGPUS \
        $HOME/bjerknes/corrdiff/train.py --config-name edm2_v2_norwayW
