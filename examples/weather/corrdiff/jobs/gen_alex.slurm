#!/bin/bash -l
#SBATCH -J CorrDiff
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:40:1
#SBATCH --partition=a40
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# #SBATCH --constraint=a100_80


# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80



export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}





export HYDRA_FULL_ERROR=1
srun apptainer run --nv --bind /anvme/workspace/b214cb13-ecodata/downscaling/:/data/ \
                        --bind $WORK/checkpoints:/checkpoints/ \
                        $WORK/own_corrdiff.sif python -u $HOME/bjerknes/corrdiff/generate.py --config-name $1 \

