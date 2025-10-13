#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out








export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}





export HYDRA_FULL_ERROR=1
srun apptainer run --nv --bind ~/data/downscaling:/data \
                        --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ \
                        ~/own_corrdiff.sif python -u ~/bjerknes/corrdiff/generate.py --config-name $1 \

