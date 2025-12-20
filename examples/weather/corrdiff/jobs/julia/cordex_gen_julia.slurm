#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=1000G


export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_HYDRA_FULL_ERROR=1
echo ${CUDA_VISIBLE_DEVICES}



echo "Copying CordexBench to local storage"
#measure runtime
start=$(date +%s.%N)
mkdir -p /tmp/CordexBench


rsync -av --progress /data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CordexBench/ /tmp/CordexBench/

wait


end=$(date +%s.%N)
echo "Copying data to local storage took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to local storage complete"

srun apptainer run --nv --bind /tmp:/data --bind ~/data/checkpoints:/checkpoints/ ~/new_corrdiff.sif python -u ~/bjerknes/corrdiff/generate.py --config-name $1 

