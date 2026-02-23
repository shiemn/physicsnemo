#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=1000G



export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}





apptainer run --nv --bind /data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/new_corrdiff.sif torchrun --nproc_per_node=1 ~/bjerknes/corrdiff/train.py --config-name $1 

