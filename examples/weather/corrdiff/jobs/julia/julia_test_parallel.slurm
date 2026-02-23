#!/bin/bash
#SBATCH -J PrecipitationDownscalingParallel
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=50G
#SBATCH --time=1:00:00



export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}




end=$(date +%s.%N)
echo "Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to scratch complete"


srun apptainer run --nv --bind /tmp/:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/new_corrdiff.sif torchrun --nproc_per_node=2  ~/bjerknes/corrdiff/train.py --config-name $1 



