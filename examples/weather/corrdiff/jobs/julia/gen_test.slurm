#!/bin/bash -l
#SBATCH -J PrecipitationDownscalingParallel
#SBATCH -c 16
#SBATCH -p h100
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00




export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

                        

srun apptainer run --nv --bind ~/data/downscaling:/data \
                        --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ \
                        --bind ~/data/outputs/norway:/outputs/ \
                        ~/new_corrdiff.sif torchrun --nproc_per_node=2  ~/bjerknes/corrdiff/generate_parallel_times.py --config-name gen_test_parallel_times