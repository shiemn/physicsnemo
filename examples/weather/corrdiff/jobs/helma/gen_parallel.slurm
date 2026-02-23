#!/bin/bash -l
#SBATCH -J GenParallel
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Usage: sbatch gen_parallel.slurm <config_name>
# Example: sbatch gen_parallel.slurm gen_heteroscedastic_parallel
#          sbatch gen_parallel.slurm gen_test_parallel_times

CONFIG_NAME=${1:-gen_test_parallel_times}
echo "Using config: ${CONFIG_NAME}"

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16
echo ${CUDA_VISIBLE_DEVICES}


srun apptainer run --nv --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/:/data/ \
                        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
                        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
                        /hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/new_corrdiff.sif \
                        torchrun --nproc_per_node=4 $HOME/bjerknes/corrdiff/generate_parallel_times.py \
                        --config-name ${CONFIG_NAME}
