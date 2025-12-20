#!/bin/bash -l
#SBATCH -J CorrDiff
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"
#SBATCH --constraint=a100_80


# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
#export APPTAINERENV_NCCL_DEBUG=INFO
#export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_OMP_NUM_THREADS=16
echo ${CUDA_VISIBLE_DEVICES}


bash $HOME/bjerknes/corrdiff/jobs/alex/copy_norway_data.sh



srun apptainer run --nv --bind /scratch/:/data/ \
                        --bind $WORK/checkpoints:/checkpoints/ \
                        $WORK/new_corrdiff.sif torchrun --nproc_per_node=8  $HOME/bjerknes/corrdiff/train.py --config-name $1 \
                        

