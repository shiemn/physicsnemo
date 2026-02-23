#!/bin/bash -l
#SBATCH -J twCRPS
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=output/train_twcrps-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

echo "=================================================="
echo "Heteroscedastic twCRPS Training"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Started: $(date)"
echo "=================================================="

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export APPTAINERENV_MASTER_PORT=${MASTER_PORT}

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

# Copy data to all nodes
srun --ntasks-per-node=1 /bin/bash $HOME/bjerknes/corrdiff/jobs/helma/copy.sh

# Run distributed training: 2 nodes x 4 GPUs = 8 GPUs total
srun apptainer run --nv \
                        --bind /scratch/:/data/ \
                        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
                        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
                        /hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/new_corrdiff.sif \
                        torchrun --nnodes=2 \
                                 --nproc_per_node=4 \
                                 --rdzv_id=${SLURM_JOB_ID} \
                                 --rdzv_backend=c10d \
                                 --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                                 $HOME/bjerknes/corrdiff/train.py --config-name $1

echo "=================================================="
echo "Training Complete (exit code: $?)"
echo "Finished: $(date)"
echo "=================================================="
