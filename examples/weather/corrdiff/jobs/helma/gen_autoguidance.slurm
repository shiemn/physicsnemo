#!/bin/bash -l
#SBATCH -J AutoGuidance
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --output=output/gen_autoguidance-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Usage:
#   sbatch gen_autoguidance.slurm \
#       /checkpoints/.../strong.mdlus \
#       /checkpoints/.../weak.mdlus \
#       1.5
#
# $1 = strong (main) checkpoint
# $2 = weak (guidance) checkpoint
# $3 = guidance scale (default 1.5)

STRONG_CKPT=${1:?"Pass strong checkpoint as \$1"}
WEAK_CKPT=${2:?"Pass weak (guidance) checkpoint as \$2"}
GUIDANCE_SCALE=${3:-1.5}

echo "Strong checkpoint : ${STRONG_CKPT}"
echo "Guidance checkpoint: ${WEAK_CKPT}"
echo "Guidance scale    : ${GUIDANCE_SCALE}"

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16
echo ${CUDA_VISIBLE_DEVICES}

srun apptainer run --nv \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/:/data/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
    /hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/new_corrdiff.sif \
    torchrun --nproc_per_node=1 \
        $HOME/bjerknes/corrdiff/generate_autoguidance.py \
        --config-name generate_autoguidance_norwayW \
        generation.io.res_ckpt_filename="${STRONG_CKPT}" \
        generation.guidance.guide_ckpt_filename="${WEAK_CKPT}" \
        generation.guidance.scale=${GUIDANCE_SCALE} \
        generation.io.output_filename="/outputs/autoguidance_w${GUIDANCE_SCALE}_${SLURM_JOB_ID}.nc"
