#!/bin/bash -l
#SBATCH -J CorrDiff-Gen
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Usage: sbatch generate.slurm <config-name> [tar-file] [extra hydra overrides...]
# Override GPU count: sbatch --gres=gpu:h100:1 generate.slurm <config>
# If tar-file is provided, the zarr is extracted from it to /scratch and bound to /data/cwa_dataset.zarr.
# Otherwise (default), the full hnvme directory is bound to /data/ (Norway setup).

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16
echo ${CUDA_VISIBLE_DEVICES}

CONFIG_NAME=${1:?"Usage: sbatch generate.slurm <config-name> [tar-file] [overrides...]"}
shift

HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
NGPUS=$(nvidia-smi -L | wc -l)

# Optional tar file as second argument
if [ -f "${1:-}" ]; then
    TAR_FILE=$1
    shift
    echo "Extracting $(basename $TAR_FILE) to /scratch..."
    tar -xf $TAR_FILE -C /scratch/
    echo "Extraction complete"
    DATA_BIND="--bind /scratch/cwa_dataset/cwa_dataset.zarr:/data/cwa_dataset.zarr"
else
    DATA_BIND="--bind ${HNVME}/:/data/"
fi

echo "Config: ${CONFIG_NAME}, GPUs: ${NGPUS}"

srun apptainer run --nv \
    ${DATA_BIND} \
    --bind ${HNVME}/checkpoints/corrdiff:/checkpoints/ \
    --bind ${HNVME}/outputs/corrdiff:/outputs/ \
    ${HNVME}/apptainer/corrdiff_ngc.sif \
    torchrun --nproc_per_node=$NGPUS \
        $HOME/bjerknes/corrdiff/generate.py --config-name ${CONFIG_NAME} "$@"
