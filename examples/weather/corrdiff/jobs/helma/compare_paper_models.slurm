#!/bin/bash -l
#SBATCH -J PaperCompare
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=output/paper-compare-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Compare completed paper-protocol outputs across models.
#
# Usage:
#   sbatch jobs/helma/compare_paper_models.slurm [config_name]

set -e

CONFIG=${1:-compare/paper_models_future_s43_all_available}

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export APPTAINERENV_OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export APPTAINERENV_MPLBACKEND=Agg

CONTAINER="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_ngc.sif"
CORRDIFF_DIR="$HOME/bjerknes/corrdiff"

echo "=================================================="
echo "Config: ${CONFIG}"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-48}"
echo "=================================================="

srun apptainer run \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/:/data/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
    ${CONTAINER} \
    python ${CORRDIFF_DIR}/compare_paper_models.py \
        --config-name ${CONFIG}
