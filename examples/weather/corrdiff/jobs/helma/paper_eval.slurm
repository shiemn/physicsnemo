#!/bin/bash -l
#SBATCH -J PaperEval
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=output/paper-eval-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

# Run the offline paper-protocol evaluation on existing predictions.nc files.
#
# Usage:
#   sbatch jobs/helma/paper_eval.slurm <config_name> [extra hydra overrides...]

set -e

CONFIG=${1:?"Usage: sbatch paper_eval.slurm <config_name> [overrides...]"}
shift

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export APPTAINERENV_WANDB__SERVICE_WAIT=300
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
    python ${CORRDIFF_DIR}/evaluate_paper.py \
        --config-name ${CONFIG} \
        hydra.run.dir=/outputs/paper_eval/${CONFIG}_${SLURM_JOB_ID}/hydra \
        wandb.results_dir=/outputs/paper_eval/${CONFIG}_${SLURM_JOB_ID}/wandb \
        "$@"
