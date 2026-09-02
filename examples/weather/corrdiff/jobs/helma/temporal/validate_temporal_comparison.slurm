#!/bin/bash -l
#SBATCH -J validate-comparison
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=output/validate-crps-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

set -euo pipefail

RUN_TAG=${1:?"Usage: sbatch jobs/helma/temporal/validate_temporal_comparison.slurm <run_tag> <expected_times>"}
EXPECTED_TIMES=${2:?"Usage: sbatch jobs/helma/temporal/validate_temporal_comparison.slurm <run_tag> <expected_times>"}
HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
CORRDIFF_DIR=$HOME/bjerknes/corrdiff
ARTIFACT_DIR=/outputs/eval/$RUN_TAG

apptainer run \
    --bind "$HNVME/outputs/corrdiff":/outputs/ \
    "$HNVME/apptainer/corrdiff_ngc.sif" \
    python "$CORRDIFF_DIR/scripts/validate_temporal_comparison.py" \
        --predictions "$ARTIFACT_DIR/predictions.nc" \
        --results "$ARTIFACT_DIR/eval_results.json" \
        --expected-times "$EXPECTED_TIMES" \
        --expected-members 10
