#!/bin/bash -l
#SBATCH -J Analysis
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=output/analysis-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

set -euo pipefail

usage() {
    echo "Usage:" >&2
    echo "  sbatch scripts/analysis/slurm/analyze.slurm annual <europa|taiwan>" >&2
    echo "  sbatch scripts/analysis/slurm/analyze.slurm europa-extremes" >&2
    exit 2
}

ANALYSIS=${1:-}
[ -n "$ANALYSIS" ] || usage
shift

CONTAINER="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_ngc.sif"
CORRDIFF_DIR="$HOME/bjerknes/corrdiff"
EVAL_ROOT="/outputs/eval"
export APPTAINERENV_MPLBACKEND=Agg
export APPTAINERENV_OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}

case "$ANALYSIS" in
    annual)
        DOMAIN=${1:-}
        [ -n "$DOMAIN" ] || usage
        shift
        [ "$#" -eq 0 ] || usage

        if [ "$DOMAIN" = "europa" ]; then
            CHANNEL="precipitation_amount_1hr"
            TITLE="Europe precipitation errors through 2021 (matched diffusion maturity)"
            UNITS="mm h-1"
            BASELINE_TAG="timeseries_europa_2021_common_baseline_diff1m_matched"
        elif [ "$DOMAIN" = "taiwan" ]; then
            CHANNEL="maximum_radar_reflectivity"
            TITLE="Taiwan radar-reflectivity errors through 2021 (matched diffusion maturity)"
            UNITS="dBZ"
            BASELINE_TAG="timeseries_taiwan_2021_common_baseline_diff2m_matched"
        else
            usage
        fi

        OUTPUT="$EVAL_ROOT/annual_performance_2021_${DOMAIN}_matched"
        SCRIPT="$CORRDIFF_DIR/scripts/analysis/analyze_annual_performance.py"
        ARGS=(
            --channel "$CHANNEL"
            --title "$TITLE"
            --units "$UNITS"
            --output-dir "$OUTPUT"
            --time-alignment intersection
            --model "Baseline=$EVAL_ROOT/$BASELINE_TAG/predictions.nc"
            --model "Symmetric 1h=$EVAL_ROOT/timeseries_${DOMAIN}_2021_common_sym1h/predictions.nc"
            --model "Symmetric 3h=$EVAL_ROOT/timeseries_${DOMAIN}_2021_common_sym3h/predictions.nc"
        )
        ;;
    europa-extremes)
        [ "$#" -eq 0 ] || usage
        OUTPUT="$EVAL_ROOT/annual_performance_2021_europa_extremes"
        SCRIPT="$CORRDIFF_DIR/scripts/analysis/analyze_europa_extreme_cases.py"
        ARGS=(
            --output-dir "$OUTPUT"
            --case-count 6
            --separation-hours 24
            --tail-fraction 0.001
            --model "Baseline=$EVAL_ROOT/timeseries_europa_2021_common_baseline_diff1m_matched/predictions.nc"
            --model "Symmetric 1h=$EVAL_ROOT/timeseries_europa_2021_common_sym1h/predictions.nc"
            --model "Symmetric 3h=$EVAL_ROOT/timeseries_europa_2021_common_sym3h/predictions.nc"
        )
        ;;
    *)
        usage
        ;;
esac

srun apptainer run \
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \
    "$CONTAINER" \
    python "$SCRIPT" "${ARGS[@]}"
