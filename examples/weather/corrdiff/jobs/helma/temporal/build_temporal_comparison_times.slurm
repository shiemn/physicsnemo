#!/bin/bash -l
#SBATCH -J temporal-common-times
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --output=output/common-times-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user "simon.hentschel@uni-wuerzburg.de"

set -euo pipefail

DOMAIN=${1:?"Usage: sbatch jobs/helma/temporal/build_temporal_comparison_times.slurm <europa|taiwan>"}
HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
CORRDIFF_DIR=$HOME/bjerknes/corrdiff
# Keep the historical artifact directory so existing generated files remain valid.
OUTPUT_DIR=$HNVME/outputs/corrdiff/eval/cross_domain_common_times
mkdir -p "$OUTPUT_DIR"

if [ "$DOMAIN" = "europa" ]; then
    # Metadata-only operation: use ZipStore directly instead of extracting the
    # 271 GB archive to node-local scratch.
    DATA_PATH=/data/Europe/wuerzburg450_corrdiff_v2.zarr.zip
    DATASET_CONFIG=$CORRDIFF_DIR/conf/base/dataset/europa.yaml
    BASE_TIMES=$CORRDIFF_DIR/conf/base/times/europa/random512_common3h.yaml
    OUTPUT_PATH=/outputs/eval/cross_domain_common_times/europa_random512_common12h.yaml
elif [ "$DOMAIN" = "taiwan" ]; then
    # Metadata-only operation: use ZipStore directly instead of extracting the
    # 1.1 TB archive to node-local scratch.
    DATA_PATH=/data/CorrDiff/cwa_dataset.zarr.zip
    DATASET_CONFIG=$CORRDIFF_DIR/conf/base/dataset/cwb.yaml
    BASE_TIMES=$CORRDIFF_DIR/conf/base/times/taiwan256.yaml
    OUTPUT_PATH=/outputs/eval/cross_domain_common_times/taiwan_random241_common12h.yaml
else
    echo "Unsupported domain: $DOMAIN" >&2
    exit 2
fi

export APPTAINERENV_OMP_NUM_THREADS=4
# This is a single-task metadata job. Run directly inside the batch allocation:
# using srun here conflicts with Helma's rewritten SLURM_CPUS_PER_TASK value on
# CPU nodes (SLURM_CPUS_PER_TASK=68 vs SLURM_TRES_PER_TASK=cpu=48).
apptainer run \
    --bind "$HNVME":/data/ \
    --bind "$HNVME/outputs/corrdiff":/outputs/ \
    "$HNVME/apptainer/corrdiff_ngc.sif" \
    env PYTHONPATH="$CORRDIFF_DIR" \
    python "$CORRDIFF_DIR/scripts/build_temporal_comparison_times.py" \
        --dataset-config "$DATASET_CONFIG" \
        --data-path "$DATA_PATH" \
        --base-times "$BASE_TIMES" \
        --output "$OUTPUT_PATH"

test -s "$OUTPUT_DIR/$(basename "$OUTPUT_PATH")"
