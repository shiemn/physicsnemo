#!/bin/bash
# Submit the four production past-only regression runs after smoke validation.

set -euo pipefail

ROOT=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff
cd "$HOME/bjerknes/corrdiff"

for dir in \
    "$ROOT/temporal_regression/europa_past3h" \
    "$ROOT/temporal_regression/europa_past12h" \
    "$ROOT/taiwan_regression_past3h" \
    "$ROOT/taiwan_regression_past12h"; do
    if [ -e "$dir" ]; then
        echo "Refusing to submit: production target already exists: $dir" >&2
        exit 2
    fi
done

sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-eu-p3-reg jobs/helma/multi_node.slurm \
    train/europa/temporal_reg_past3h europa
sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-eu-p12-reg jobs/helma/multi_node.slurm \
    train/europa/temporal_reg_past12h europa
sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-tw-p3-reg jobs/helma/multi_node.slurm \
    train/taiwan/temporal_reg_past3h taiwan
sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-tw-p12-reg jobs/helma/multi_node.slurm \
    train/taiwan/temporal_reg_past12h taiwan
