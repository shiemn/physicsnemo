#!/bin/bash
# Submit the four production past-only diffusion runs after regression QA approval.

set -euo pipefail

ROOT=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff
cd "$HOME/bjerknes/corrdiff"

for checkpoint in \
    "$ROOT/temporal_regression/europa_past3h/checkpoints_regression/UNet.0.2000128.mdlus" \
    "$ROOT/temporal_regression/europa_past12h/checkpoints_regression/UNet.0.2000128.mdlus" \
    "$ROOT/taiwan_regression_past3h/checkpoints_regression/UNet.0.2000128.mdlus" \
    "$ROOT/taiwan_regression_past12h/checkpoints_regression/UNet.0.2000128.mdlus"; do
    if [ ! -s "$checkpoint" ]; then
        echo "Refusing to submit: regression checkpoint missing: $checkpoint" >&2
        exit 2
    fi
done

for dir in \
    "$ROOT/temporal_diffusion/europa_past3h" \
    "$ROOT/temporal_diffusion/europa_past12h" \
    "$ROOT/taiwan_diffusion_past3h" \
    "$ROOT/taiwan_diffusion_past12h"; do
    if [ -e "$dir" ]; then
        echo "Refusing to submit: production target already exists: $dir" >&2
        exit 2
    fi
done

sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-eu-p3-diff jobs/helma/multi_node.slurm \
    train/europa/temporal_diff_past3h europa
sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-eu-p12-diff jobs/helma/multi_node.slurm \
    train/europa/temporal_diff_past12h europa
sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-tw-p3-diff jobs/helma/multi_node.slurm \
    train/taiwan/temporal_diff_past3h taiwan
sbatch --account=b214cb --nodes=4 --time=24:00:00 \
    --job-name=past-tw-p12-diff jobs/helma/multi_node.slurm \
    train/taiwan/temporal_diff_past12h taiwan
