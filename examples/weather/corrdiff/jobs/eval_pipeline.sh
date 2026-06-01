#!/bin/bash
# Re-evaluate all runs from saved predictions.nc on Helma.
# Run from: $HOME/bjerknes/corrdiff/
# Usage: bash jobs/eval_pipeline.sh

SIF=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_ngc.sif
EVAL_DIR=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff/eval

run() {
    local tag=$1
    echo "=== $tag ==="
    apptainer run $SIF python evaluate.py \
        --config-name=evaluate \
        run_tag="$tag" \
        "eval.predictions_file=$EVAL_DIR/$tag/predictions.nc" \
        generation.io.reg_ckpt_filename=/dev/null \
        generation.io.res_ckpt_filename=/dev/null
}

run baseline_noguide
run baseline_selfguide_g-0.25
run baseline_selfguide_g0.25
run baseline_selfguide_g0.5
run baseline_selfguide_g1.0
# run eval_edm2_v2_norwayW_295146
run high75_model_512
run high_lowguide
run high_model_512
run high_noguide
# # run log_model
# run log_model_512
# # run log_model_log_reg
# run low_model_512
# run smallguide_g0_25
# run smallguide_g1_0
# run smallguide_gn0_25
# run smallguide_gn0_25_alpha1_0
# run smallguide_gn0_25_alpha2_0
# run smallguide_gn0_25_alphaneg1_0
# run smallguide_gneg0_25_alpha1_0
# run smallguide_gneg0_25_alpha2_0
# run smallguide_gneg0_25_alphan1_0
# run smallguide_standalone
# run trial10
# run trial48
run trial8
