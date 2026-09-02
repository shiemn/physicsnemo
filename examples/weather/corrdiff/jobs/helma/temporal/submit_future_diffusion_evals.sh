#!/usr/bin/env bash
# Submit the missing seed-43, 10-member future diffusion pools.
#
# Usage:
#   DEPENDENCY_JOB_ID=<t0-training-job> bash jobs/helma/temporal/submit_future_diffusion_evals.sh

set -euo pipefail

DEPENDENCY_JOB_ID=${DEPENDENCY_JOB_ID:?Set DEPENDENCY_JOB_ID to the matched t0 diffusion training job}
OUTPUT_ROOT=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff/eval

declare -A PERIOD_YEARS=(
  [current_2005]="2005"
  [mid_start_2041_2042]="2041,2042"
  [mid_end_2059_2060]="2059,2060"
  [end_start_2081_2082]="2081,2082"
  [end_end_2099_2100]="2099,2100"
)

declare -A PERIOD_TIMES=(
  [current_2005]="random512_current_2005_seed43_24h_compatible"
  [mid_start_2041_2042]="random512_midcentury_start_2041_2042_seed43_24h_compatible"
  [mid_end_2059_2060]="random512_midcentury_end_2059_2060_seed43_24h_compatible"
  [end_start_2081_2082]="random512_endcentury_start_2081_2082_seed43_24h_compatible"
  [end_end_2099_2100]="random512_endcentury_end_2099_2100_seed43_24h_compatible"
)

declare -A MODEL_CONFIG=(
  [t0]="evaluate"
  [past3h]="evaluate_temporal"
  [past12h]="evaluate_temporal"
)

declare -A MODEL_REG=(
  [t0]="/checkpoints/temporal_regression/norway_t0/checkpoints_regression/UNet.0.2000128.mdlus"
  [past3h]="/checkpoints/temporal_regression/norway_past3h/checkpoints_regression/UNet.0.2000128.mdlus"
  [past12h]="/checkpoints/temporal_regression/norway_past12h/checkpoints_regression/UNet.0.2000128.mdlus"
)

declare -A MODEL_DIFF=(
  [t0]="/checkpoints/tr_diff_t0_ladder/checkpoints_diffusion/EDMPrecondSuperResolution.0.2000000.mdlus"
  [past3h]="/checkpoints/tr_diff_temporal_past3h/checkpoints_diffusion/EDMPrecondSuperResolution.0.2000000.mdlus"
  [past12h]="/checkpoints/tr_diff_temporal_past12h/checkpoints_diffusion/EDMPrecondSuperResolution.0.2000000.mdlus"
)

declare -A MODEL_OFFSETS=(
  [past3h]="[-3,0]"
  [past12h]="[-12,-6,-3,0]"
)

MODELS=(t0 past3h past12h)
PERIODS=(current_2005 mid_start_2041_2042 mid_end_2059_2060 end_start_2081_2082 end_end_2099_2100)

for model in "${MODELS[@]}"; do
  for period in "${PERIODS[@]}"; do
    tag="future_conf_s43_${model}_diffusion_${period}"
    output=${OUTPUT_ROOT}/${tag}
    if [[ -s ${output}/eval_results.json && -s ${output}/predictions.nc ]]; then
      echo "SKIP ${tag}: complete output exists"
      continue
    fi
    if [[ -e ${output} ]]; then
      echo "REFUSE ${tag}: output path exists but is incomplete" >&2
      exit 2
    fi

    args=(
      jobs/helma/eval.slurm
      "${MODEL_CONFIG[$model]}"
      "${tag}"
      "times=${PERIOD_TIMES[$period]}"
      "dataset.years=[${PERIOD_YEARS[$period]}]"
      "generation.io.reg_ckpt_filename=${MODEL_REG[$model]}"
      "generation.io.res_ckpt_filename=${MODEL_DIFF[$model]}"
      generation.inference_mode=all
      generation.num_ensembles=10
      generation.seed_batch_size=10
      eval.n_plot_events=0
    )
    if [[ ${model} != t0 ]]; then
      args+=("dataset.temporal_inputs.offset_hours=${MODEL_OFFSETS[$model]}")
    fi

    job_id=$(sbatch --parsable --account=b214cb --dependency="afterok:${DEPENDENCY_JOB_ID}" "${args[@]}")
    echo "${job_id},${tag},${model},${period},afterok:${DEPENDENCY_JOB_ID}"
  done
done
