#!/usr/bin/env bash
# Submit the extra random-draw future-period regression evals.
#
# Assumes scripts/generate_future_time_configs.py has been run and synced first.
# Seed 42 is the already-completed draw; default submits only seeds 43..46.

set -euo pipefail

SEEDS="${SEEDS:-43 44 45 46}"
EXCLUDE="${EXCLUDE:-}"

declare -A PERIOD_YEARS=(
  [current_2005]="2005"
  [current_2004_2005]="2004,2005"
  [mid_start_2041_2042]="2041,2042"
  [mid_end_2059_2060]="2059,2060"
  [end_start_2081_2082]="2081,2082"
  [end_end_2099_2100]="2099,2100"
)

declare -A PERIOD_TIMES_PREFIX=(
  [current_2005]="random512_current_2005"
  [current_2004_2005]="random512_current_2004_2005"
  [mid_start_2041_2042]="random512_midcentury_start_2041_2042"
  [mid_end_2059_2060]="random512_midcentury_end_2059_2060"
  [end_start_2081_2082]="random512_endcentury_start_2081_2082"
  [end_end_2099_2100]="random512_endcentury_end_2099_2100"
)

declare -A MODEL_CONFIG=(
  [t0]="evaluate"
  [past3h]="evaluate_temporal_reg"
  [sym3h]="evaluate_temporal_reg"
  [reg07]="evaluate_temporal_reg"
  [past12h]="evaluate_temporal_reg"
)

declare -A MODEL_CKPT=(
  [t0]="/checkpoints/temporal_regression/norway_t0/checkpoints_regression/UNet.0.2000128.mdlus"
  [past3h]="/checkpoints/temporal_regression/norway_past3h/checkpoints_regression/UNet.0.2000128.mdlus"
  [sym3h]="/checkpoints/temporal_regression/norway_sym3h/checkpoints_regression/UNet.0.2000128.mdlus"
  [reg07]="/checkpoints/hp_reg_grid/reg_07_flexi_p125_temporal/checkpoints_regression/UNet.0.2000128.mdlus"
  [past12h]="/checkpoints/temporal_regression/norway_past12h/checkpoints_regression/UNet.0.2000128.mdlus"
)

declare -A MODEL_OFFSETS=(
  [past3h]="[-3,0]"
  [sym3h]="[-3,0,3]"
  [reg07]="[-3,0,3]"
  [past12h]="[-12,-6,-3,0]"
)

MODELS=(${MODELS:-t0 past3h sym3h reg07 past12h})
PERIODS=(${PERIODS:-current_2005 current_2004_2005 mid_start_2041_2042 mid_end_2059_2060 end_start_2081_2082 end_end_2099_2100})

for seed in ${SEEDS}; do
  for period in "${PERIODS[@]}"; do
    years="${PERIOD_YEARS[$period]}"
    if [[ "${seed}" == "42" ]]; then
      times="${PERIOD_TIMES_PREFIX[$period]}_24h_compatible"
    else
      times="${PERIOD_TIMES_PREFIX[$period]}_seed${seed}_24h_compatible"
    fi
    for model in "${MODELS[@]}"; do
      tag="future_conf_s${seed}_${model}_${period}"
      output="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff/eval/${tag}"
      if [[ -s "${output}/eval_results.json" && -s "${output}/predictions.nc" ]]; then
        echo "SKIP ${tag}: output already exists"
        continue
      fi

      sbatch_args=()
      if [[ -n "${EXCLUDE}" ]]; then
        sbatch_args+=(--exclude="${EXCLUDE}")
      fi

      args=(
        jobs/helma/eval.slurm
        "${MODEL_CONFIG[$model]}"
        "${tag}"
        "times=future_confidence/${times}"
        "dataset.years=[${years}]"
        "generation.io.reg_ckpt_filename=${MODEL_CKPT[$model]}"
        "generation.io.res_ckpt_filename=/dev/null"
        "generation.inference_mode=regression"
        "generation.num_ensembles=1"
        "generation.seed_batch_size=1"
      )

      if [[ "${model}" != "t0" ]]; then
        args+=("dataset.temporal_inputs.offset_hours=${MODEL_OFFSETS[$model]}")
      fi

      job_id=$(sbatch --parsable "${sbatch_args[@]}" "${args[@]}")
      echo "${job_id},${tag},${model},${period},${seed}"
    done
  done
done
