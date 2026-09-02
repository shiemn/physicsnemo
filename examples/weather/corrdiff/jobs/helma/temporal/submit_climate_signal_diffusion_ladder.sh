#!/usr/bin/env bash
# Submit missing matched-2M climate chunks model-by-model, with validation and
# analysis gates between models. The completed sym3h chunks are reused.

set -euo pipefail

ACCOUNT=${ACCOUNT:-b214cb}
OUTPUT_ROOT=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff/climate_signal

MODELS=(t0_2m past3h past12h)
declare -A CONFIG=(
  [t0_2m]=eval/norway/climate_signal_t0_2m
  [past3h]=eval/norway/climate_signal_past3h
  [past12h]=eval/norway/climate_signal_past12h
)
declare -A LABEL=(
  [t0_2m]="t0 matched 2M"
  [past3h]="Past 3 h"
  [past12h]="Past 12 h"
)
declare -A PREFIX=(
  [t0_2m]=climate_signal_t0_2m
  [past3h]=climate_signal_past3h
  [past12h]=climate_signal_past12h
)

CHUNKS=(hist_1986_1995 hist_1996_2005 mid_2041_2050 mid_2051_2060 end_2081_2090 end_2091_2100)
declare -A YEARS=(
  [hist_1986_1995]="1986,1987,1988,1989,1990,1991,1992,1993,1994,1995"
  [hist_1996_2005]="1996,1997,1998,1999,2000,2001,2002,2003,2004,2005"
  [mid_2041_2050]="2041,2042,2043,2044,2045,2046,2047,2048,2049,2050"
  [mid_2051_2060]="2051,2052,2053,2054,2055,2056,2057,2058,2059,2060"
  [end_2081_2090]="2081,2082,2083,2084,2085,2086,2087,2088,2089,2090"
  [end_2091_2100]="2091,2092,2093,2094,2095,2096,2097,2098,2099,2100"
)
declare -A FIRST=(
  [hist_1986_1995]=1986-01-02T00:00:00 [hist_1996_2005]=1996-01-02T00:00:00
  [mid_2041_2050]=2041-01-02T00:00:00 [mid_2051_2060]=2051-01-02T00:00:00
  [end_2081_2090]=2081-01-02T00:00:00 [end_2091_2100]=2091-01-02T00:00:00
)
declare -A LAST=(
  [hist_1986_1995]=1995-12-30T21:00:00 [hist_1996_2005]=2005-12-30T21:00:00
  [mid_2041_2050]=2050-12-30T21:00:00 [mid_2051_2060]=2060-12-30T21:00:00
  [end_2081_2090]=2090-12-30T21:00:00 [end_2091_2100]=2100-12-30T21:00:00
)

# Refuse collisions before creating any jobs.
for model in "${MODELS[@]}"; do
  for chunk in "${CHUNKS[@]}"; do
    tag=${PREFIX[$model]}_${chunk}
    if [[ -e ${OUTPUT_ROOT}/${tag} ]]; then
      echo "REFUSE ${tag}: output path already exists" >&2
      exit 2
    fi
  done
done

previous_analysis=""
for model in "${MODELS[@]}"; do
  jobs=()
  for chunk in "${CHUNKS[@]}"; do
    tag=${PREFIX[$model]}_${chunk}
    submit=(sbatch --parsable --account="${ACCOUNT}")
    if [[ -n ${previous_analysis} ]]; then
      submit+=(--dependency="afterok:${previous_analysis}")
    fi
    job_id=$("${submit[@]}" jobs/helma/temporal/climate_eval.slurm "${CONFIG[$model]}" "${tag}" \
      "dataset.years=[${YEARS[$chunk]}]" \
      "generation.times_range=[${FIRST[$chunk]},${LAST[$chunk]},3]")
    jobs+=("${job_id}")
    echo "GEN,${job_id},${model},${chunk},${tag}"
  done
  dependency=$(IFS=:; echo "${jobs[*]}")
  analysis_id=$(sbatch --parsable --account="${ACCOUNT}" --dependency="afterok:${dependency}" \
    scripts/analysis/slurm/analyze.slurm climate-model \
    "${model}" "${LABEL[$model]}" "${PREFIX[$model]}")
  echo "ANALYZE,${analysis_id},${model},afterok:${dependency}"
  previous_analysis=${analysis_id}
done

comparison_id=$(sbatch --parsable --account="${ACCOUNT}" --dependency="afterok:${previous_analysis}" \
  scripts/analysis/slurm/analyze.slurm climate-compare)
echo "COMPARE,${comparison_id},afterok:${previous_analysis}"
