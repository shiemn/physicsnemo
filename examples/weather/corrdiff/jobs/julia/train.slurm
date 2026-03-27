#!/bin/bash
#SBATCH -J CorrDiff
#SBATCH -c 16
#SBATCH -p h100
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out

# Usage: sbatch train.slurm <config-name> [--direct]
#   --direct  bind home storage directly as /data (default: copy to node-local /tmp first)
# Override GPU count: sbatch --gres=gpu:1 train.slurm <config>
# Note: scratch mode requires reserving tmp space: sbatch --tmp=3000G train.slurm <config>

export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

DATA_HOME=/home/sih25nq/data/downscaling
DATA_MODE=${2:-scratch}

if [ "$DATA_MODE" = "--direct" ]; then
    echo "Data mode: direct (binding home storage)"
    DATA_BIND="${DATA_HOME}"
else
    echo "Data mode: scratch (copying to /tmp)"
    DATA_PATH=Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH
    mkdir -p /tmp/${DATA_PATH}
    # Copy static files
    cp ${DATA_HOME}/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc \
       /tmp/Norway/HCLIM3/.
    cp ${DATA_HOME}/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc \
       /tmp/Norway/HCLIM3/preprocessed_large/.
    # Copy .pt files and training years (1980-2005) in parallel
    cp ${DATA_HOME}/${DATA_PATH}/*.pt /tmp/${DATA_PATH}/.
    find ${DATA_HOME}/${DATA_PATH} -name '*.h5' \
        | awk -F '[_.]' '{ year=$(NF-1); if (year >= 1980 && year <= 2005) print $0 }' \
        | xargs -P 5 -I{} rsync --info=progress2 {} /tmp/${DATA_PATH}/.
    wait
    DATA_BIND="/tmp"
fi

srun apptainer run --nv \
    --bind ${DATA_BIND}/:/data \
    --bind /data/42-julia-hpc-rz-lsx/sih25nq/checkpoints/corrdiff:/checkpoints/ \
    ~/corrdiff_ngc.sif \
    torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE ~/bjerknes/corrdiff/train.py --config-name $1
