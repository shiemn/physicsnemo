#!/bin/bash -l
#SBATCH -J DiffHP
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --array=1-50%5
#SBATCH --output=hp_search/slurm/logs/trial-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user="simon.hentschel@uni-wuerzburg.de"

# =============================================================================
# Diffusion HP Search
# =============================================================================
#
# Architecture:
#   1. Container: Ask Optuna for params → params.json
#   2. Host srun: Train diffusion model (multi-node)
#   3. Host srun: Evaluate with multiple sampler configs (multi-node)
#   4. Container: Report results to Optuna & WandB
#
# Usage:
#   sbatch hp_search/slurm/run_trial.slurm
#   sbatch --array=51-100%4 hp_search/slurm/run_trial.slurm
#
# =============================================================================

set -e  # Exit on error

echo "=================================================="
echo "Diffusion HP Search - Trial ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "Job ID:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Nodes:       ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo "=================================================="

# =============================================================================
# Configuration
# =============================================================================

STUDY_NAME="${STUDY_NAME:-diff_hp_search}"
STORAGE_DIR="${STORAGE_DIR:-/hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_search}"
WANDB_PROJECT="${WANDB_PROJECT:-corrdiff-hp-search}"
NUM_NODES="${SLURM_NNODES:-2}"
NUM_GPUS="${NUM_GPUS:-4}"

# Paths
CONTAINER="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_infini.sif"
CORRDIFF_DIR="$HOME/bjerknes/corrdiff"
DIFF_CHECKPOINT_DIR="/hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_diff_search"

# Multi-node setup
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID % 100))

# Trial directories
TRIALS_DIR="${STORAGE_DIR}/trials"
mkdir -p "$TRIALS_DIR"
mkdir -p "$STORAGE_DIR"
mkdir -p "$DIFF_CHECKPOINT_DIR"
mkdir -p "hp_search/slurm/logs"

STORAGE="sqlite:///${STORAGE_DIR}/${STUDY_NAME}.db"

# Proxy setup
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export APPTAINERENV_WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONNOUSERSITE=1

export NCCL_DEBUG=WARN
export NCCL_PROTO=simple

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_HCA="=mlx5_2,mlx5_3,mlx5_4,mlx5_5"
export TRITON_CACHE_DIR="/tmp"

# Common apptainer bind mounts
BINDS="/scratch/:/data/,/hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/,/hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/,${STORAGE_DIR}:${STORAGE_DIR},${CORRDIFF_DIR}:${CORRDIFF_DIR}"

echo ""
echo "Configuration:"
echo "  Study:       $STUDY_NAME"
echo "  Storage:     $STORAGE"
echo "  Nodes:       $NUM_NODES"
echo "  GPUs/node:   $NUM_GPUS"
echo "  Master:      $MASTER_ADDR:$MASTER_PORT"
echo "  Container:   $CONTAINER"
echo ""

# =============================================================================
# Step 0: Copy data to local scratch
# =============================================================================

echo "[Step 0] Copying data to local scratch..."
srun --ntasks-per-node=1 /bin/bash ${CORRDIFF_DIR}/jobs/helma/copy.sh

# =============================================================================
# Step 1: Ask Optuna for hyperparameters
# =============================================================================

echo ""
echo "[Step 1] Asking Optuna for hyperparameters..."

PARAMS_FILE="${TRIALS_DIR}/trial_${SLURM_ARRAY_TASK_ID}_params.json"

apptainer exec --nv \
    --bind ${BINDS} \
    --pwd ${CORRDIFF_DIR} \
    ${CONTAINER} \
    python hp_search/optuna_ask.py \
        --study-name $STUDY_NAME \
        --storage $STORAGE \
        --output $PARAMS_FILE

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Failed to get parameters from Optuna"
    exit 1
fi

# Parse parameters from JSON
TRIAL_NUM=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['trial_number'])")
REG_CKPT=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['reg_checkpoint'])")
LR=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['lr'])")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['total_batch_size'])")
TRAIN_DUR=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['training_duration'])")
SIGMA_DATA=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['sigma_data'])")
DISTRIBUTION=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['distribution'])")

# Distribution-specific params
if [ "$DISTRIBUTION" = "student_t" ]; then
    STUDENT_T_NU=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['student_t_nu'])")
    DIST_ARGS="training.hp.distribution=student_t training.hp.student_t_nu=${STUDENT_T_NU}"
else
    P_MEAN=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['P_mean'])")
    P_STD=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE'))['hparams']['P_std'])")
    DIST_ARGS="training.hp.distribution=normal training.hp.P_mean=${P_MEAN} training.hp.P_std=${P_STD}"
fi

TRIAL_CKPT_DIR="${DIFF_CHECKPOINT_DIR}/trial_${TRIAL_NUM}"
mkdir -p "$TRIAL_CKPT_DIR"

echo "  Trial:       $TRIAL_NUM"
echo "  Reg ckpt:    $REG_CKPT"
echo "  LR:          $LR"
echo "  Batch size:  $BATCH_SIZE"
echo "  Duration:    $TRAIN_DUR"
echo "  Sigma data:  $SIGMA_DATA"
echo "  Distribution: $DISTRIBUTION"
echo "  Checkpoint:  $TRIAL_CKPT_DIR"

# =============================================================================
# Step 2: Train diffusion model
# =============================================================================

echo ""
echo "[Step 2] Training diffusion model..."

TRAIN_EXIT=0
srun --nodes=${NUM_NODES} --ntasks-per-node=1 \
    apptainer exec --nv \
    --bind ${BINDS} \
    --pwd ${CORRDIFF_DIR} \
    ${CONTAINER} \
    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${NUM_GPUS} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        train.py \
        --config-name=hp_base_traindif \
        training.hp.lr=${LR} \
        training.hp.total_batch_size=${BATCH_SIZE} \
        training.hp.training_duration=${TRAIN_DUR} \
        training.hp.sigma_data=${SIGMA_DATA} \
        ${DIST_ARGS} \
        training.io.regression_checkpoint_path=${REG_CKPT} \
        training.io.checkpoint_dir=${TRIAL_CKPT_DIR} \
        hydra.job.name=trial_${TRIAL_NUM} \
    || TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "WARNING: Training failed with exit code $TRAIN_EXIT"
fi

# =============================================================================
# Step 3: Find checkpoint and evaluate
# =============================================================================

echo ""
echo "[Step 3] Evaluating model..."

# Find latest diffusion checkpoint
DIFF_CKPT=$(find ${TRIAL_CKPT_DIR} -name "EDMPrecondSuperResolution*.mdlus" 2>/dev/null | sort -V | tail -1)

if [ -z "$DIFF_CKPT" ]; then
    DIFF_CKPT=$(find ${TRIAL_CKPT_DIR} -name "*.mdlus" 2>/dev/null | sort -V | tail -1)
fi

EVAL_EXIT=1
FINAL_METRICS="${TRIALS_DIR}/trial_${TRIAL_NUM}_metrics.json"

if [ -n "$DIFF_CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
    echo "  Diffusion checkpoint: $DIFF_CKPT"
    
    # Use hp_eval mode: runs multiple sampler configs internally and picks best
    # This is more efficient as it loads models once and runs all configs
    srun --nodes=${NUM_NODES} --ntasks-per-node=1 \
        apptainer exec --nv \
        --bind ${BINDS} \
        --pwd ${CORRDIFF_DIR} \
        ${CONTAINER} \
        torchrun \
            --nnodes=${NUM_NODES} \
            --nproc_per_node=${NUM_GPUS} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            generate_parallel_times.py \
            --config-name=hp_base_gen \
            generation.io.reg_ckpt_filename=${REG_CKPT} \
            generation.io.res_ckpt_filename=${DIFF_CKPT} \
            +generation.compute_metrics=true \
            +generation.skip_netcdf=true \
            +generation.hp_eval=true \
            +generation.metrics_output=${FINAL_METRICS} \
        && EVAL_EXIT=0 \
        || echo "  Warning: HP eval failed"
else
    echo "  No checkpoint found or training failed, skipping evaluation"
    FINAL_METRICS=""
fi

# =============================================================================
# Step 4: Report to Optuna
# =============================================================================

echo ""
echo "[Step 4] Reporting results to Optuna..."

TELL_ARGS="--study-name $STUDY_NAME --storage $STORAGE --trial-number $TRIAL_NUM --wandb-project $WANDB_PROJECT"

if [ -n "$FINAL_METRICS" ] && [ -f "$FINAL_METRICS" ] && [ $EVAL_EXIT -eq 0 ]; then
    TELL_ARGS="$TELL_ARGS --metrics-file $FINAL_METRICS --params-file $PARAMS_FILE"
else
    TELL_ARGS="$TELL_ARGS --failed"
fi

apptainer exec --nv \
    --bind ${BINDS} \
    --pwd ${CORRDIFF_DIR} \
    ${CONTAINER} \
    python hp_search/optuna_tell.py $TELL_ARGS

# =============================================================================
# Done
# =============================================================================

echo ""
echo "=================================================="
echo "Trial ${TRIAL_NUM} Complete"
echo "Training exit:   $TRAIN_EXIT"
echo "Evaluation exit: $EVAL_EXIT"
echo "Finished:        $(date)"
echo "=================================================="