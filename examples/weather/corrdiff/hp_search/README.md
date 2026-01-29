# Start Here - Simplified HP Search

## Quick Start (3 Steps)

### 1. Submit Your First Trial
```bash
# Run a single trial to test everything works
sbatch --array=1-1%1 hp_search/slurm/run_trial.slurm

# Or run your full experiment directly
sbatch --array=1-50%4 hp_search/slurm/run_trial.slurm
```

That's it! The trial will:
1. Ask Optuna for hyperparameters
2. Train a diffusion model (2 nodes, 8 GPUs)
3. Evaluate with 5 generation configs
4. Report results to WandB

### 2. Monitor Progress
```bash
# Watch trial output
tail -f hp_search/slurm/logs/trial-*.out

# Check which trials are running
squeue -u $USER -n DiffHP

# Check trial status in database
sqlite3 /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_search/diff_hp_search.db "
  SELECT COUNT(*) FROM trials WHERE state='COMPLETE';
"
```

### 3. View Results
Visit your WandB project: `https://wandb.ai/your-entity/corrdiff-hp-search`
- Each trial is a separate run
- Grouped by study name
- Auto-generated sweep visualization

---

## Configuration Options

### Study Name (Group your trials)
```bash
sbatch --export=STUDY_NAME=exp1_baseline hp_search/slurm/run_trial.slurm
```

### Parallel Workers (Change concurrency)
```bash
# 2 concurrent (slower, less database contention)
sbatch --array=1-50%2 hp_search/slurm/run_trial.slurm

# 8 concurrent (faster, more contention)
sbatch --array=1-50%8 hp_search/slurm/run_trial.slurm
```

### Number of Trials
```bash
# 20 trials total
sbatch --array=1-20%4 hp_search/slurm/run_trial.slurm

# 100 trials (continue later with more)
sbatch --array=1-100%4 hp_search/slurm/run_trial.slurm
```

### Add More Trials Later
```bash
# First batch: 50 trials
sbatch --array=1-50%4 hp_search/slurm/run_trial.slurm

# Wait for some to complete, then add more
sbatch --array=51-100%4 hp_search/slurm/run_trial.slurm

# Continue even further
sbatch --array=101-200%4 hp_search/slurm/run_trial.slurm
```

---

## Evaluation Configurations

The HP search tests **5 generation configs** and picks the best CRPS:

1. **det_best_crps** - Deterministic Heun, 5 steps (fast CRPS)
2. **det_best_rmse** - Deterministic Euler, 5 steps (fast RMSE)
3. **det_standard** - Deterministic Euler, 9 steps (standard)
4. **stoch_18** - Stochastic, 18 steps (ensemble)
5. **stoch_50** - Stochastic, 50 steps (high quality, slow)

**Final metric**: Best CRPS across all 5 configs

To modify evaluation configs, edit `optuna_trial.py`:
```python
GENERATION_CONFIGS = {
    "det_best_crps": {...},
    # Modify or add configs here
}
```

---

## Hyperparameter Search Space

Current ranges (edit `optuna_trial.py` to change):

```python
lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
training_duration = trial.suggest_categorical(
    'training_duration',
    [100000, 250000, 500000, 1000000]
)
P_mean = trial.suggest_float('P_mean', -2.0, 0.0)
P_std = trial.suggest_float('P_std', 0.5, 2.0)
```

### To Change Search Space:

Edit `optuna_trial.py` around line 465-495:

```python
def __call__(self, trial: optuna.Trial) -> float:
    # ...
    hparams = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),  # Changed bounds
        'batch_size': trial.suggest_categorical('batch_size', [64, 256, 512]),  # Added 64
        'training_duration': trial.suggest_categorical(
            'training_duration',
            [50000, 100000, 500000]  # Different durations
        ),
        # ... rest
    }
```

---

## Common Commands

### Start Experiments
```bash
# Single trial test
sbatch --array=1-1%1 hp_search/slurm/run_trial.slurm

# Full experiment (50 trials, 4 concurrent)
sbatch --array=1-50%4 hp_search/slurm/run_trial.slurm

# Custom study name
sbatch --export=STUDY_NAME=my_exp hp_search/slurm/run_trial.slurm

# Different parallelism
sbatch --array=1-50%8 hp_search/slurm/run_trial.slurm
```

### Monitor
```bash
# Watch real-time output
tail -f hp_search/slurm/logs/trial-*.out

# Check queue
squeue -u $USER -n DiffHP

# Cancel all
scancel -n DiffHP
```

### Query Results
```bash
# Best trial
sqlite3 /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_search/diff_hp_search.db "
  SELECT trial_id, value FROM trials ORDER BY value ASC LIMIT 1;
"

# All trials
sqlite3 /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_search/diff_hp_search.db "
  SELECT trial_id, value, state FROM trials ORDER BY trial_id;
"

# Statistics
sqlite3 /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_search/diff_hp_search.db "
  SELECT
    COUNT(*) as total,
    COUNT(CASE WHEN state='COMPLETE' THEN 1 END) as complete,
    COUNT(CASE WHEN state='FAIL' THEN 1 END) as failed,
    MIN(value) as best_crps
  FROM trials;
"
```

---

## What Happens When You Submit

1. **SLURM receives job**
   - Allocates 2 nodes, 4 GPUs each
   - Sets 24-hour time limit
   - Up to 4 array tasks run in parallel

2. **Each trial starts**
   - Copies data to local scratch
   - Starts apptainer container
   - Runs `optuna_trial.py`

3. **Inside trial**
   - Asks Optuna for next hyperparameters (from DB)
   - Creates training directory: `/checkpoints/hp_diff_search/trial_N/`
   - Runs: `srun torchrun train.py` (2 nodes × 4 GPUs)
   - Training takes 15-60 minutes depending on duration
   - Finds best checkpoint
   - Runs 5 evaluation configs: `torchrun generate.py`
   - Each eval takes 5-10 minutes
   - Computes CRPS for each config, picks best
   - Reports result to Optuna DB
   - Logs metrics to WandB
   - Trial completes ✓

4. **SLURM continues**
   - Next array task can start (if queued)
   - If timeout: job exits, trial marked as FAIL

---

## File Locations

```
Study database:
  /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff/hp_search/diff_hp_search.db

Trial checkpoints:
  /checkpoints/hp_diff_search/trial_0/
  /checkpoints/hp_diff_search/trial_1/
  ... etc

Job logs:
  hp_search/slurm/logs/trial-JOBID_1.out
  hp_search/slurm/logs/trial-JOBID_2.out
  ... etc

Production code:
  hp_search/optuna_trial.py
  hp_search/slurm/run_trial.slurm
```

---

## Troubleshooting

### No regression models found
Check:
```bash
ls /checkpoints/hp_reg_grid/  # Should have subdirectories
ls /checkpoints/hp_reg_grid/reg_01_baseline/checkpoints_regression/  # Should have UNet files
```

### Trials failing immediately
Check SLURM log:
```bash
cat hp_search/slurm/logs/trial-JOBID_1.out
```

### Database locked errors
Reduce concurrent workers:
```bash
sbatch --array=1-50%2 hp_search/slurm/run_trial.slurm
```

### WandB not showing runs
1. `wandb login`
2. Check project name matches: `--export=WANDB_PROJECT=my-project`

---

## Next Steps

1. **Try single trial**: `sbatch --array=1-1%1 hp_search/slurm/run_trial.slurm`
2. **Wait 30-60 minutes** for trial to complete
3. **Check output**: `tail hp_search/slurm/logs/trial-*.out`
4. **View results**: Open WandB dashboard
5. **Run full experiment**: `sbatch --array=1-50%4 hp_search/slurm/run_trial.slurm`

---

## System Overview

**Simple model**: One SLURM job = One trial

```
sbatch --array=1-50%4
    ↓
[SLURM array jobs 1-4 run, jobs 5+ wait in queue]
    ↓
Each job:
  - Allocates 2 nodes
  - Runs optuna_trial.py
  - Asks Optuna for HPs
  - Trains & evaluates
  - Reports result
    ↓
All trials coordinate via SQLite database
All results visible in WandB
```

---

## Features

✅ Bayesian optimization (Optuna TPE sampler)
✅ Multi-node training (2 nodes × 4 GPUs = 8 GPU)
✅ 5 evaluation configs (deterministic + stochastic)
✅ WandB logging & visualization
✅ Parallel workers (up to ~10 concurrent)
✅ Automatic job restart (submit more array tasks)
✅ Graceful timeout handling
✅ Database coordination (SQLite)

---

## Key Files

- **optuna_trial.py** - Main trial script (220 lines, easy to read)
- **run_trial.slurm** - SLURM job script (70 lines)

Modify these to customize:
- `optuna_trial.py` → change HP ranges, evaluation configs
- `run_trial.slurm` → change nodes, time, concurrency

---

## Questions?

- **How do I change HP ranges?** Edit `optuna_trial.py` in the `DiffusionTrial.__call__()` method
- **How do I add more trials?** Submit another array job: `sbatch --array=51-100%4 hp_search/slurm/run_trial.slurm`
- **How long does a trial take?** ~30-60 minutes (training + eval)
- **How do I view best results?** Check WandB or query SQLite database

**Ready to start? Run:**
```bash
sbatch --array=1-50%4 hp_search/slurm/run_trial.slurm
```

Then monitor with:
```bash
tail -f hp_search/slurm/logs/trial-*.out
```
