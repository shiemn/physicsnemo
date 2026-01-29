#!/usr/bin/env python3
"""
Single Optuna trial for diffusion hyperparameter search.

This script runs ONE trial: it asks Optuna for hyperparameters, trains a model,
evaluates it, and reports the result back to the shared Optuna study.

Usage:
    python optuna_trial.py \
        --study-name my_study \
        --storage "sqlite:////path/to/study.db" \
        --wandb-project my-project

The script is designed to run as a single SLURM job within an array.
Multiple jobs coordinate through the shared Optuna SQLite database.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
from optuna.samplers import TPESampler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

CHECKPOINT_BASE = Path(os.environ.get('CHECKPOINT_BASE', '/checkpoints'))
REG_CHECKPOINT_DIR = CHECKPOINT_BASE / 'hp_reg_grid'
DIFF_CHECKPOINT_DIR = CHECKPOINT_BASE / 'hp_diff_search'

CORRDIFF_ROOT = Path(__file__).parent.parent

# Generation configs for evaluation (test with multiple samplers)
# Each config tests different sampler settings; final score is best across all
GENERATION_CONFIGS = {
    # Deterministic configs (fast, reproducible)
    "det_best_crps": {
        "description": "Deterministic, optimized for CRPS (Heun, 5 steps)",
        "overrides": {
            "generation.sampler.type": "deterministic",
            "generation.sampler.num_steps": 5,
            "generation.sampler.solver": "heun",
        }
    },
    "det_best_rmse": {
        "description": "Deterministic, optimized for RMSE (Euler, 5 steps)",
        "overrides": {
            "generation.sampler.type": "deterministic",
            "generation.sampler.num_steps": 5,
            "generation.sampler.solver": "euler",
        }
    },
    "det_standard": {
        "description": "Deterministic standard (Euler, 9 steps)",
        "overrides": {
            "generation.sampler.type": "deterministic",
            "generation.sampler.num_steps": 9,
            "generation.sampler.solver": "euler",
        }
    },
    # Stochastic configs (ensemble diversity)
    # Note: stochastic sampler doesn't have num_steps in schema, use ++ to force add
    "stoch_18": {
        "description": "Stochastic, 18 steps",
        "overrides": {
            "generation.sampler.type": "stochastic",
            "++generation.sampler.num_steps": 18,
        }
    },
    "stoch_50": {
        "description": "Stochastic, 50 steps (high quality)",
        "overrides": {
            "generation.sampler.type": "stochastic",
            "++generation.sampler.num_steps": 50,
        }
    },
}


# ==============================================================================
# Checkpoint Discovery
# ==============================================================================

def discover_regression_models() -> Dict[str, Dict]:
    """Find all available regression models and their checkpoints."""
    models = {}

    if not REG_CHECKPOINT_DIR.exists():
        logger.warning(f"Regression checkpoint directory not found: {REG_CHECKPOINT_DIR}")
        return models

    for model_dir in sorted(REG_CHECKPOINT_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        ckpt_subdir = model_dir / 'checkpoints_regression'
        if not ckpt_subdir.exists():
            ckpt_subdir = model_dir

        checkpoints = {}
        for ckpt_file in ckpt_subdir.glob('UNet.*.mdlus'):
            match = re.search(r'UNet\.(\d+)\.(\d+)\.mdlus', ckpt_file.name)
            if match:
                step = int(match.group(2))
                checkpoints[step] = str(ckpt_file)

        if checkpoints:
            models[model_dir.name] = {
                'config': model_dir.name,
                'dir': str(model_dir),
                'checkpoints': checkpoints
            }
            logger.info(f"Found model {model_dir.name} with {len(checkpoints)} checkpoints")

    return models


# ==============================================================================
# Training & Evaluation
# ==============================================================================

def run_training(
    config_name: str,
    reg_checkpoint: str,
    hparams: Dict[str, Any],
    num_nodes: int = 2,
    num_gpus: int = 4,
) -> bool:
    """Run diffusion training with torchrun (multi-node via srun + apptainer).
    
    Writes a bash script and executes it, allowing srun to work from outside container.
    """
    train_script = CORRDIFF_ROOT / 'train.py'
    
    # Get container and paths from environment (set by SLURM script)
    container = os.environ.get('HP_CONTAINER', '')
    corrdiff_dir = os.environ.get('HP_CORRDIFF_DIR', str(CORRDIFF_ROOT))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    if not container:
        logger.error("HP_CONTAINER environment variable not set for multi-node training")
        return False

    # Build the torchrun command arguments
    torchrun_args = [
        f'training.hp.lr={hparams["lr"]}',
        f'training.hp.total_batch_size={hparams["total_batch_size"]}',
        f'training.hp.training_duration={hparams["training_duration"]}',
        f'training.hp.distribution={hparams["distribution"]}',
        f'training.io.regression_checkpoint_path={reg_checkpoint}',
        f'training.io.checkpoint_dir={DIFF_CHECKPOINT_DIR}/{config_name}',
        f'hydra.job.name={config_name}',
    ]
    
    # Add distribution-specific parameters
    if 'student_t_nu' in hparams:
        torchrun_args.append(f'training.hp.student_t_nu={hparams["student_t_nu"]}')
    if 'P_mean' in hparams:
        torchrun_args.append(f'training.hp.P_mean={hparams["P_mean"]}')
    if 'P_std' in hparams:
        torchrun_args.append(f'training.hp.P_std={hparams["P_std"]}')

    # Build the full torchrun command to run inside container
    torchrun_cmd = (
        f"torchrun "
        f"--nnodes={num_nodes} "
        f"--nproc_per_node={num_gpus} "
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint={master_addr}:{master_port} "
        f"{train_script} "
        f"--config-name=hp_base_traindif "
        + " ".join(torchrun_args)
    )

    # Write bash script to shared location (accessible from host)
    script_path = Path(f'/tmp/hp_train_{config_name}.sh')
    script_content = f"""#!/bin/bash
srun --nodes={num_nodes} --ntasks-per-node=1 \\
    apptainer exec --nv \\
    --bind /scratch/:/data/ \\
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/ \\
    --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/ \\
    --bind {corrdiff_dir}:{corrdiff_dir} \\
    --pwd {corrdiff_dir} \\
    {container} \\
    bash -c "{torchrun_cmd}"
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    logger.info(f"Running training: {num_nodes} nodes x {num_gpus} GPUs")
    logger.info(f"  Script: {script_path}")
    
    try:
        # Execute the script from the host (works because /tmp is shared)
        result = subprocess.run(
            ['bash', str(script_path)],
            cwd=str(CORRDIFF_ROOT),
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with return code {e.returncode}")
        return False
    except FileNotFoundError as e:
        logger.error(f"Training failed: {e}")
        return False


def find_latest_checkpoint(trial_dir: Path) -> Optional[str]:
    """Find the latest diffusion checkpoint."""
    checkpoints = list(trial_dir.glob('EDMPrecondSuperResolution*.mdlus'))
    if not checkpoints:
        checkpoints = list(trial_dir.glob('*.mdlus'))

    if not checkpoints:
        return None

    def get_step(path):
        match = re.search(r'\.(\d+)\.mdlus$', path.name)
        return int(match.group(1)) if match else 0

    return str(max(checkpoints, key=get_step))


def run_single_evaluation(
    config_name: str,
    config_settings: Dict,
    reg_checkpoint: str,
    diff_checkpoint: str,
    num_nodes: int = 2,
    num_gpus: int = 4,
) -> Dict[str, float]:
    """Run evaluation with a single generation config (multi-node via srun + apptainer)."""
    gen_script = CORRDIFF_ROOT / 'generate_parallel_times.py'

    # Get container and paths from environment (set by SLURM script)
    container = os.environ.get('HP_CONTAINER', '')
    corrdiff_dir = os.environ.get('HP_CORRDIFF_DIR', str(CORRDIFF_ROOT))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    if not container:
        logger.error("HP_CONTAINER environment variable not set for multi-node evaluation")
        return {'error': True}

    # Build generation arguments
    gen_args = [
        f'generation.io.reg_ckpt_filename={reg_checkpoint}',
        f'generation.io.res_ckpt_filename={diff_checkpoint}',
        '+generation.compute_metrics=true',
        'generation.num_ensembles=10',
    ]
    
    # Add config-specific overrides
    for key, value in config_settings.get('overrides', {}).items():
        gen_args.append(f'{key}={value}')

    # Build the full torchrun command to run inside container
    torchrun_cmd = (
        f"torchrun "
        f"--nnodes={num_nodes} "
        f"--nproc_per_node={num_gpus} "
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint={master_addr}:{master_port} "
        f"{gen_script} "
        f"--config-name=hp_base_gen "
        + " ".join(gen_args)
    )

    # Use srun to launch apptainer on all nodes
    cmd = [
        'srun',
        '--nodes', str(num_nodes),
        '--ntasks-per-node', '1',
        'apptainer', 'exec', '--nv',
        '--bind', '/scratch/:/data/',
        '--bind', '/hnvme/workspace/b214cb11-helma-ecodata/downscaling/checkpoints/corrdiff:/checkpoints/',
        '--bind', '/hnvme/workspace/b214cb11-helma-ecodata/downscaling/outputs/corrdiff:/outputs/',
        '--bind', f'{corrdiff_dir}:{corrdiff_dir}',
        '--pwd', corrdiff_dir,
        container,
        'bash', '-c', torchrun_cmd
    ]

    logger.info(f"Running evaluation [{config_name}]")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CORRDIFF_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=7200
        )
        output = result.stdout + result.stderr
        metrics = {}
        
        # Parse metrics from output
        crps_match = re.search(r'CRPS:\s+([0-9.]+)', output)
        if crps_match:
            metrics['crps'] = float(crps_match.group(1))
        
        rmse_match = re.search(r'RMSE:\s+([0-9.]+)', output)
        if rmse_match:
            metrics['rmse'] = float(rmse_match.group(1))
            
        spread_match = re.search(r'Spread:\s+([0-9.]+)', output)
        if spread_match:
            metrics['spread'] = float(spread_match.group(1))
            
        bias_match = re.search(r'Bias:\s+([0-9.-]+)', output)
        if bias_match:
            metrics['bias'] = float(bias_match.group(1))
        
        if not metrics:
            return {'error': True}
        return metrics
    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for {config_name}")
        return {'error': True}
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        return {'error': True}


def run_evaluation(
    reg_checkpoint: str,
    diff_checkpoint: str,
    config_name: str,
    num_nodes: int = 2,
    num_gpus: int = 4,
) -> Dict[str, Any]:
    """Evaluate and return best metrics across all configs."""
    best_crps = float('inf')
    best_metrics = {}
    best_config = None

    for eval_config_name, config_settings in GENERATION_CONFIGS.items():
        logger.info(f"Evaluating: {eval_config_name}")
        metrics = run_single_evaluation(
            config_name=eval_config_name,
            config_settings=config_settings,
            reg_checkpoint=reg_checkpoint,
            diff_checkpoint=diff_checkpoint,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
        )

        if 'error' not in metrics:
            crps = metrics.get('crps', float('inf'))
            rmse = metrics.get('rmse')
            spread = metrics.get('spread')
            bias = metrics.get('bias')
            
            logger.info(f"  CRPS: {crps:.4f}" + 
                       (f", RMSE: {rmse:.4f}" if rmse else "") +
                       (f", Spread: {spread:.4f}" if spread else "") +
                       (f", Bias: {bias:.4f}" if bias else ""))
            
            if crps < best_crps:
                best_crps = crps
                best_metrics = metrics
                best_config = eval_config_name

    if best_config:
        best_metrics['best_config'] = best_config
        
    return best_metrics


# ==============================================================================
# Optuna Trial Objective
# ==============================================================================

class DiffusionTrial:
    """Single trial for Optuna HP search."""

    def __init__(
        self,
        regression_models: Dict[str, Dict],
        num_nodes: int = 2,
        num_gpus: int = 4,
    ):
        self.regression_models = regression_models
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus

        # Get list of model names
        self.model_names = sorted(regression_models.keys())

        # Build mapping of model -> available steps
        self.model_to_steps = {}
        for name in self.model_names:
            self.model_to_steps[name] = sorted(regression_models[name]['checkpoints'].keys())

        # Collect all steps and round to major milestones (100k increments)
        all_steps = set()
        for steps in self.model_to_steps.values():
            for step in steps:
                # Round to nearest 100k
                milestone = round(step / 100000) * 100000
                if milestone > 0:
                    all_steps.add(milestone)
        self.all_steps = sorted(all_steps)

        logger.info(f"Initialized with {len(self.model_names)} regression models")
        logger.info(f"  Models: {self.model_names}")
        logger.info(f"  Step milestones: {self.all_steps}")

    def __call__(self, trial: optuna.Trial) -> float:
        """Execute one trial."""

        # Sample regression model and step separately
        model_name = trial.suggest_categorical('reg_model', self.model_names)
        target_step = trial.suggest_categorical('reg_step', self.all_steps)

        # Find the closest available checkpoint step for this model
        available_steps = self.model_to_steps[model_name]
        checkpoint_step = min(available_steps, key=lambda s: abs(s - target_step))

        # Get the checkpoint path
        reg_checkpoint = self.regression_models[model_name]['checkpoints'][checkpoint_step]
        
        if checkpoint_step != target_step:
            logger.info(f"  Step {target_step} not available for {model_name}, using {checkpoint_step}")

        trial.set_user_attr('regression_model', model_name)
        trial.set_user_attr('regression_step', checkpoint_step)

        # Sample diffusion hyperparameters
        distribution = trial.suggest_categorical('distribution', ['gaussian', 'student_t'])
        
        hparams = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'total_batch_size': trial.suggest_categorical('total_batch_size', [128, 256, 512]),
            'training_duration': trial.suggest_int('training_duration_mult', 1, 3) * 1000,
            'distribution': distribution,
        }
        
        # Add distribution-specific parameters
        if distribution == 'student_t':
            hparams['student_t_nu'] = trial.suggest_int('student_t_nu', 2, 30)
        else:
            hparams['P_mean'] = trial.suggest_float('P_mean', -2.0, 2.0)
            hparams['P_std'] = trial.suggest_float('P_std', 0.5, 2.0)

        logger.info(f"Trial {trial.number}: {model_name} @ {checkpoint_step}")
        logger.info(f"  HP: lr={hparams['lr']:.2e}, batch={hparams['total_batch_size']}, dist={hparams['distribution']}")

        # Create checkpoint directory
        config_name = f'trial_{trial.number}'
        trial_dir = DIFF_CHECKPOINT_DIR / config_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Train
        success = run_training(
            config_name=config_name,
            reg_checkpoint=reg_checkpoint,
            hparams=hparams,
            num_nodes=self.num_nodes,
            num_gpus=self.num_gpus,
        )

        if not success:
            logger.warning(f"Trial {trial.number}: Training failed")
            return float('inf')

        # Find checkpoint
        diff_checkpoint = find_latest_checkpoint(trial_dir)
        if not diff_checkpoint:
            logger.warning(f"Trial {trial.number}: No checkpoint found")
            return float('inf')

        # Evaluate
        metrics = run_evaluation(
            reg_checkpoint=reg_checkpoint,
            diff_checkpoint=diff_checkpoint,
            config_name=config_name,
            num_nodes=self.num_nodes,
            num_gpus=self.num_gpus,
        )

        if not metrics:
            logger.warning(f"Trial {trial.number}: Evaluation failed")
            return float('inf')

        crps = metrics.get('crps', float('inf'))
        
        # Store all metrics as user attributes
        for key, value in metrics.items():
            trial.set_user_attr(key, value)

        logger.info(f"Trial {trial.number}: CRPS = {crps:.4f}")
        return crps


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run a single Optuna trial for diffusion HP search'
    )
    parser.add_argument('--study-name', required=True, help='Optuna study name')
    parser.add_argument('--storage', required=True, help='Optuna storage URL (e.g., sqlite:///path/to/study.db)')
    parser.add_argument('--num-nodes', type=int, default=2, help='Nodes per trial')
    parser.add_argument('--num-gpus', type=int, default=4, help='GPUs per node')
    parser.add_argument('--wandb-project', default='corrdiff-hp', help='WandB project name')
    parser.add_argument('--wandb-entity', help='WandB entity')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Diffusion HP Search - Single Trial")
    logger.info("=" * 60)
    logger.info(f"Study: {args.study_name}")
    logger.info(f"Storage: {args.storage}")
    logger.info(f"Nodes: {args.num_nodes}, GPUs/node: {args.num_gpus}")

    # Initialize WandB if available
    wandb_run = None
    if WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.study_name,
            reinit=True,
        )
        logger.info(f"WandB initialized: {args.wandb_project}")

    try:
        # Discover regression models
        logger.info("\nDiscovering regression models...")
        models = discover_regression_models()

        if not models:
            logger.error("No regression models found!")
            sys.exit(1)

        logger.info(f"Found {len(models)} regression models\n")

        # Setup Optuna storage with timeout for concurrent access
        if args.storage.startswith('sqlite:///'):
            storage = optuna.storages.RDBStorage(
                url=args.storage,
                engine_kwargs={"connect_args": {"timeout": 60}},
            )
        else:
            storage = args.storage

        # Create or load study
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5),
        )

        logger.info(f"Study loaded: {len(study.trials)} trials so far")

        # Create trial objective
        objective = DiffusionTrial(
            regression_models=models,
            num_nodes=args.num_nodes,
            num_gpus=args.num_gpus,
        )

        # Run ONE trial
        logger.info("\nRunning trial...")
        trial = study.ask()

        try:
            objective_value = objective(trial)
            study.tell(trial, objective_value)

            # Log to WandB
            if wandb_run:
                wandb.log({
                    'trial_number': trial.number,
                    'crps': objective_value,
                    **trial.user_attrs,  # Include all metrics (rmse, spread, bias, best_config, etc.)
                })

            logger.info(f"\nTrial {trial.number} completed with CRPS={objective_value:.4f}")

        except Exception as e:
            logger.error(f"Trial failed with exception: {e}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            if wandb_run:
                wandb.finish(exit_code=1)
            raise

        # Print best so far
        if study.best_trial:
            logger.info(f"\nBest trial so far: {study.best_trial.number}")
            logger.info(f"Best CRPS: {study.best_value:.4f}")
            logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

        if wandb_run:
            wandb.finish()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == '__main__':
    main()
