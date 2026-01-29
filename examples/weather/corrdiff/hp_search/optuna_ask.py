#!/usr/bin/env python3
"""
Ask Optuna for hyperparameters and write them to a JSON file.

Usage:
    python optuna_ask.py \
        --study-name my_study \
        --storage "sqlite:///path/to/study.db" \
        --output /path/to/trial_params.json
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Checkpoint paths
CHECKPOINT_BASE = Path(os.environ.get('CHECKPOINT_BASE', '/checkpoints'))
REG_CHECKPOINT_DIR = CHECKPOINT_BASE / 'hp_reg_grid'


def discover_regression_models():
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
                'dir': str(model_dir),
                'checkpoints': checkpoints
            }

    return models


def sample_hyperparameters(trial, regression_models):
    """Sample hyperparameters for a trial."""
    
    # Regression model selection
    model_names = sorted(regression_models.keys())
    
    # Collect step milestones (rounded to 100k)
    all_steps = set()
    model_to_steps = {}
    for name in model_names:
        steps = sorted(regression_models[name]['checkpoints'].keys())
        model_to_steps[name] = steps
        for step in steps:
            milestone = round(step / 100000) * 100000
            if milestone > 0:
                all_steps.add(milestone)
    all_steps = sorted(all_steps)

    # Sample model and step
    model_name = trial.suggest_categorical('reg_model', model_names)
    target_step = trial.suggest_categorical('reg_step', all_steps)
    
    # Find closest available checkpoint
    available_steps = model_to_steps[model_name]
    checkpoint_step = min(available_steps, key=lambda s: abs(s - target_step))
    reg_checkpoint = regression_models[model_name]['checkpoints'][checkpoint_step]

    # Distribution type
    distribution = trial.suggest_categorical('distribution', ['normal', 'student_t'])

    # Core hyperparameters
    hparams = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'total_batch_size': trial.suggest_categorical('total_batch_size', [128, 256, 512]),
        'training_duration': trial.suggest_int('training_duration_mult', 1, 20) * 100000,
        'sigma_data': trial.suggest_float('sigma_data', 0.3, 1.0),
        'distribution': distribution,
    }

    # Distribution-specific parameters
    if distribution == 'student_t':
        hparams['student_t_nu'] = trial.suggest_int('student_t_nu', 3, 30)
    else:
        hparams['P_mean'] = trial.suggest_float('P_mean', -1.5, 1.5)
        hparams['P_std'] = trial.suggest_float('P_std', 0.8, 1.6)

    return {
        'trial_number': trial.number,
        'reg_checkpoint': reg_checkpoint,
        'reg_model': model_name,
        'reg_step': checkpoint_step,
        'target_step': target_step,
        'hparams': hparams,
    }


def main():
    parser = argparse.ArgumentParser(description='Ask Optuna for hyperparameters')
    parser.add_argument('--study-name', required=True, help='Optuna study name')
    parser.add_argument('--storage', required=True, help='Optuna storage URL')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()

    logger.info(f"Study: {args.study_name}")
    logger.info(f"Storage: {args.storage}")

    # Discover regression models
    regression_models = discover_regression_models()
    if not regression_models:
        logger.error("No regression models found!")
        sys.exit(1)
    logger.info(f"Found {len(regression_models)} regression models")

    # Setup Optuna storage
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
    logger.info(f"Study has {len(study.trials)} trials so far")

    # Ask for a new trial
    trial = study.ask()
    logger.info(f"Got trial {trial.number}")

    # Sample hyperparameters
    params = sample_hyperparameters(trial, regression_models)

    # Write to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Wrote params to {output_path}")
    logger.info(f"Trial {params['trial_number']}: {params['reg_model']} @ {params['reg_step']}")
    logger.info(f"  lr={params['hparams']['lr']:.2e}, batch={params['hparams']['total_batch_size']}, "
                f"dist={params['hparams']['distribution']}, sigma_data={params['hparams']['sigma_data']:.2f}")


if __name__ == '__main__':
    main()
