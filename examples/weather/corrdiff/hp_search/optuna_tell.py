#!/usr/bin/env python3
"""
Report trial results to Optuna and log to WandB.

Usage:
    python optuna_tell.py \
        --study-name my_study \
        --storage "sqlite:///path/to/study.db" \
        --trial-number 42 \
        --metrics-file /path/to/metrics.json \
        --params-file /path/to/params.json \
        [--failed]
"""

import argparse
import json
import logging
import sys

import optuna

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Report trial results to Optuna')
    parser.add_argument('--study-name', required=True, help='Optuna study name')
    parser.add_argument('--storage', required=True, help='Optuna storage URL')
    parser.add_argument('--trial-number', type=int, required=True, help='Trial number')
    parser.add_argument('--metrics-file', help='Metrics JSON file (from evaluation)')
    parser.add_argument('--params-file', help='Params JSON file (from optuna_ask)')
    parser.add_argument('--failed', action='store_true', help='Mark trial as failed')
    parser.add_argument('--wandb-project', default='corrdiff-hp-search', help='WandB project')
    parser.add_argument('--wandb-entity', help='WandB entity')
    args = parser.parse_args()

    logger.info(f"Study: {args.study_name}, Trial: {args.trial_number}")

    # Load params if provided
    params = {}
    if args.params_file:
        try:
            with open(args.params_file) as f:
                params = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load params file: {e}")

    # Load metrics if not failed
    metrics = {}
    if not args.failed and args.metrics_file:
        try:
            with open(args.metrics_file) as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Could not load metrics file: {e}")
            args.failed = True

    # Setup Optuna storage
    if args.storage.startswith('sqlite:///'):
        storage = optuna.storages.RDBStorage(
            url=args.storage,
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
    else:
        storage = args.storage

    # Load study
    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage,
    )

    # Verify trial exists
    trial_exists = any(t.number == args.trial_number for t in study.trials)
    if not trial_exists:
        logger.error(f"Trial {args.trial_number} not found in study")
        sys.exit(1)

    # Report results (use trial number, not FrozenTrial object)
    if args.failed:
        logger.info(f"Marking trial {args.trial_number} as FAILED")
        study.tell(args.trial_number, state=optuna.trial.TrialState.FAIL)
    else:
        crps = metrics.get('crps', float('inf'))
        logger.info(f"Trial {args.trial_number}: CRPS = {crps:.4f}")
        
        # Report the objective value
        study.tell(args.trial_number, crps)
        
        # Store metrics as user attributes (need to get the trial after tell)
        trial = study.trials[args.trial_number]
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                study.set_user_attr(f"trial_{args.trial_number}_{key}", value)

    # Log to WandB
    if WANDB_AVAILABLE and not args.failed:
        try:
            # Build config with hparams and regression model info
            wandb_config = params.get('hparams', {}).copy()
            wandb_config['reg_model'] = params.get('reg_model', '')
            wandb_config['reg_step'] = params.get('reg_step', 0)

            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.study_name,
                name=f"trial_{args.trial_number}",
                config=wandb_config,
                reinit=True,
            )
            
            # Log all metrics
            log_data = {
                'trial_number': args.trial_number,
                'reg_model': params.get('reg_model', ''),
                'reg_step': params.get('reg_step', 0),
                **metrics,
            }
            wandb.log(log_data)
            
            # Log best so far (only if a completed trial exists)
            try:
                best_trial = study.best_trial
                wandb.run.summary['best_crps'] = study.best_value
                wandb.run.summary['best_trial'] = best_trial.number
            except ValueError:
                pass  # No completed trials yet
            
            wandb.finish()
            logger.info("Logged to WandB")
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")

    # Print summary
    logger.info(f"Study now has {len(study.trials)} trials")
    try:
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number} with CRPS={study.best_value:.4f}")
    except ValueError:
        logger.info("No completed trials yet")


if __name__ == '__main__':
    main()
