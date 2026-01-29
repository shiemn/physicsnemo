"""
Hyperparameter optimization module for CorrDiff.

This module provides tools for hyperparameter optimization of diffusion models
using pre-trained regression models.

Main scripts:
- optuna_trial.py: Single Optuna trial for SLURM array jobs

Usage (SLURM array + Optuna):
    sbatch hp_search/slurm/run_trial.slurm

"""

__all__ = []

# Lazy imports to avoid dependency issues
def _lazy_import():
    """Import main components (requires optuna)."""
    from .optuna_trial import (
        discover_regression_models,
        DiffusionTrial,
    )
    return {
        'discover_regression_models': discover_regression_models,
        'DiffusionTrial': DiffusionTrial,
    }

