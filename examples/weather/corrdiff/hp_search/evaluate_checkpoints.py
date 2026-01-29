#!/usr/bin/env python3
"""
Evaluate checkpoint combinations from the HP regression grid.

This script finds all regression checkpoints from hp_reg_grid and pairs them
with diffusion checkpoints (if any) for evaluation.

Usage:
    python hp_search/evaluate_checkpoints.py --list-only
    python hp_search/evaluate_checkpoints.py --output-script hp_search/slurm/eval_generated.sh
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import re


# Default generation configs to evaluate with
DEFAULT_GEN_CONFIGS = {
    "det_best_crps": {
        "generation.sampler.type": "deterministic",
        "generation.sampler.num_steps": 5,
        "generation.sampler.solver": "heun"
    },
    "det_best_rmse": {
        "generation.sampler.type": "deterministic",
        "generation.sampler.num_steps": 5,
        "generation.sampler.solver": "euler"
    },
    "det_standard": {
        "generation.sampler.type": "deterministic",
        "generation.sampler.num_steps": 9,
        "generation.sampler.solver": "euler"
    }
}


def find_regression_checkpoints(checkpoint_base: Path, min_steps: int = 0) -> Dict[str, dict]:
    """Find all regression checkpoints from hp_reg_grid."""

    reg_checkpoints = {}
    reg_grid_dir = checkpoint_base / "hp_reg_grid"

    if not reg_grid_dir.exists():
        print(f"Warning: hp_reg_grid directory not found: {reg_grid_dir}")
        return reg_checkpoints

    for model_dir in sorted(reg_grid_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        model_name = model_dir.name

        # Check for checkpoints in checkpoints_regression subdirectory
        ckpt_dir = model_dir / "checkpoints_regression"
        if not ckpt_dir.exists():
            ckpt_dir = model_dir  # Fallback to model dir

        # Find UNet checkpoints
        checkpoints = {}
        for ckpt_file in ckpt_dir.glob("UNet.*.mdlus"):
            match = re.search(r'UNet\.(\d+)\.(\d+)\.mdlus', ckpt_file.name)
            if match:
                step = int(match.group(2))
                if step >= min_steps:
                    checkpoints[step] = str(ckpt_file)

        if checkpoints:
            # Get the latest checkpoint
            latest_step = max(checkpoints.keys())
            reg_checkpoints[model_name] = {
                'dir': str(model_dir),
                'checkpoints': checkpoints,
                'latest_step': latest_step,
                'latest_path': checkpoints[latest_step]
            }

    return reg_checkpoints


def find_diffusion_checkpoints(checkpoint_base: Path, min_steps: int = 0) -> Dict[str, dict]:
    """Find all diffusion checkpoints from hp_diff_search."""

    diff_checkpoints = {}
    diff_dir = checkpoint_base / "hp_diff_search"

    if not diff_dir.exists():
        print(f"Note: hp_diff_search directory not found: {diff_dir}")
        return diff_checkpoints

    for trial_dir in sorted(diff_dir.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
            continue

        trial_name = trial_dir.name

        # Find EDMPrecond checkpoints
        checkpoints = {}
        for ckpt_file in trial_dir.glob("EDMPrecond*.mdlus"):
            match = re.search(r'\.(\d+)\.mdlus$', ckpt_file.name)
            if match:
                step = int(match.group(1))
                if step >= min_steps:
                    checkpoints[step] = str(ckpt_file)

        if checkpoints:
            latest_step = max(checkpoints.keys())
            diff_checkpoints[trial_name] = {
                'dir': str(trial_dir),
                'checkpoints': checkpoints,
                'latest_step': latest_step,
                'latest_path': checkpoints[latest_step]
            }

    return diff_checkpoints


def load_gen_configs(config_path: Optional[str]) -> Dict[str, dict]:
    """Load generation configs from JSON file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return configs
    return DEFAULT_GEN_CONFIGS


def check_already_evaluated(output_dir: Path, reg_name: str, diff_name: str, config_name: str) -> bool:
    """Check if this combination has already been evaluated."""
    if diff_name:
        metrics_file = output_dir / f"metrics_{reg_name}_{diff_name}_{config_name}.json"
    else:
        metrics_file = output_dir / f"metrics_{reg_name}_reg_only_{config_name}.json"
    return metrics_file.exists()


def create_evaluation_script(
    reg_checkpoints: Dict[str, dict],
    diff_checkpoints: Dict[str, dict],
    base_gen_config: str,
    output_dir: str,
    gen_configs: Dict[str, dict],
    num_gpus: int = 4,
    reg_only: bool = False
) -> str:
    """Create a bash script to evaluate checkpoint combinations."""

    # Build list of evaluation combinations
    eval_list = []

    if reg_only or not diff_checkpoints:
        # Regression-only evaluation (no diffusion)
        for reg_name, reg_info in reg_checkpoints.items():
            eval_list.append({
                'reg_name': reg_name,
                'reg_path': reg_info['latest_path'],
                'reg_step': reg_info['latest_step'],
                'diff_name': None,
                'diff_path': None,
                'diff_step': None
            })
    else:
        # Pair each diffusion checkpoint with regression checkpoints
        # For now, pair with all regression models (can be refined later)
        for diff_name, diff_info in diff_checkpoints.items():
            for reg_name, reg_info in reg_checkpoints.items():
                eval_list.append({
                    'reg_name': reg_name,
                    'reg_path': reg_info['latest_path'],
                    'reg_step': reg_info['latest_step'],
                    'diff_name': diff_name,
                    'diff_path': diff_info['latest_path'],
                    'diff_step': diff_info['latest_step']
                })

    total_evals = len(eval_list) * len(gen_configs)
    config_names = list(gen_configs.keys())

    script_lines = [
        "#!/bin/bash",
        "# Auto-generated checkpoint evaluation script",
        f"# {len(eval_list)} checkpoint combinations x {len(gen_configs)} configs = {total_evals} evaluations",
        f"# Generation configs: {', '.join(config_names)}",
        "",
        "FAILED_COUNT=0",
        "SUCCESS_COUNT=0",
        "SKIPPED_COUNT=0",
        "",
        f'OUTPUT_DIR="{output_dir}"',
        'mkdir -p "$OUTPUT_DIR"',
        "",
        "echo '========================================'",
        f"echo 'Evaluating {len(eval_list)} combinations x {len(gen_configs)} configs = {total_evals} total'",
        "echo '========================================'",
        "",
    ]

    eval_num = 0
    for i, ckpt in enumerate(eval_list):
        reg_name = ckpt['reg_name']
        reg_path = ckpt['reg_path']
        reg_step = ckpt['reg_step']
        diff_name = ckpt['diff_name']
        diff_path = ckpt['diff_path']
        diff_step = ckpt['diff_step']

        combo_id = f"{reg_name}_{diff_name}" if diff_name else f"{reg_name}_reg_only"

        script_lines.extend([
            f"echo ''",
            f"echo '========================================'",
            f"echo 'Combination [{i+1}/{len(eval_list)}]: {combo_id}'",
            f"echo '  Regression: {reg_name} (step {reg_step:,})'",
        ])

        if diff_name:
            script_lines.append(f"echo '  Diffusion:  {diff_name} (step {diff_step:,})'")
        else:
            script_lines.append(f"echo '  Diffusion:  NONE (regression only)'")

        script_lines.extend([
            f"echo '========================================'",
            "",
        ])

        for config_name, config_params in gen_configs.items():
            eval_num += 1
            output_file = f"$OUTPUT_DIR/output_{combo_id}_{config_name}.nc"
            metrics_file = f"$OUTPUT_DIR/metrics_{combo_id}_{config_name}.json"

            # Build overrides
            overrides = []
            for k, v in config_params.items():
                overrides.append(f"{k}={v}")
            overrides_str = " \\\n        ".join(overrides)

            # Base command
            if diff_path:
                cmd = f"""torchrun --nproc_per_node={num_gpus} generate_parallel_times.py \\
        --config-name=hp_base_gen \\
        generation.io.reg_ckpt_filename={reg_path} \\
        generation.io.res_ckpt_filename={diff_path} \\
        generation.io.output_filename={output_file} \\
        +generation.metrics_output={metrics_file} \\
        {overrides_str}"""
            else:
                # Regression-only: skip diffusion checkpoint
                cmd = f"""torchrun --nproc_per_node={num_gpus} generate_parallel_times.py \\
        --config-name=hp_base_gen \\
        generation.io.reg_ckpt_filename={reg_path} \\
        generation.io.output_filename={output_file} \\
        +generation.metrics_output={metrics_file} \\
        generation.sampler.type=deterministic \\
        generation.sampler.num_steps=1 \\
        {overrides_str}"""

            script_lines.extend([
                f"echo ''",
                f"echo '  [{eval_num}/{total_evals}] Config: {config_name}'",
                "",
                f'if [ -f "{metrics_file}" ]; then',
                f"    echo '    Already evaluated, skipping...'",
                f"    ((SKIPPED_COUNT++))",
                f"else",
                f"    if {cmd}; then",
                f"        echo '    Done!'",
                f"        ((SUCCESS_COUNT++))",
                f"    else",
                f"        echo '    FAILED!'",
                f"        ((FAILED_COUNT++))",
                f"    fi",
                f"fi",
                "",
            ])

    script_lines.extend([
        "echo ''",
        "echo '========================================'",
        "echo 'All evaluations complete!'",
        'echo "Success: $SUCCESS_COUNT / Skipped: $SKIPPED_COUNT / Failed: $FAILED_COUNT"',
        "echo '========================================'",
        "",
        "if [ $FAILED_COUNT -gt 0 ]; then",
        "    exit 1",
        "fi",
    ])

    return "\n".join(script_lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate HP grid checkpoints')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/checkpoints',
                        help='Base checkpoint directory')
    parser.add_argument('--base-gen-config', type=str,
                        default='conf/hp_base_gen.yaml',
                        help='Base generation config')
    parser.add_argument('--gen-configs', type=str,
                        default=None,
                        help='JSON file with generation configs (uses defaults if not provided)')
    parser.add_argument('--output-dir', type=str,
                        default='/outputs/hp_search/checkpoint_eval',
                        help='Output directory for results')
    parser.add_argument('--min-diff-steps', type=int, default=50000,
                        help='Minimum diffusion steps')
    parser.add_argument('--min-reg-steps', type=int, default=50000,
                        help='Minimum regression steps')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs')
    parser.add_argument('--output-script', type=str,
                        default='hp_search/slurm/eval_checkpoints_generated.sh',
                        help='Output script path')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list checkpoints')
    parser.add_argument('--reg-only', action='store_true',
                        help='Evaluate regression models only (no diffusion)')
    # Unused but kept for compatibility with run_checkpoint_eval.slurm
    parser.add_argument('--config-dir', type=str, default='hp_search_configs',
                        help='(Unused) Config directory')

    args = parser.parse_args()

    checkpoint_base = Path(args.checkpoint_dir)

    print("=" * 60)
    print("HP Grid Checkpoint Evaluation")
    print("=" * 60)
    print(f"Checkpoint dir: {checkpoint_base}")
    print(f"Min reg steps:  {args.min_reg_steps:,}")
    print(f"Min diff steps: {args.min_diff_steps:,}")
    print()

    # Find checkpoints
    print("Scanning for regression checkpoints...")
    reg_ckpts = find_regression_checkpoints(checkpoint_base, args.min_reg_steps)
    print(f"  Found {len(reg_ckpts)} regression models")

    print("Scanning for diffusion checkpoints...")
    diff_ckpts = find_diffusion_checkpoints(checkpoint_base, args.min_diff_steps)
    print(f"  Found {len(diff_ckpts)} diffusion trials")
    print()

    # Load generation configs
    gen_configs = load_gen_configs(args.gen_configs)
    print(f"Generation configs: {', '.join(gen_configs.keys())}")
    print()

    # Print summary
    print("=" * 60)
    print("Regression Checkpoints")
    print("=" * 60)
    for name, info in sorted(reg_ckpts.items()):
        steps = sorted(info['checkpoints'].keys())
        print(f"  {name}: {len(steps)} checkpoints (latest: {info['latest_step']:,})")

    if diff_ckpts:
        print()
        print("=" * 60)
        print("Diffusion Checkpoints")
        print("=" * 60)
        for name, info in sorted(diff_ckpts.items()):
            print(f"  {name}: step {info['latest_step']:,}")

    if args.list_only:
        return 0

    if not reg_ckpts:
        print("\nNo regression checkpoints found. Nothing to evaluate.")
        return 1

    # Create evaluation script
    print()
    print("=" * 60)
    print(f"Creating evaluation script: {args.output_script}")
    print("=" * 60)

    script = create_evaluation_script(
        reg_checkpoints=reg_ckpts,
        diff_checkpoints=diff_ckpts,
        base_gen_config=args.base_gen_config,
        output_dir=args.output_dir,
        gen_configs=gen_configs,
        num_gpus=args.num_gpus,
        reg_only=args.reg_only
    )

    output_path = Path(args.output_script)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script)
    os.chmod(output_path, 0o755)

    print(f"Script written to: {args.output_script}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
