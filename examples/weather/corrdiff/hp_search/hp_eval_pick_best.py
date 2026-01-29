#!/usr/bin/env python3
"""
Pick the best result from multiple evaluation metrics files.

Usage:
    python hp_eval_pick_best.py \
        --metrics-dir /path/to/metrics/ \
        --output /path/to/best_metrics.json

Expected files in metrics-dir: det_heun_5.json, det_euler_5.json, etc.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Pick best result from multiple evals')
    parser.add_argument('--metrics-dir', required=True, help='Directory with metrics JSON files')
    parser.add_argument('--output', required=True, help='Output best metrics JSON path')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    
    best_crps = float('inf')
    best_config = None
    best_metrics = None
    all_results = {}

    for metrics_file in metrics_dir.glob('*.json'):
        config_name = metrics_file.stem
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            crps = metrics.get('crps', float('inf'))
            all_results[config_name] = metrics
            logger.info(f"{config_name}: CRPS={crps:.4f}")
            
            if crps < best_crps:
                best_crps = crps
                best_config = config_name
                best_metrics = metrics
                
        except Exception as e:
            logger.warning(f"Failed to load {metrics_file}: {e}")

    if best_metrics is None:
        logger.error("No valid metrics found!")
        output = {'success': False, 'error': 'No valid metrics found'}
    else:
        logger.info(f"Best config: {best_config} with CRPS={best_crps:.4f}")
        output = {
            'success': True,
            'best_config': best_config,
            **best_metrics,
        }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote best result to {args.output}")


if __name__ == '__main__':
    main()
