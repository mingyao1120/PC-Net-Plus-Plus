#!/usr/bin/env python3
"""Train PC-Net++ from scratch and report its performance.

One command per dataset: trains the model end to end (per-epoch validation
and model selection already use the complete CPR decision rule), then
evaluates the selected model and prints the PC-Net++ results.

    python run.py --dataset charades          # TT / NC / NW
    python run.py --dataset activitynet       # TT / NC / NW
    python run.py --dataset tacos
    python run.py --dataset ego4d
    python run.py --dataset qvhighlights

The required dataset features are listed in README.md. Results are written to
``outputs/<dataset>/``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from evaluate import DATASETS, ROOT, SRC, build_config, run_eval, write

REPORT_SPLITS = {
    'charades': ('tt', 'nc', 'nw'),
    'activitynet': ('tt', 'nc', 'nw'),
    'tacos': ('val',),
    'ego4d': ('test',),
    'qvhighlights': ('val',),
}


def train(dataset, output, gpu, limit_train_batches=None,
          limit_eval_batches=None):
    """Train from scratch; returns the run directory of the trained model."""
    config = build_config(dataset, output)
    command = [sys.executable, str(SRC/'train.py'),
               '--config-path', str(config), '--protocol', 'unified_paper',
               '--deterministic-eval-mask', '--output-root',
               str(output/'runner'), '--tag', f'pcnetpp_{dataset}']
    if limit_train_batches is not None:
        command.extend(['--limit-train-batches', str(limit_train_batches)])
    if limit_eval_batches is not None:
        command.extend(['--limit-eval-batches', str(limit_eval_batches)])
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONPATH': str(SRC),
                'PYTHONUNBUFFERED': '1',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    write(output/'train_command.json', {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'dataset': dataset, 'command': command,
    })
    subprocess.run(command, cwd=SRC, env=env, check=True)
    runs = sorted((output/'runner').glob('*/model-best.pt'),
                  key=lambda path: path.stat().st_mtime)
    if not runs:
        raise SystemExit('training produced no model-best.pt')
    return runs[-1].parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output-root', type=Path, default=None,
                        help='default: outputs/<dataset>')
    parser.add_argument('--eval-only', action='store_true',
                        help='skip training and re-evaluate the stored run')
    parser.add_argument('--limit-train-batches', type=int, default=None)
    parser.add_argument('--limit-eval-batches', type=int, default=None)
    args = parser.parse_args()

    output = (args.output_root.resolve() if args.output_root else
              ROOT/'outputs'/args.dataset)
    output.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        model = _latest_run(output)
    else:
        model = train(args.dataset, output, args.gpu,
                      args.limit_train_batches, args.limit_eval_batches)
    print('\nEvaluating PC-Net++ with %s' % (model/'model-best.pt'))
    results = {}
    for split in REPORT_SPLITS[args.dataset]:
        results[split] = run_eval(
            args.dataset, split, model/'model-best.pt',
            output/f'final_{split}', args.gpu, args.limit_eval_batches)
    write(output/'pcnetpp_results.json', {
        'dataset': args.dataset, 'model': str(model/'model-best.pt'),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'splits': results,
    })
    print('\nAll PC-Net++ results written to %s' %
          (output/'pcnetpp_results.json'))


def _latest_run(output):
    runs = sorted((output/'runner').glob('*/model-best.pt'),
                  key=lambda path: path.stat().st_mtime)
    if not runs:
        raise SystemExit('no trained model under %s; run without --eval-only'
                         % (output/'runner'))
    return runs[-1].parent


if __name__ == '__main__':
    main()
