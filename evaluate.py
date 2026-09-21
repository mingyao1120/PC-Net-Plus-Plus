#!/usr/bin/env python3
"""Evaluate a trained PC-Net++ model with full CPR inference.

This is the evaluation half of :mod:`run`; it loads a model trained by
``run.py`` (or any ``model-best.pt`` produced by ``src/train.py``) and
reports the PC-Net++ metrics.  Every metric is computed through the
complete decision rule (counterfactual re-scoring + compatible boundary
consensus; the set-level rule on QVHighlights).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT/'src'

DATASETS = ('charades', 'activitynet', 'tacos', 'ego4d', 'qvhighlights')
DEFAULT_SPLIT = {'charades': 'tt', 'activitynet': 'tt', 'tacos': 'val',
                 'ego4d': 'test', 'qvhighlights': 'val'}
SPLIT_DATA = {
    ('charades', 'tt'): None,
    ('charades', 'nc'): 'data/charades/novel_comp.json',
    ('charades', 'nw'): 'data/charades/novel_word.json',
    ('activitynet', 'tt'): None,
    ('activitynet', 'nc'): 'data/activitynet/novel_comp.json',
    ('activitynet', 'nw'): 'data/activitynet/novel_word.json',
    ('tacos', 'val'): None,
    ('ego4d', 'test'): None,
    ('qvhighlights', 'val'): None,
}
PATH_KEYS = ('train_data', 'test_data', 'val_data', 'vocab_path',
             'feature_path', 'sampled_feature_cache',
             'sampled_feature_index', 'qid2saliency')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')


def resolve_paths(config):
    values = config['dataset']
    for key in PATH_KEYS:
        path = values.get(key)
        if path and not Path(path).is_absolute():
            values[key] = str((ROOT/path).resolve())
    gt = config.get('evaluation', {}).get('qv_gt_jsonl')
    if gt and not Path(gt).is_absolute():
        config['evaluation']['qv_gt_jsonl'] = str((ROOT/gt).resolve())


def cpr_candidate(config):
    """Return the CPR configuration declared by the dataset config."""
    cpr = copy.deepcopy(config.get('cpr'))
    if not cpr or 'family' not in cpr:
        raise ValueError('dataset config is missing its "cpr" section')
    return dict(cpr, id='pcnetpp', mode='raw', views=1)


def build_config(dataset, output, test_data=None):
    config = copy.deepcopy(read(ROOT/'configs'/f'{dataset}.json'))
    resolve_paths(config)
    if test_data is not None:
        config['dataset']['test_data'] = str((ROOT/test_data).resolve())
        config['protocol']['selection_split'] = 'test'
    config['train']['model_saved_path'] = str(output/'runner')
    config.setdefault('evaluation', {}).update({
        'dump_predictions': False,
        'selector_candidates': [cpr_candidate(config)],
    })
    path = output/'expanded_config.json'
    write(path, config)
    return path


# Paper metrics per dataset.  The official QVHighlights evaluator already
# reports percentages; the moment-retrieval metrics of the other datasets
# are fractions and are scaled by 100 for reporting.
METRIC_SCALE = {ds: 100.0 for ds in
                ('charades', 'activitynet', 'tacos', 'ego4d')}
METRIC_SCALE['qvhighlights'] = 1.0

PAPER_METRICS = {
    'charades': ('R@1,IoU@0.5', 'R@1,IoU@0.7', 'R@1,mIoU'),
    'activitynet': ('R@1,IoU@0.5', 'R@1,IoU@0.7', 'R@1,mIoU'),
    'tacos': ('R@1,IoU@0.1', 'R@1,IoU@0.3', 'R@1,IoU@0.5', 'R@1,mIoU'),
    'ego4d': ('R@1,IoU@0.1', 'R@1,IoU@0.3', 'R@1,IoU@0.5', 'R@1,mIoU'),
    'qvhighlights': ('MR-mAP@0.5', 'MR-mAP@0.75', 'MR-full-mAP',
                     'MR-full-R1@0.5', 'MR-full-R1@0.7',
                     'HD-VeryGood-HL-mAP', 'HD-VeryGood-HL-Hit1'),
}


def report(dataset, summary_path):
    summary = read(summary_path)
    metrics = summary['best']['metrics']
    keys = PAPER_METRICS[dataset]
    scale = METRIC_SCALE[dataset]
    values = [scale*float(metrics[key]) for key in keys]
    print('\nPC-Net++ (%s, split=%s):  ' % (dataset, summary.get('split', '?'))
          + ' / '.join('%.2f' % value for value in values))
    print('  full metrics and per-query details: %s' % summary_path)
    return dict(zip(keys, values))


def run_eval(dataset, split, checkpoint, output, gpu='0',
             limit_eval_batches=None):
    """Run CPR inference of ``checkpoint``; returns the paper metrics."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit('model not found: %s' % checkpoint)
    config = build_config(dataset, output, SPLIT_DATA[(dataset, split)])
    command = [sys.executable, str(SRC/'train.py'),
               '--config-path', str(config), '--protocol', 'unified_paper',
               '--deterministic-eval-mask', '--output-root',
               str(output/'runner'), '--tag', f'pcnetpp_{dataset}_{split}',
               '--init-from', str(checkpoint), '--eval',
               '--skip-prediction-dump', '--top1-selector-output',
               str(output/'selector')]
    if limit_eval_batches is not None:
        command.extend(['--limit-eval-batches', str(limit_eval_batches)])
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONPATH': str(SRC),
                'PYTHONUNBUFFERED': '1',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    write(output/'command.json', {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'dataset': dataset, 'split': split,
        'checkpoint': str(checkpoint), 'command': command,
    })
    subprocess.run(command, cwd=SRC, env=env, check=True)
    return report(dataset, output/'selector'/'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--split', default=None,
                        help='charades/activitynet: tt|nc|nw; the other '
                             'datasets use their protocol split')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='model-best.pt of a run.py training run '
                             '(default: latest run under outputs/<dataset>/)')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--limit-eval-batches', type=int, default=None)
    args = parser.parse_args()
    split = args.split or DEFAULT_SPLIT[args.dataset]
    if (args.dataset, split) not in SPLIT_DATA:
        raise SystemExit('unknown split %r for %s' % (split, args.dataset))
    output = (args.output.resolve() if args.output else
              ROOT/'outputs'/f'{args.dataset}_{split}')
    checkpoint = args.checkpoint
    if checkpoint is None:
        runs = sorted((ROOT/'outputs'/args.dataset/'runner').glob(
            '*/model-best.pt'), key=lambda path: path.stat().st_mtime)
        if not runs:
            raise SystemExit('no trained model under outputs/%s/runner; '
                             'run run.py first' % args.dataset)
        checkpoint = runs[-1]
    run_eval(args.dataset, split, checkpoint, output, args.gpu,
             args.limit_eval_batches)


if __name__ == '__main__':
    main()
