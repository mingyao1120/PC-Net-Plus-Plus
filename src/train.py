import argparse
import time
import os

from utils import load_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='PC-Net++ training and '
                                     'evaluation entry.')
    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='config file path')
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--resume', type=str, default=None,
                                  help='resume a matching architecture and optimizer step')
    checkpoint_group.add_argument('--init-from', type=str, default=None,
                                  help='initialize (or load for --eval) a model from a checkpoint')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--skip-prediction-dump', action='store_true',
                        help='write eval metrics but omit large per-query prediction dumps')
    parser.add_argument('--tag', default='base', type=str, help='experiment tag')
    parser.add_argument('--output-root', default=None, type=str,
                        help='override config checkpoint root for organized runs')
    parser.add_argument('--seed', default=None, type=int,
                        help='override config seed; absent configs fall back to 8')
    parser.add_argument('--protocol', default=None, choices=['unified_paper'],
                        help='override config protocol; absent configs fall back to unified_paper')
    parser.add_argument('--limit-train-batches', default=None, type=int,
                        help='stop each training epoch after this many batches')
    parser.add_argument('--limit-eval-batches', default=None, type=int,
                        help='stop each eval split after this many batches')
    parser.add_argument('--deterministic-eval-mask', action='store_true',
                        help='use the deterministic content-first word mask during evaluation')
    parser.add_argument('--top1-selector-output', default=None, type=str,
                        help='run the CPR evaluation on the selection split and exit')
    parser.add_argument('--log_dir', default=None, type=str, help='log file save path')
    parser.add_argument('--test-data', default=None, type=str)
    parser.add_argument('--val-data', default=None, type=str)
    parser.add_argument('--slot_num_iteration', default=None, type=int)
    parser.add_argument('--num_props', default=None, type=int)
    parser.add_argument('--co_cl_loss', default=None, type=float)
    parser.add_argument('--co_qua_loss', default=None, type=float)
    parser.add_argument('--co-qua-start', default=None, type=float)
    parser.add_argument('--co-qua-end', default=None, type=float)
    parser.add_argument('--co-qua-warmup-epochs', default=None, type=int)
    parser.add_argument('--inter_lambda', default=None, type=float)
    parser.add_argument('--max-epochs', default=None, type=int)
    parser.add_argument('--early-stop-patience', default=None, type=int)
    parser.add_argument('--early-stop-min-epoch', default=None, type=int)
    parser.add_argument('--early-stop-min-delta', default=None, type=float)
    parser.add_argument('--save-every-epoch', action='store_true')
    parser.add_argument('--num-workers', default=None, type=int)
    parser.add_argument('--val-num-workers', default=None, type=int)
    parser.add_argument('--test-num-workers', default=None, type=int)
    parser.add_argument('--batch-size', default=None, type=int)
    parser.add_argument('--learning-rate', default=None, type=float)
    parser.add_argument('--warmup-updates', default=None, type=int)
    parser.add_argument('--evaluate-before-train', action='store_true', default=None)
    return parser.parse_args(argv)


def main(kargs):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    args = load_json(kargs.config_path)
    if kargs.test_data is not None:
        args['dataset']['test_data'] = kargs.test_data
    if kargs.val_data is not None:
        args['dataset']['val_data'] = kargs.val_data
    seed = kargs.seed if kargs.seed is not None else int(args.get('seed', 8))
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from protocols import apply_protocol
    protocol_name = (kargs.protocol or
                     args.get('protocol', {}).get('mode', 'legacy_exact'))
    args = apply_protocol(args, protocol_name)
    output_root = kargs.output_root or args['train']['model_saved_path']
    run_name = str(seed)+"_"+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if kargs.tag and kargs.tag != 'base':
        run_name += "_" + kargs.tag
    args['train']['model_saved_path'] = os.path.join(output_root, run_name)
    args.setdefault('vote', False)
    if kargs.skip_prediction_dump:
        args.setdefault('evaluation', {})['dump_predictions'] = False
    model_config = args['model']['config']
    if kargs.slot_num_iteration is not None:
        model_config['num_iteration'] = kargs.slot_num_iteration
    else:
        model_config.setdefault('num_iteration', 3)
    if kargs.num_props is not None:
        model_config['num_props'] = kargs.num_props
    else:
        model_config.setdefault('num_props', 8)
    if kargs.max_epochs is not None:
        args['train']['max_num_epochs'] = kargs.max_epochs
    if kargs.early_stop_patience is not None:
        args['train']['early_stop_patience'] = kargs.early_stop_patience
    else:
        args['train'].setdefault('early_stop_patience', 0)
    if kargs.early_stop_min_epoch is not None:
        args['train']['early_stop_min_epoch'] = kargs.early_stop_min_epoch
    else:
        args['train'].setdefault('early_stop_min_epoch', 15)
    if kargs.early_stop_min_delta is not None:
        args['train']['early_stop_min_delta'] = kargs.early_stop_min_delta
    else:
        args['train'].setdefault('early_stop_min_delta', 0.0)
    if (args['train']['early_stop_patience'] < 0 or
            args['train']['early_stop_min_epoch'] < 1):
        raise ValueError('invalid early-stop settings')
    if kargs.save_every_epoch:
        args['train']['save_every_epoch'] = True
    if kargs.num_workers is not None:
        args['train']['num_workers'] = kargs.num_workers
    if kargs.val_num_workers is not None:
        args['train']['val_num_workers'] = kargs.val_num_workers
    if kargs.test_num_workers is not None:
        args['train']['test_num_workers'] = kargs.test_num_workers
    if kargs.batch_size is not None:
        if kargs.batch_size < 1:
            raise ValueError('--batch-size must be positive')
        args['train']['batch_size'] = kargs.batch_size
    if kargs.limit_train_batches is not None:
        if kargs.limit_train_batches < 1:
            raise ValueError('--limit-train-batches must be positive')
        args['train']['limit_train_batches'] = kargs.limit_train_batches
    if kargs.limit_eval_batches is not None:
        if kargs.limit_eval_batches < 1:
            raise ValueError('--limit-eval-batches must be positive')
        args['train']['limit_eval_batches'] = kargs.limit_eval_batches
    if kargs.learning_rate is not None:
        args['train']['optimizer']['lr'] = kargs.learning_rate
    if kargs.warmup_updates is not None:
        args['train']['optimizer']['warmup_updates'] = kargs.warmup_updates
    args['train'].setdefault('early_stop_patience', 0)
    args['train'].setdefault('early_stop_min_epoch', 15)
    args['train'].setdefault('early_stop_min_delta', 0.0)
    if (args['train']['early_stop_patience'] < 0 or
            args['train']['early_stop_min_epoch'] < 1):
        raise ValueError('invalid early-stop settings')
    args['train'].setdefault('evaluate_before_train', False)
    if kargs.co_cl_loss is not None:
        args['loss']['co_cl_loss'] = kargs.co_cl_loss
    else:
        args['loss'].setdefault('co_cl_loss', 1.0)
    if kargs.co_qua_loss is not None:
        args['loss']['co_qua_loss'] = kargs.co_qua_loss
    else:
        args['loss'].setdefault('co_qua_loss', 1.0)
    schedule_values = (kargs.co_qua_start, kargs.co_qua_end,
                       kargs.co_qua_warmup_epochs)
    if any(value is not None for value in schedule_values):
        if not all(value is not None for value in schedule_values):
            raise ValueError('quality schedule requires start, end, and warmup epochs')
        if kargs.co_qua_warmup_epochs < 1:
            raise ValueError('quality schedule warmup must be positive')
        args['loss']['co_qua_loss_schedule'] = {
            'start': kargs.co_qua_start,
            'end': kargs.co_qua_end,
            'warmup_epochs': kargs.co_qua_warmup_epochs,
        }
    if kargs.inter_lambda is not None:
        args['loss']['inter_lambda'] = kargs.inter_lambda
    else:
        args['loss'].setdefault('inter_lambda', 0.13)
    args['loss'].setdefault('proposal_aggregation', 'hard_min')
    args['loss'].setdefault('proposal_temperature', 0.5)
    args['loss'].setdefault('proposal_topk', 2)
    if kargs.evaluate_before_train is not None:
        args['train']['evaluate_before_train'] = kargs.evaluate_before_train
    else:
        args['train'].setdefault('evaluate_before_train', False)
    args['model']['config'].setdefault('train_mask_mode', 'random')
    args['loss'].setdefault('masked_rec_weight', 0.0)
    args['seed'] = seed

    from active_scope import validate_active_scope
    validate_active_scope(args)

    logging.basicConfig(filename=None, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(str(args))

    runner = MainRunner(args)

    if kargs.init_from:
        runner._load_model(kargs.init_from, resume_updates=False)
    if kargs.eval:
        if kargs.top1_selector_output:
            from pcnetpp import evaluate_cpr
            evaluate_cpr(runner, kargs.top1_selector_output)
            return
        runner.eval()
        return
    runner.train()


if __name__ == '__main__':
    main(parse_args())
