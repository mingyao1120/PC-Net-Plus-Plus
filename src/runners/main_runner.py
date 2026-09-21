import collections
import json
import logging
import os

import numpy as np
import torch

from models.loss import ivc_loss, rec_loss
from utils import TimeMeter, AverageMeter


def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        self._build_dataset()
        self._build_token_frequency()
        self._build_token_frequency()

        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['config']['max_epoch'] = self.args['train']['max_num_epochs']

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0
        import shutil,zipfile
        self.model_saved_path = self.args['train']['model_saved_path']
        os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
        shutil.copy('models/pcnet.py', self.model_saved_path)
        shutil.copy('runners/main_runner.py', self.model_saved_path)

        path = 'models' # soruce direction
        zipName = os.path.join(self.model_saved_path , 'models.zip')
        
        f = zipfile.ZipFile( zipName, 'w', zipfile.ZIP_DEFLATED )
        for dirpath, dirnames, filenames in os.walk( path ):
            for filename in filenames:
                f.write(os.path.join(dirpath,filename))
        f.close()

    def train(self):
        best_results = None
        selection_history = []
        best_epoch = None
        non_improving_epochs = 0
        early_patience = int(self.args['train'].get('early_stop_patience', 0))
        early_min_epoch = int(self.args['train'].get('early_stop_min_epoch', 15))
        early_min_delta = float(self.args['train'].get('early_stop_min_delta', 0.0))
        selection_metric = self.args.get('protocol', {}).get(
            'selection_metric', 'R@1,mIoU')
        selection_weights = self.args.get('protocol', {}).get(
            'selection_metric_weights')

        def selection_value(results):
            if not selection_weights:
                if selection_metric not in results:
                    raise KeyError('unknown selection metric: %s' % selection_metric)
                return results[selection_metric].avg
            missing = [key for key in selection_weights if key not in results]
            if missing:
                raise KeyError('unknown composite selection metrics: %s' % missing)
            denominator = sum(float(value) for value in selection_weights.values())
            if denominator <= 0:
                raise ValueError('selection_metric_weights must have positive sum')
            return sum(results[key].avg * float(weight)
                       for key, weight in selection_weights.items()) / denominator
        if self.args['train'].get('evaluate_before_train', False):
            selection_split = self.args.get('protocol', {}).get('selection_split', 'test')
            selection_loader = self.val_loader if selection_split == 'val' else self.test_loader
            best_results = self.eval(
                loader=selection_loader, split=selection_split, dump=False, epoch=0)
            selection_history.append({
                'epoch': 0,
                **{key: meter.avg for key, meter in best_results.items()},
            })
            self._save_model(os.path.join(self.model_saved_path, 'model-best.pt'))
            self._write_selection_artifacts(best_results, selection_history)
            best_epoch = 0
        for epoch in range(1, self.args['train']['max_num_epochs']+1):
            info('Start Epoch {}'.format(epoch))
            self.model_saved_path = self.args['train']['model_saved_path']
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            self._train_one_epoch(epoch)

            selection_split = self.args.get('protocol', {}).get('selection_split', 'test')
            selection_loader = self.val_loader if selection_split == 'val' else self.test_loader
            results = self.eval(loader=selection_loader, split=selection_split, dump=False, epoch=epoch)
            selection_history.append({
                'epoch': epoch,
                **{key: meter.avg for key, meter in results.items()},
            })
            improved = (best_results is None or
                        selection_value(results) >
                        selection_value(best_results) + early_min_delta)
            if improved:
                best_results = results
                best_epoch = epoch
                non_improving_epochs = 0

                best_path = os.path.join(self.model_saved_path, 'model-best.pt')
                self._save_model(best_path)
                self._write_selection_artifacts(best_results, selection_history)

                info('Best results have been updated.')
            else:
                non_improving_epochs += 1
                self._write_selection_artifacts(best_results, selection_history)
            self._save_model(os.path.join(self.model_saved_path, 'model-last.pt'))
            if self.args['train'].get('save_every_epoch', False):
                self._save_model(os.path.join(
                    self.model_saved_path, 'model-epoch-%d.pt' % epoch))
            if (early_patience > 0 and epoch >= early_min_epoch and
                    non_improving_epochs >= early_patience):
                import json
                status = {
                    'stopped_epoch': epoch,
                    'best_epoch': best_epoch,
                    'selection_split': self.args.get('protocol', {}).get(
                        'selection_split', 'test'),
                    'selection_metric': selection_metric,
                    'selection_metric_weights': selection_weights,
                    'patience': early_patience,
                    'min_epoch': early_min_epoch,
                    'non_improving_epochs': non_improving_epochs,
                }
                with open(os.path.join(self.model_saved_path, 'early_stop.json'),
                          'w', encoding='utf8') as handle:
                    json.dump(status, handle, indent=2)
                info('Early stopping at epoch %d (best epoch %s).' %
                     (epoch, str(best_epoch)))
                break
            info('=' * 60)
        
        msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in best_results.items()])
        info('Best results:')
        info('|'+msg+'|')

    def _write_selection_artifacts(self, best_results, history):
        import json
        with open(os.path.join(self.model_saved_path, 'best_selection_metrics.json'),
                  'w', encoding='utf8') as handle:
            json.dump({key: meter.avg for key, meter in best_results.items()}, handle, indent=2)
        with open(os.path.join(self.model_saved_path, 'selection_history.json'),
                  'w', encoding='utf8') as handle:
            json.dump(history, handle, indent=2)

    def _build_token_frequency(self):
        """Map dataset token ids to train-corpus counts for reliability gates."""
        counts = torch.zeros(self.train_set.vocab_size, dtype=torch.float32)
        counter = self.train_set.vocab.get('counter', {})
        for word, token_id in self.train_set.keep_vocab.items():
            counts[int(token_id)] = float(counter.get(word, 0))
        self.token_frequency = counts

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        limit_train_batches = self.args['train'].get('limit_train_batches')
        for bid, batch in enumerate(self.train_loader, 1):
            if limit_train_batches is not None and bid > limit_train_batches:
                break
            self.optimizer.zero_grad()
            net_input = move_to_device(batch['net_input'], self.device)
            output = self.model(epoch=epoch, **net_input)
            cl_loss = output['cl_loss']

            loss_args = dict(self.args['loss'])
            quality_schedule = loss_args.get('co_qua_loss_schedule')
            if quality_schedule:
                progress = min(float(epoch) /
                               max(float(quality_schedule['warmup_epochs']), 1.0), 1.0)
                loss_args['co_qua_loss'] = (
                    float(quality_schedule['start']) + progress *
                    (float(quality_schedule['end']) -
                     float(quality_schedule['start'])))
            loss, loss_dict = rec_loss(
                **output, num_props=self.model.num_props, **loss_args)
            rnk_loss, rnk_loss_dict = ivc_loss(
                **output, num_props=self.model.num_props, **loss_args)
            loss_dict.update(rnk_loss_dict)
            loss = loss + rnk_loss + cl_loss*self.args['loss']['co_cl_loss']
            if quality_schedule:
                loss_dict['co_qua_loss_effective'] = loss_args['co_qua_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

    def eval(self, loader=None, split='test', dump=True, epoch=0):
        """Evaluate PC-Net++ on a split with the complete CPR decision rule.

        Every validation -- including per-epoch model selection during
        training -- scores the model through counterfactual re-scoring and
        compatible boundary consensus, so selected and reported metrics are
        always those of the full method.
        """
        from pcnetpp import evaluate_cpr
        self.model.eval()
        loader = self.test_loader if loader is None else loader
        if loader is None:
            raise ValueError("requested evaluation split has no DataLoader")
        info("evaluation split=%s protocol=%s" %
             (split, self.args.get('protocol', {}).get('mode', 'legacy_exact')))
        output_dir = os.path.join(self.model_saved_path, 'cpr_eval', split)
        summary = evaluate_cpr(self, output_dir, loader=loader, split=split)
        values = dict(summary['best']['metrics'])
        count = max(int(summary['best'].get('selector_diagnostics', {})
                        .get('n', 1)), 1)
        metrics = {}
        for key, value in values.items():
            meter = AverageMeter()
            meter.update(value, count)
            metrics[key] = meter
        info('|' + '|'.join(' %s %.4f ' % (key, value)
                            for key, value in values.items()) + '|')
        if dump:
            with open(os.path.join(self.model_saved_path,
                                   split + '_metrics.json'), 'w',
                      encoding='utf8') as handle:
                json.dump(values, handle, indent=2)
        return metrics

    def _build_dataset(self):
        import datasets as da
        import pickle
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        if self.args.get('protocol', {}).get('strict_train_vocab', False):
            vocab = self._restrict_vocab_to_train(vocab, args['train_data'])
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, split='train')
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split='val') if args['val_data'] else None
        info('train: {} samples, val: {}, test: {} samples'.format(
            len(self.train_set), len(self.val_set) if self.val_set is not None else 0, len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                # DataLoader workers perform lightweight NumPy/HDF5 work.  A
                # Torch thread pool per worker causes severe CPU oversubscription
                # on Ego4D/TACoS when multiple experiments run concurrently.
                torch.set_num_threads(1)

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            set_seed(self.args['seed'] + worker_id)

        num_workers = self.args['train'].get('num_workers', 4)
        test_workers = self.args['train'].get('test_num_workers',
                                              min(num_workers, 4))
        val_workers = self.args['train'].get('val_num_workers',
                                             min(num_workers, 2))
        pin_memory = self.args['train'].get('pin_memory', True)
        persistent = self.args['train'].get('persistent_workers', True)
        prefetch_factor = self.args['train'].get('prefetch_factor', 2)

        def loader_kwargs(workers, seed_workers=False):
            kwargs = {
                'num_workers': workers,
                'pin_memory': pin_memory,
            }
            if workers > 0:
                kwargs['persistent_workers'] = persistent
                kwargs['prefetch_factor'] = prefetch_factor
                if seed_workers:
                    kwargs['worker_init_fn'] = worker_init_fn
            return kwargs

        info('DataLoader workers train/test/val=%d/%d/%d, persistent=%s, '
             'prefetch=%d, query_preencoded=True' % (
                 num_workers, test_workers, val_workers, persistent,
                 prefetch_factor))
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data,
                                       **loader_kwargs(num_workers, True))
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      **loader_kwargs(test_workers))
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     **loader_kwargs(val_workers)) if args['val_data'] else None

    def _build_model(self):
        model_config = self.args['model']
        import models

        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        requested = self.args.get('device', 'cuda')
        if requested == 'cuda' and not torch.cuda.is_available():
            requested = 'cpu'
            info('CUDA unavailable; falling back to CPU')
        self.device = torch.device(requested)
        self.model = self.model.to(self.device)
        print(self.model)

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["optimizer"]
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        import hashlib
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        with open(path + '.sha256', 'w', encoding='ascii') as handle:
            handle.write(digest.hexdigest() + '  ' + os.path.basename(path) + '\n')
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path, resume_updates=True):
        state_dict = torch.load(path, map_location=self.device)
        self.num_updates = state_dict.get('num_updates', 0) if resume_updates else 0
        self.lr_scheduler.step_update(self.num_updates)
        self.model.load_state_dict(state_dict['model_parameters'])
        info('load model from {}, num_updates {}.'.format(
            path, self.num_updates))


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(value) for value in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return move_to_device(sample, device)


def move_to_device(sample, device):
    def _move_to_cuda(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_cuda, sample)
