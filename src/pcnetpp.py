"""PC-Net++ inference: counterfactual-guided proposal refinement (CPR).

This module implements the decision stage of PC-Net++ on top of a frozen
PC-Net retriever.  A single shared forward pass produces, for every query,
the factual reconstruction logits, the counterfactual deletion-view logits,
and the proposal geometry (center / width / temporal response).

Single-moment datasets (Charades-CG, ActivityNet-CG, TACoS, Ego4D-NLQ) use
the ``cpr`` rule:

  * CF re-scoring -- each proposal is scored by
    ``score = inside - lambda_CF * deleted``, where ``inside`` is the query
    reconstruction NLL when the proposal's visual evidence is retained and
    ``deleted`` the NLL after the evidence is removed; smaller is better.
  * CBC boundary consensus -- the top-1 anchor's endpoints are moved toward
    the weighted average endpoint of the compatible (IoU >= rho_B) windows
    among the top K_B, with softmax weights over the relative CF scores at
    temperature tau_B, using convex coefficient alpha_B.

QVHighlights uses the ``cf_marginal_set`` rule for multi-moment retrieval:
the moment set is ordered greedily by marginal counterfactual set-deletion
gains, the rank-1 window keeps the factual anchor, and highlight detection
uses the causal-gain weighted envelope of the temporal responses.

Every quantity is derived from the same frozen forward pass; no annotation
information is used at any point.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.loss import cal_nll_loss
from runners.main_runner import move_to_device

THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _components(output, num_props):
    deletion = output.get('counterfactual_deletion_words_logit')
    if deletion is None:
        raise KeyError('CPR requires counterfactual deletion reconstruction')
    bsz = output['words_id'].size(0)
    mask = output['words_mask'].unsqueeze(1).expand(-1, num_props, -1)
    ids = output['words_id'].unsqueeze(1).expand(-1, num_props, -1)
    inside, _ = cal_nll_loss(
        output['words_logit'], ids.reshape(bsz*num_props, -1),
        mask.reshape(bsz*num_props, -1))
    deleted, _ = cal_nll_loss(
        deletion, ids.reshape(bsz*num_props, -1),
        mask.reshape(bsz*num_props, -1))
    return inside.view(bsz, num_props), deleted.view(bsz, num_props)


def _robust_z(value):
    """Query-local robust normalisation used by transferable CF scoring."""
    # torch.median(..., dim=) has no deterministic CUDA kernel. Sorting eight
    # proposals is cheap and remains bitwise reproducible under the protocol.
    middle = (value.size(-1)-1)//2
    median = value.sort(dim=-1).values[..., middle:middle+1]
    deviation = (value-median).abs()
    mad = deviation.sort(dim=-1).values[..., middle:middle+1]
    # 1.4826 makes MAD comparable to standard deviation for Gaussian data.
    return (value-median)/(1.4826*mad).clamp_min(1e-4)


def _proposal_score(inside, deleted, candidate):
    """CF ranking score: lower means stronger retained-vs-deleted evidence."""
    return inside-float(candidate.get('lambda', 0.0))*deleted


def _interval_iou(first, second):
    left = torch.maximum(first[..., 0], second[..., 0])
    right = torch.minimum(first[..., 1], second[..., 1])
    intersection = (right-left).clamp_min(0)
    union = (torch.maximum(first[..., 1], second[..., 1])-
             torch.minimum(first[..., 0], second[..., 0]))
    return intersection/union.clamp_min(1e-6)


def _weighted_median(value, weight):
    ordered, index = value.sort(dim=1)
    ordered_weight = weight.gather(1, index)
    # cumsum is not available in deterministic CUDA mode on the archived
    # PyTorch build. K <= 8, so an explicit scan has negligible cost.
    cumulative_parts = [ordered_weight[:, 0]]
    for position in range(1, ordered_weight.size(1)):
        cumulative_parts.append(
            cumulative_parts[-1]+ordered_weight[:, position])
    cumulative = torch.stack(cumulative_parts, dim=1)
    cutoff = .5*ordered_weight.sum(dim=1, keepdim=True)
    position = (cumulative >= cutoff).to(torch.int64).argmax(dim=1)
    return ordered.gather(1, position.unsqueeze(1)).squeeze(1)


def _cf_marginal_windows(runner, output, inside, deleted, candidate,
                         factual_anchor):
    """Greedy multi-moment ordering by marginal counterfactual set deletion.

    At every step the frozen reconstructor scores the complement of the
    already-selected evidence with and without one extra candidate, so a
    proposal whose moment is already covered adds little reconstruction loss
    and sinks in the ranking while distinct relevant moments surface first.
    The rank-1 window stays byte-for-byte identical to the factual anchor;
    optional endpoint consensus fuses each selected moment with its
    CF-weighted overlap cluster, so counterfactual necessity decides both
    selection and fusion.
    """
    power = float(candidate.get('deletion_power', 1.0))
    bsz, num_props = inside.shape
    curves = output['gauss_weight'].view(bsz, num_props, -1)
    weights = curves/curves.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
    center = output['center'].view(bsz, num_props)
    width = output['width'].view(bsz, num_props)
    raw = torch.stack(((center-width/2).clamp(0, 1),
                       (center+width/2).clamp(0, 1)), dim=-1)
    if factual_anchor is None:
        factual_anchor = raw.gather(
            1, inside.argmin(dim=-1).view(-1, 1, 1).expand(-1, 1, 2)
        ).squeeze(1)
    handle = output.get('recon_handle')
    if handle is None:
        raise KeyError('cf_marginal_set requires expose_reconstruction')
    words_id = output['words_id']

    def deletion_mask(union):
        mask = (1.0-union).clamp_min(0).pow(power)
        return mask/mask.max(dim=-1, keepdim=True).values.clamp_min(1e-6)

    first = _interval_iou(raw, factual_anchor.unsqueeze(1)).squeeze(
        1).argmax(dim=1)
    selected = torch.zeros(bsz, num_props, dtype=torch.bool,
                           device=raw.device)
    selected.scatter_(1, first.unsqueeze(1), True)
    union = weights.gather(
        1, first.view(-1, 1, 1).expand(-1, 1, weights.size(-1))).squeeze(1)
    order = [first]
    for _ in range(num_props-1):
        cand_union = torch.maximum(union.unsqueeze(1), weights)
        views = torch.cat(
            [deletion_mask(union).unsqueeze(1), deletion_mask(cand_union)],
            dim=1)
        nll = runner.model.counterfactual_view_nll(handle, views, words_id)
        marginal = nll[:, 1:]-nll[:, :1]
        marginal = marginal.masked_fill(selected, float('-inf'))
        best = marginal.argmax(dim=1)
        selected.scatter_(1, best.unsqueeze(1), True)
        union = torch.maximum(union, weights.gather(
            1, best.view(-1, 1, 1).expand(-1, 1, weights.size(-1))).squeeze(1))
        order.append(best)
    set_order = torch.stack(order, dim=1)
    ranked = raw.gather(1, set_order.unsqueeze(-1).expand(-1, -1, 2))
    ranked[:, 0] = factual_anchor

    alpha = float(candidate.get('fuse_alpha', 0.0))
    accepted = torch.zeros(bsz, dtype=torch.bool, device=raw.device)
    if alpha > 0.0:
        rho = float(candidate.get('fuse_min_iou', 0.5))
        tau = float(candidate.get('fuse_temperature', 0.5))
        max_move = float(candidate.get('fuse_max_move', 0.35))
        cf = deleted-inside
        for position in range(1, num_props):
            anchor = ranked[:, position]
            overlap = _interval_iou(raw, anchor.unsqueeze(1)).squeeze(1)
            valid = overlap >= rho
            weight = F.softmax(cf/tau, dim=1)*valid.to(cf.dtype)
            weight = weight/weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
            target = torch.stack((
                _weighted_median(raw[..., 0], weight),
                _weighted_median(raw[..., 1], weight)), dim=-1)
            proposed = ((1-alpha)*anchor+alpha*target).clamp(0, 1)
            span = (anchor[:, 1]-anchor[:, 0]).clamp_min(1e-4)
            move = (proposed-anchor).abs().max(dim=-1).values/span
            keep = (move <= max_move) & (valid.sum(dim=1) >= 2)
            ranked[:, position] = torch.where(
                keep.unsqueeze(-1), proposed, anchor)
            accepted |= keep

    # Optional anchor refinement: move the rank-1 (factual) endpoints toward a
    # CF-weighted consensus of its compatible cluster.  Necessity weights
    # decide which proposals contribute endpoints; a move-ratio gate falls
    # back to the byte-for-byte anchor whenever the cluster is uncertain.
    top1_alpha = float(candidate.get('top1_alpha', 0.0))
    top1_accepted = torch.zeros(bsz, dtype=torch.bool, device=raw.device)
    if top1_alpha > 0.0:
        rho_t = float(candidate.get('top1_min_iou', 0.3))
        tau_t = float(candidate.get('top1_temperature', 0.25))
        max_move_t = float(candidate.get('top1_max_move', 0.2))
        anchor = ranked[:, 0]
        overlap = _interval_iou(raw, anchor.unsqueeze(1)).squeeze(1)
        valid = overlap >= rho_t
        weight = F.softmax(_robust_z(deleted-inside)/tau_t, dim=1)
        weight = weight*valid.to(weight.dtype)
        weight = weight/weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target = torch.stack((
            _weighted_median(raw[..., 0], weight),
            _weighted_median(raw[..., 1], weight)), dim=-1)
        proposed = ((1-top1_alpha)*anchor+top1_alpha*target).clamp(0, 1)
        span = (anchor[:, 1]-anchor[:, 0]).clamp_min(1e-4)
        move = (proposed-anchor).abs().max(dim=-1).values/span
        keep_t = (move <= max_move_t) & (valid.sum(dim=1) >= 2)
        ranked[:, 0] = torch.where(keep_t.unsqueeze(-1), proposed, anchor)
        ranked[:, 0, 1] = torch.maximum(
            ranked[:, 0, 1], ranked[:, 0, 0]+1e-5).clamp(max=1)
        top1_accepted = keep_t
    diagnostics = {'accepted': accepted,
                   'top1_accepted': top1_accepted.to(inside.dtype)}
    return ranked, diagnostics


def _windows(output, score, candidate, runner=None, inside=None, deleted=None):
    bsz, num_props = score.shape
    center = output['center'].view(bsz, num_props)
    width = output['width'].view(bsz, num_props)
    raw = torch.stack(((center-width/2).clamp(0, 1),
                       (center+width/2).clamp(0, 1)), dim=-1)
    factual_anchor = raw.gather(
        1, score.argmin(dim=-1).view(-1, 1, 1).expand(-1, 1, 2)
    ).squeeze(1)
    if candidate['family'] == 'cf_marginal_set':
        return _cf_marginal_windows(
            runner, output, inside, deleted, candidate, factual_anchor)
    order = score.argsort(dim=-1)
    ranked = raw.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
    ranked_score = score.gather(1, order)
    if candidate['family'] == 'cpr':
        # CBC: compatible boundary consensus over the CF-ranked top-K.
        k = min(int(candidate.get('boundary_topk', 2)), num_props)
        top = ranked[:, :k]
        anchor = top[:, :1]
        left = torch.maximum(anchor[..., 0], top[..., 0])
        right = torch.minimum(anchor[..., 1], top[..., 1])
        inter = (right-left).clamp_min(0)
        union = torch.maximum(anchor[..., 1], top[..., 1])-torch.minimum(
            anchor[..., 0], top[..., 0])
        overlap = inter/union.clamp_min(1e-6)
        valid = overlap >= float(candidate.get('boundary_min_iou', 0.1))
        score_weight = F.softmax(
            -(ranked_score[:, :k]-ranked_score[:, :1]) /
            float(candidate.get('boundary_temperature', 0.5)), dim=-1)
        endpoint_weight = score_weight.unsqueeze(-1).expand(-1, -1, 2)
        endpoint_weight = endpoint_weight*valid.unsqueeze(-1)
        endpoint_weight = endpoint_weight/endpoint_weight.sum(
            dim=1, keepdim=True).clamp_min(1e-6)
        if candidate.get('boundary_estimator', 'mean') == 'median':
            target = torch.stack((
                _weighted_median(top[..., 0], endpoint_weight[..., 0]),
                _weighted_median(top[..., 1], endpoint_weight[..., 1])), -1)
        else:
            target = (top*endpoint_weight).sum(dim=1)
        alpha = float(candidate['boundary_alpha'])
        proposed = ((1-alpha)*ranked[:, 0]+alpha*target).clamp(0, 1)
        ranked[:, 0] = proposed
        ranked[:, 0, 1] = torch.maximum(
            ranked[:, 0, 1], ranked[:, 0, 0]+1e-5).clamp(max=1)
        diagnostics = {'accepted': torch.ones(
            bsz, dtype=torch.bool, device=raw.device)}
        return ranked, diagnostics
    raise ValueError('unknown CPR family: %r' % candidate['family'])


def _empty():
    return {'n': 0, 'top1_iou_sum': 0.0, 'top5_iou_sum': 0.0,
            'top1_hits': {str(x): 0 for x in THRESHOLDS},
            'top5_hits': {str(x): 0 for x in THRESHOLDS}}


def _update(acc, windows, raw_batch):
    windows = windows.detach().cpu().numpy()
    gt = np.asarray([
        np.asarray(raw[2], dtype=np.float64)/max(float(raw[1]), 1e-10)
        for raw in raw_batch], dtype=np.float64)
    left = np.maximum(windows[..., 0], gt[:, None, 0])
    right = np.minimum(windows[..., 1], gt[:, None, 1])
    intersection = np.maximum(right-left, 0.0)
    union = np.maximum(windows[..., 1], gt[:, None, 1])-np.minimum(
        windows[..., 0], gt[:, None, 0])
    iou = intersection/np.maximum(union, 1e-10)
    top1 = iou[:, 0]
    top5 = iou[:, :min(5, iou.shape[1])].max(axis=1)
    acc['n'] += len(raw_batch)
    acc['top1_iou_sum'] += float(top1.sum())
    acc['top5_iou_sum'] += float(top5.sum())
    for threshold in THRESHOLDS:
        acc['top1_hits'][str(threshold)] += int((top1 >= threshold).sum())
        acc['top5_hits'][str(threshold)] += int((top5 >= threshold).sum())


def _top1_ious(windows, raw_batch):
    """Return per-query IoUs for post-hoc paired uncertainty estimates."""
    windows = windows.detach().cpu().numpy()
    gt = np.asarray([
        np.asarray(raw[2], dtype=np.float64)/max(float(raw[1]), 1e-10)
        for raw in raw_batch], dtype=np.float64)
    left = np.maximum(windows[:, 0, 0], gt[:, 0])
    right = np.minimum(windows[:, 0, 1], gt[:, 1])
    intersection = np.maximum(right-left, 0.0)
    union = np.maximum(windows[:, 0, 1], gt[:, 1])-np.minimum(
        windows[:, 0, 0], gt[:, 0])
    return intersection/np.maximum(union, 1e-10)


def _metrics(acc):
    n = max(acc['n'], 1)
    result = {'R@1,mIoU': acc['top1_iou_sum']/n,
              'R@5,mIoU': acc['top5_iou_sum']/n}
    for threshold in THRESHOLDS:
        label = 'IoU@%.1f' % threshold
        result['R@1,'+label] = acc['top1_hits'][str(threshold)]/n
        result['R@5,'+label] = acc['top5_hits'][str(threshold)]/n
    return result


def _candidate_saliency(output, inside, deleted, candidate):
    """Build a GT-free temporal evidence field for QVHighlights HD."""
    curves = output['gauss_weight'].view(inside.size(0), inside.size(1), -1)
    mode = candidate.get('saliency_mode', 'max')
    if mode == 'max':
        return curves.max(dim=1).values
    if mode == 'causal_gain':
        temperature = float(candidate.get('saliency_temperature', 1.0))
        if temperature <= 0:
            raise ValueError('saliency_temperature must be positive')
        gain = _robust_z(deleted-inside)
        weight = F.softmax(gain/temperature, dim=-1)
        return torch.einsum('bp,bpt->bt', weight, curves)
    if mode == 'causal_positive':
        gain = _robust_z(deleted-inside).clamp_min(0)
        weight = gain/gain.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum('bp,bpt->bt', weight, curves)
    raise ValueError('unknown saliency_mode: %s' % mode)


def _append_qv_records(records, windows, saliency, raw_batch):
    windows = windows.detach().cpu().numpy()
    saliency = saliency.detach().cpu().numpy()
    for index, raw in enumerate(raw_batch):
        duration = float(raw[1])
        count = windows.shape[1]
        records.append({
            'qid': int(raw[4]), 'vid': raw[0], 'duration': duration,
            'query': raw[3],
            'pred_windows': [
                [float(window[0]*duration), float(window[1]*duration),
                 float(count-rank)]
                for rank, window in enumerate(windows[index])],
            'highlight_scores': saliency[index].tolist(),
        })


def evaluate_cpr(runner, output_dir, loader=None, split=None):
    """Run PC-Net++ CPR inference on a split and write summary.json.

    ``runner`` only supplies the frozen forward, the data loader, and the CPR
    configuration; the deletion view is enabled here so that training itself
    is untouched.  Used both for per-epoch model selection during training and
    for the final reported evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if split is None:
        split = runner.args.get('protocol', {}).get('selection_split', 'test')
    if loader is None:
        loader = runner.val_loader if split == 'val' else runner.test_loader
    candidates = runner.args.get('evaluation', {}).get('selector_candidates')
    if not candidates:
        raise ValueError('CPR evaluation requires an explicit candidate')
    saved_flags = (runner.model.use_counterfactual_utility,
                   runner.model.counterfactual_deletion_power)
    runner.model.use_counterfactual_utility = True
    runner.model.counterfactual_deletion_power = float(
        candidates[0].get('deletion_power', 1.0))
    try:
        return _evaluate_cpr_loop(runner, output_dir, loader, split,
                                  candidates)
    finally:
        (runner.model.use_counterfactual_utility,
         runner.model.counterfactual_deletion_power) = saved_flags


def _evaluate_cpr_loop(runner, output_dir, loader, split, candidates):
    accumulators = {candidate['id']: _empty() for candidate in candidates}
    selector_diagnostics = {
        candidate['id']: {'n': 0, 'accepted': 0, 'signals': {}}
        for candidate in candidates}
    dump_records = bool(runner.args.get('evaluation', {}).get(
        'dump_selector_records', False))
    paired_records = []
    qv_enabled = (runner.args.get('evaluation', {}).get('evaluator') ==
                  'qvhighlights_official')
    qv_records = ({candidate['id']: [] for candidate in candidates}
                  if qv_enabled else None)
    runner.model.eval()
    with torch.no_grad():
        limit = runner.args['train'].get('limit_eval_batches')
        for batch_index, batch in enumerate(loader, 1):
            if limit is not None and batch_index > limit:
                break
            net_input = move_to_device(batch['net_input'], runner.device)
            output = runner.model(
                epoch=0, positive_only=True,
                expose_reconstruction=any(
                    item.get('family') == 'cf_marginal_set'
                    for item in candidates), **net_input)
            inside, deleted = _components(output, runner.model.num_props)
            for candidate in candidates:
                score = _proposal_score(inside, deleted, candidate)
                windows, diagnostic = _windows(
                    output, score, candidate,
                    runner=runner, inside=inside, deleted=deleted)
                _update(accumulators[candidate['id']], windows, batch['raw'])
                current = selector_diagnostics[candidate['id']]
                current['n'] += len(batch['raw'])
                current['accepted'] += int(diagnostic['accepted'].sum().item())
                for name, value in diagnostic.items():
                    if name == 'accepted':
                        continue
                    current['signals'][name] = (
                        current['signals'].get(name, 0.0) +
                        float(value.sum().item()))
                if qv_enabled:
                    saliency = _candidate_saliency(
                        output, inside, deleted, candidate)
                    _append_qv_records(
                        qv_records[candidate['id']], windows, saliency,
                        batch['raw'])
                if dump_records:
                    candidate_ious = _top1_ious(windows, batch['raw'])
                    for index, raw in enumerate(batch['raw']):
                        paired_records.append({
                            'candidate': candidate['id'], 'vid': raw[0],
                            'query': raw[3],
                            'candidate_iou': float(candidate_ious[index]),
                        })
    metric_name = runner.args.get('protocol', {}).get(
        'selection_metric', 'R@1,mIoU')
    qv_metrics = {}
    if qv_enabled:
        from evaluation.qv_convert import official_qv_metrics
        evaluation = runner.args['evaluation']
        for candidate in candidates:
            qv_metrics[candidate['id']] = official_qv_metrics(
                qv_records[candidate['id']], evaluation['qv_gt_jsonl'],
                evaluator_root=evaluation.get('qv_evaluator_root'),
                match_number=runner.args['train'].get(
                    'limit_eval_batches') is None)
    rows = []
    for candidate in candidates:
        metrics = _metrics(accumulators[candidate['id']])
        metrics.update(qv_metrics.get(candidate['id'], {}))
        diagnostic = selector_diagnostics[candidate['id']]
        n = max(diagnostic['n'], 1)
        rows.append({
            'id': candidate['id'], 'candidate': candidate,
            'metrics': metrics, 'paper_score': metrics[metric_name],
            'selector_diagnostics': {
                'coverage': diagnostic['accepted']/n,
                'accepted': diagnostic['accepted'], 'n': diagnostic['n'],
                'mean_signals': {
                    name: value/n for name, value in diagnostic['signals'].items()
                },
            },
        })
    rows.sort(key=lambda row: row['paper_score'], reverse=True)
    summary = {'status': 'complete', 'split': split, 'training': False,
               'parameter_updates': 0, 'shared_forward': True,
               'max_views': 1, 'single_view_protocol': 'deterministic_single_forward',
               'num_candidates': len(rows),
               'best': rows[0], 'ranking': rows}
    (output_dir/'summary.json').write_text(
        json.dumps(summary, indent=2)+'\n', encoding='utf8')
    if dump_records:
        with (output_dir/'paired_records.jsonl').open('w', encoding='utf8') as handle:
            for record in paired_records:
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
    print('CPR summary written to %s (score %.6f)' % (
        output_dir/'summary.json', rows[0]['paper_score']))
    return summary
