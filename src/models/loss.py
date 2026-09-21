import torch
import torch.nn.functional as F
import math


def cal_nll_loss(logit, idx, mask, weights=None, target_is_feat=False,
                 query_target=None, query_weight=0.0, temperature=1.0):
    eps = 0.1
    if target_is_feat:
        if temperature <= 0:
            raise ValueError('continuous reconstruction temperature must be positive')
        if logit.shape != idx.shape:
            raise ValueError('continuous reconstruction shape mismatch: %s != %s' %
                             (tuple(logit.shape), tuple(idx.shape)))
        error = 1.0-F.cosine_similarity(logit, idx, dim=-1)
        if weights is None:
            error = error.masked_fill(mask == 0, 0)
            error = error.sum(dim=-1)/mask.sum(dim=-1).clamp_min(1)
        else:
            effective = weights * mask.to(weights.dtype)
            effective = effective/effective.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            error = (error*effective).sum(dim=-1)
        error = error / float(temperature)
        if query_weight:
            effective = mask.to(logit.dtype)
            if weights is not None:
                effective = effective*weights.to(logit.dtype)
            effective = effective/effective.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            predicted_query = (logit*effective.unsqueeze(-1)).sum(dim=-2)
            if query_target is None:
                target_query = (idx*effective.unsqueeze(-1)).sum(dim=-2)
            else:
                if query_target.shape != predicted_query.shape:
                    raise ValueError('query target shape mismatch: %s != %s' %
                                     (tuple(query_target.shape),
                                      tuple(predicted_query.shape)))
                target_query = query_target
            error = error + float(query_weight) * (
                1.0-F.cosine_similarity(predicted_query, target_query, dim=-1))
        return error.contiguous(), logit.new_zeros(())
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()

    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def proposal_reconstruction_targets(words_id, words_clip, bsz, num_props):
    """Expand either discrete token ids or continuous CLIP token targets."""
    if words_clip is not None:
        target = words_clip.unsqueeze(1).expand(
            bsz, num_props, -1, -1).contiguous().view(
                bsz*num_props, words_clip.size(1), words_clip.size(2))
        return target, True
    target = words_id.unsqueeze(1).expand(
        bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    return target, False


def aggregate_proposal_nll(nll, mode='hard_min', temperature=0.5, topk=2):
    """Aggregate proposal reconstruction losses without changing legacy hard-min.

    The smooth variants use a *normalised* soft minimum, so their scale remains
    comparable across different numbers of proposals. Returned weights are also
    used to align the negative-proposal term with the positive aggregation.
    """
    if nll.dim() != 2:
        raise ValueError('proposal NLL must have shape [batch, num_props]')
    if mode == 'hard_min':
        value, index = nll.min(dim=-1)
        weights = F.one_hot(index, num_classes=nll.size(1)).to(nll.dtype)
        return value, index, weights
    if temperature <= 0:
        raise ValueError('proposal_temperature must be positive')

    if mode == 'softmin':
        selected = nll
        selected_index = None
    elif mode == 'soft_topk':
        k = min(max(int(topk), 1), nll.size(1))
        selected, selected_index = nll.topk(k, dim=-1, largest=False, sorted=False)
    else:
        raise ValueError('unknown proposal aggregation: %s' % mode)

    # -tau log mean(exp(-loss/tau)): smooth min with proposal-count correction.
    value = -temperature * (
        torch.logsumexp(-selected / temperature, dim=-1) - math.log(selected.size(1)))
    selected_weights = F.softmax(-selected / temperature, dim=-1)
    if selected_index is None:
        weights = selected_weights
    else:
        weights = torch.zeros_like(nll).scatter(1, selected_index, selected_weights)
    index = nll.min(dim=-1)[1]
    return value, index, weights


def rec_loss(words_logit, words_id, words_mask, num_props, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_clip = kwargs.get('words_clip')
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1, target_is_feat = proposal_reconstruction_targets(
        words_id, words_clip, bsz, num_props)
    token_weights = kwargs.get('weights')
    token_weights1 = None if token_weights is None else token_weights.unsqueeze(1).expand(
        bsz, num_props, -1).reshape(bsz*num_props, -1)
    query_target = kwargs.get('words_clip_pool')
    query_target1 = None if query_target is None else query_target.unsqueeze(1).expand(
        bsz, num_props, -1).reshape(bsz*num_props, -1)
    clip_cfg = kwargs.get('clip_reconstruction_config') or {}

    nll_loss, acc = cal_nll_loss(
        words_logit, words_id1, words_mask1,
        weights=token_weights1 if target_is_feat else None,
        target_is_feat=target_is_feat, query_target=query_target1,
        query_weight=clip_cfg.get('query_weight', 0.0),
        temperature=clip_cfg.get('temperature', 1.0))
    nll_loss = nll_loss.view(bsz, num_props)
    proposal_nll_loss, _, _ = aggregate_proposal_nll(
        nll_loss,
        mode=kwargs.get('proposal_aggregation', 'hard_min'),
        temperature=kwargs.get('proposal_temperature', 0.5),
        topk=kwargs.get('proposal_topk', 2))

    final_loss = proposal_nll_loss.mean()

    if ref_words_logit is not None:
        ref_target = words_clip if target_is_feat else words_id
        ref_nll_loss, ref_acc = cal_nll_loss(
            ref_words_logit, ref_target, words_mask,
            weights=token_weights if target_is_feat else None,
            target_is_feat=target_is_feat, query_target=query_target,
            query_weight=clip_cfg.get('query_weight', 0.0),
            temperature=clip_cfg.get('temperature', 1.0))
        final_loss = final_loss + ref_nll_loss.mean()
        final_loss = final_loss / 2
    
    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': proposal_nll_loss.mean().item(),
    }
    if ref_words_logit is not None:
        loss_dict.update({
            'ref_nll_loss': ref_nll_loss.mean().item(),
            })

    return final_loss, loss_dict
    
def ivc_loss(words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_clip = kwargs.get('words_clip')
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1, target_is_feat = proposal_reconstruction_targets(
        words_id, words_clip, bsz, num_props)
    token_weights = kwargs.get('weights')
    token_weights1 = None if token_weights is None else token_weights.unsqueeze(1).expand(
        bsz, num_props, -1).reshape(bsz*num_props, -1)
    query_target = kwargs.get('words_clip_pool')
    query_target1 = None if query_target is None else query_target.unsqueeze(1).expand(
        bsz, num_props, -1).reshape(bsz*num_props, -1)
    clip_cfg = kwargs.get('clip_reconstruction_config') or {}
    clip_args = dict(
        weights=token_weights1 if target_is_feat else None,
        target_is_feat=target_is_feat, query_target=query_target1,
        query_weight=clip_cfg.get('query_weight', 0.0),
        temperature=clip_cfg.get('temperature', 1.0))

    nll_loss, acc = cal_nll_loss(
        words_logit, words_id1, words_mask1,
        **clip_args)
    prop_nll = nll_loss.view(bsz, num_props)
    proposal_nll_loss, idx, proposal_weights = aggregate_proposal_nll(
        prop_nll,
        mode=kwargs.get('proposal_aggregation', 'hard_min'),
        temperature=kwargs.get('proposal_temperature', 0.5),
        topk=kwargs.get('proposal_topk', 2))

    intra_loss = 0.0

    if ref_words_logit is not None:
        # Comparative learning of the entire video as a reference proposal
        ref_target = words_clip if target_is_feat else words_id
        ref_nll_loss, ref_acc = cal_nll_loss(
            ref_words_logit, ref_target, words_mask,
            weights=token_weights if target_is_feat else None,
            target_is_feat=target_is_feat, query_target=query_target,
            query_weight=clip_cfg.get('query_weight', 0.0),
            temperature=clip_cfg.get('temperature', 1.0))
        tmp_0 = torch.zeros_like(proposal_nll_loss)
        tmp_0.requires_grad = False
        ref_loss = torch.max(proposal_nll_loss - ref_nll_loss + kwargs["margin_1"], tmp_0)
        rank_loss = ref_loss.mean()

        # =================================== Quality Margin Regularizer
        strong_mask = prop_nll < (ref_nll_loss.unsqueeze(1))  # Strong proposal conditions
        weak_mask = ~strong_mask
        # Average loss of strong positive group (high query relevance)
        strong_loss = (prop_nll * strong_mask).sum(1) / (strong_mask.sum(1) + 1e-6)
        # Average loss of weak positive group (low query relevance)
        weak_loss = (prop_nll * weak_mask).sum(1) / (weak_mask.sum(1) + 1e-6)
        # Contrast
        intra_loss = torch.max(strong_loss - weak_loss + kwargs["inter_lambda"], tmp_0).mean() 
        # ====================================

    else:
        rank_loss = proposal_nll_loss.mean()
    
    # Simple Negative contrast
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(
            neg_words_logit_1, words_id1, words_mask1,
            **clip_args)
        neg_nll_loss_1 = (neg_nll_loss_1.view(bsz, num_props) * proposal_weights.detach()).sum(dim=-1)
        tmp_0 = torch.zeros_like(proposal_nll_loss)
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(proposal_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    
    # Simple Negative contrast 
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(
            neg_words_logit_2, words_id1, words_mask1,
            **clip_args)
        neg_nll_loss_2 = (neg_nll_loss_2.view(bsz, num_props) * proposal_weights.detach()).sum(dim=-1)
        tmp_0 = torch.zeros_like(proposal_nll_loss)
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(proposal_nll_loss - neg_nll_loss_2 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    loss = kwargs['alpha_1'] * rank_loss + intra_loss*kwargs['co_qua_loss'] 

    gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(num_props, device=gauss_weight.device).unsqueeze(0) * kwargs["lambda"]
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2))**2

    loss = loss + kwargs['alpha_2'] * div_loss.mean()
    return loss, {
        'ivc_loss': loss.item(),
        'neg_loss_1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        'neg_loss_2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        'ref_loss': ref_loss.mean().item() if ref_words_logit is not None else 0.0,
        'intra_loss': intra_loss.mean().item() if ref_words_logit is not None else 0.0,
        'div_loss': div_loss.mean().item()
    }
