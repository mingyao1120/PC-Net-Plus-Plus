"""PC-Net++ model.

Proposal-centric architecture for weakly supervised compositional moment
retrieval, trained with video--query pairs only.  One forward pass couples
proposal learning with the evidence views used by decision-stage refinement:

  * DPG -- dual-granularity proposal generation (global regression fused
    with SlotAttention proposals through a learnable convex weight);
  * PFA -- peak-aware Gaussian aggregation of each proposal's visual
    representation;
  * QMR -- quality-margin regularization of reconstruction quality;
  * counterfactual deletion view -- alongside the factual reconstruction of
    the masked query, the same forward re-decodes the query with a
    proposal's evidence removed (``use_counterfactual_utility``), yielding
    the retained-vs-deleted evidence pair consumed by the CPR decision
    stage in :mod:`pcnetpp`.

Every validation and the final evaluation score the model through the
complete CPR decision rule.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.slot_atten import SlotAttention
from models.transformer import DualTransformer

# ======= Deterministic operating environment (bitwise reproducible runs).
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# ===========


class PCNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config['sigma']
        self.sigma_neg = config['sigma_neg']
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        # Negative proposals start next to the ground-truth-free centre and
        # grow outwards over this horizon (gamma controls the schedule shape).
        self.negative_curriculum_epochs = config.get(
            'negative_curriculum_epochs', self.max_epoch)
        self.gamma = config['gamma']
        self.proposal_match_mode = config.get('proposal_match_mode',
                                              'per_sample')
        if self.proposal_match_mode != 'per_sample':
            raise ValueError("proposal_match_mode must be 'per_sample'")
        self.proposal_branch_mode = 'dual'
        self.semantic_temperature = float(config.get('semantic_temperature', 1.0))
        self.semantic_margin = float(config.get('semantic_margin', 0.5))
        if self.semantic_temperature <= 0:
            raise ValueError('semantic_temperature must be positive')
        self.proposal_generator_mode = config.get('proposal_generator_mode',
                                                  'slot_hungarian')
        if self.proposal_generator_mode != 'slot_hungarian':
            raise ValueError("proposal_generator_mode must be 'slot_hungarian'")
        # QVHighlights contains many short moments inside long clips; a
        # monotone power parameterization keeps widths in (0, 1].  The value
        # 1.0 is an exact identity.
        self.proposal_width_power = float(config.get('proposal_width_power', 1.0))
        if self.proposal_width_power <= 0:
            raise ValueError('proposal_width_power must be positive')
        self.skip_negative_eval = config.get('skip_negative_eval', False)
        # Counterfactual deletion view: alongside the factual reconstruction,
        # the forward re-decodes the query with a proposal's evidence removed;
        # the CPR decision stage scores proposals against this view.
        self.use_counterfactual_utility = bool(config.get(
            'use_counterfactual_utility', False))
        self.counterfactual_deletion_power = float(config.get(
            'counterfactual_deletion_power', 1.0))
        if self.counterfactual_deletion_power <= 0:
            raise ValueError('counterfactual deletion power must be positive')
        self.deterministic_eval_mask = config.get('deterministic_eval_mask', False)
        self.train_mask_mode = config.get('train_mask_mode', 'random')
        if self.train_mask_mode != 'random':
            raise ValueError("train_mask_mode must be 'random'")
        self.gaussian_mode = config.get('gaussian_mode', 'fixed')
        if self.gaussian_mode != 'fixed':
            raise ValueError("gaussian_mode must be 'fixed'")
        self.proposal_kernel = config.get('proposal_kernel', 'gaussian')
        if self.proposal_kernel != 'gaussian':
            raise ValueError("proposal_kernel must be 'gaussian'")
        # The paper path uses T/4 proposal features.  QVHighlights takes the
        # explicit all-frame branch: arange(T) is auditable and cannot silently
        # omit or duplicate a frame through floating-point linspace.
        self.proposal_feature_downsample = int(
            config.get('proposal_feature_downsample', 4))
        if self.proposal_feature_downsample < 1:
            raise ValueError('proposal_feature_downsample must be positive')
        self.proposal_use_all_frames = bool(config.get(
            'proposal_use_all_frames', False))
        if (self.proposal_use_all_frames and
                self.proposal_feature_downsample != 1):
            raise ValueError(
                'proposal_use_all_frames requires proposal_feature_downsample=1')
        self.proposal_fusion_mode = config.get(
            'proposal_fusion_mode', 'center_width_scalar')
        if self.proposal_fusion_mode not in (
                'center_width_scalar', 'center_width_fixed',
                'center_width_learnable'):
            raise ValueError('unknown proposal_fusion_mode: %s' %
                             self.proposal_fusion_mode)
        fusion_alpha = float(config.get('proposal_fusion_alpha', 0.5))
        if not 0.0 < fusion_alpha < 1.0:
            raise ValueError('proposal_fusion_alpha must be in (0, 1)')
        peak_beta_init = float(config.get('peak_beta_init', 0.2))
        if peak_beta_init <= 0:
            raise ValueError('peak_beta_init must be positive')
        self.peak_beta_parameterization = config.get(
            'peak_beta_parameterization', 'softplus')
        if self.peak_beta_parameterization == 'softplus':
            # Keep the historical state-dict name while constraining the
            # effective paper beta to be positive.
            peak_beta_raw = math.log(math.expm1(peak_beta_init))
        elif self.peak_beta_parameterization == 'direct':
            # Exact paper parameterization; not silently clamped.
            peak_beta_raw = peak_beta_init
        else:
            raise ValueError('unknown peak_beta_parameterization: %s' %
                             self.peak_beta_parameterization)

        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)
        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        self.fc_gauss = nn.Linear(config['hidden_size'], self.num_props*2)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)
        # ========= local proposal generation
        self.slot_atten = SlotAttention(config['num_iteration'], self.num_props,
                                        config['hidden_size'])
        self.fc_sl = nn.Linear(config['hidden_size'], 2)
        # Unified convex global/local fusion:
        #   proposal = alpha * global + (1-alpha) * local.
        # The historical behavior is the learnable scalar initialized at 0.5.
        fusion_logit = math.log(fusion_alpha / (1.0-fusion_alpha))
        self.merge_alpha = nn.Parameter(torch.tensor(fusion_logit))
        self.proposal_fusion_logits = nn.Parameter(torch.tensor([
            math.log(fusion_alpha), math.log(1.0-fusion_alpha)]))
        if self.proposal_fusion_mode == 'center_width_learnable':
            self.proposal_fusion_logits.requires_grad_(False)
        else:
            self.merge_alpha.requires_grad_(False)
            self.proposal_fusion_logits.requires_grad_(False)
        self.range_width = nn.Parameter(torch.tensor(peak_beta_raw))

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len,
                weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        self.trans.set_progressive_drop(None, 1.0)
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.to(words_feat.device)
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # generate Gaussian masks
        enc_out, h = self.trans(frames_feat, frames_mask,
                                words_feat + words_pos, words_mask, decoding=1)
        word_cls = enc_out[:, 0]

        # ===================== proposal boundary generation
        gauss_center, gauss_width = self.proposal_generator(h, bsz)

        # ====================== proposal feature aggregation
        ## ========= semantic alignment
        cl_loss = self.semantic_alignment(frames_feat, word_cls, gauss_center,
                                          gauss_width, self.num_props)
        ## ======== peak-aware Gaussian weighting
        keep_idx = self.proposal_frame_indices(n_frames, frames_feat.device)
        props_len = int(keep_idx.numel())
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        effective_sigma = torch.full((bsz*self.num_props,), float(self.sigma),
                                     device=gauss_center.device)
        gauss_weight = self.peak_aware_gauss_weight(
            props_len, gauss_center, gauss_width, effective_sigma)

        # =================== Masked Query Reconstruction
        words_feat, _ = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)

        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, _ = self.trans(props_feat, props_mask, words_feat1, words_mask1,
                             decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)
        recon_handle = None
        if kwargs.get('expose_reconstruction'):
            # Per-query copy of the reconstruction inputs, consumed by the
            # set-level counterfactual rule on QVHighlights.
            recon_handle = (
                props_feat.view(bsz, self.num_props, props_feat.size(-2),
                                props_feat.size(-1))[:, 0].contiguous(),
                props_mask.view(bsz, self.num_props, -1)[:, 0].contiguous(),
                words_feat, words_mask)
        counterfactual_deletion_words_logit = None
        if self.use_counterfactual_utility:
            deletion_weight = (1.0-pos_weight).clamp_min(0).pow(
                self.counterfactual_deletion_power) * \
                props_mask.to(pos_weight.dtype)
            deletion_weight = deletion_weight / deletion_weight.max(
                dim=-1, keepdim=True).values.clamp_min(1e-6)
            _, deletion_h = self.trans(
                props_feat, props_mask, words_feat1, words_mask1,
                decoding=2, gauss_weight=deletion_weight)
            counterfactual_deletion_words_logit = self.fc_comp(deletion_h)

        if (self.use_negative and not kwargs.get('positive_only', False) and
                not (not self.training and self.skip_negative_eval)):
            # Simple negative proposals for the contrastive ranking terms
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(
                props_len, gauss_center, gauss_width, kwargs['epoch'])
            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1,
                                    words_mask1, decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1,
                                    words_mask1, decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)
            # The entire video is used as a reference proposal for contrastive learning
            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'counterfactual_deletion_words_logit': (
                counterfactual_deletion_words_logit),
            'words_id': words_id,
            'weights': weights,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'cl_loss': cl_loss,
            'recon_handle': recon_handle,
        }

    def counterfactual_view_nll(self, recon_handle, views, words_id):
        """Reconstruction NLL for arbitrary proposal-set deletion views.

        ``views`` holds one normalized attention row per view with shape
        ``(batch, V, props_len)``; each row plays exactly the role of the
        per-proposal deletion weight, so the set-level counterfactual rule can
        score the complement of a whole proposal set.
        """
        from models.loss import cal_nll_loss
        props_feat, props_mask, words_feat, words_mask = recon_handle
        bsz, num_views, props_len = views.shape
        words_feat1 = words_feat.unsqueeze(1).expand(
            bsz, num_views, -1, -1).contiguous().view(
                bsz*num_views, words_feat.size(1), -1)
        words_mask1 = words_mask.unsqueeze(1).expand(
            bsz, num_views, -1).contiguous().view(bsz*num_views, -1)
        props1 = props_feat.unsqueeze(1).expand(
            bsz, num_views, -1, -1).contiguous().view(
                bsz*num_views, props_feat.size(1), -1)
        mask1 = props_mask.unsqueeze(1).expand(
            bsz, num_views, -1).contiguous().view(bsz*num_views, -1)
        _, hidden = self.trans(
            props1, mask1, words_feat1, words_mask1, decoding=2,
            gauss_weight=views.reshape(bsz*num_views, props_len))
        logits = self.fc_comp(hidden)
        ids = words_id.unsqueeze(1).expand(
            bsz, num_views, -1).contiguous().view(bsz*num_views, -1)
        nll, _ = cal_nll_loss(logits, ids, words_mask1)
        return nll.view(bsz, num_views)

    def proposal_frame_indices(self, n_frames, device):
        """Indices consumed by proposal reconstruction on the temporal axis."""
        n_frames = int(n_frames)
        if n_frames < 1:
            raise ValueError('proposal reconstruction requires at least one frame')
        if self.proposal_use_all_frames:
            return torch.arange(n_frames, device=device, dtype=torch.long)
        props_len = max(
            1, n_frames // int(self.proposal_feature_downsample))
        return torch.linspace(
            0, n_frames-1, steps=props_len, device=device).long()

    def proposal_generator(self, h, bsz):
        """DPG: dual-granularity proposal boundaries.

        Global proposals come from a direct regression head, local proposals
        from SlotAttention; the Hungarian assignment pairs the two sets per
        sample and a learnable convex weight fuses them.
        """
        h_slot = self.fc_sl(self.slot_atten(h))
        reshaped_tensor = h_slot.squeeze(-1).view(bsz*self.num_props, 2)
        gauss_param = (torch.tanh(reshaped_tensor) + 1) / 2  # [-1,1] -> [0,1]
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        gauss_param_raw = torch.sigmoid(
            self.fc_gauss(h[:, -1]).view(bsz*self.num_props, 2))
        from scipy.optimize import linear_sum_assignment
        local_param = torch.stack([gauss_center, gauss_width], dim=1)
        global_b = gauss_param_raw.view(bsz, self.num_props, 2)
        local_b = local_param.view(bsz, self.num_props, 2)
        matched = []
        for sample_idx in range(bsz):
            cost = torch.cdist(global_b[sample_idx], local_b[sample_idx]).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            order = np.argsort(row_ind)
            col_ind = torch.as_tensor(col_ind[order], device=h.device, dtype=torch.long)
            matched.append(local_b[sample_idx].index_select(0, col_ind))
        matched = torch.stack(matched, dim=0).view(bsz*self.num_props, 2)
        matched_center, matched_width = matched[:, 0], matched[:, 1]
        global_center = gauss_param_raw[:, 0]
        global_width = gauss_param_raw[:, 1]

        fusion_alpha = torch.sigmoid(self.merge_alpha)
        fusion_weights = torch.stack(
            [fusion_alpha, 1.0-fusion_alpha], dim=0)
        gauss_center = (global_center * fusion_weights[0] +
                        matched_center * fusion_weights[1])
        gauss_width = (global_width * fusion_weights[0] +
                       matched_width * fusion_weights[1])
        if self.proposal_width_power != 1.0:
            gauss_width = gauss_width.clamp_min(1e-6).pow(self.proposal_width_power)
        return gauss_center, gauss_width

    def semantic_alignment(self, frames_feat, word_cls, gauss_center, gauss_width, num_proposal, margin=0.5):
        """L_sem: align proposal-interior representations with the query."""
        bs, seq_len, dim = frames_feat.shape
        device = frames_feat.device
        num_total = bs * num_proposal

        # Coordinate transformation
        center_idx = (gauss_center * (seq_len-1)).long()  # (num_total,)
        width = (gauss_width * seq_len).long()            # (num_total,)

        # Generate Index Grid
        frames_expanded = frames_feat.repeat_interleave(num_proposal, dim=0)  # (num_total, seq_len, dim)
        time_idx = torch.arange(seq_len, device=device).expand(num_total, seq_len)

        # Create proposal masks
        start = torch.clamp(center_idx - width//2, 0, seq_len-1).unsqueeze(1)
        end = torch.clamp(center_idx + width//2 + 1, 0, seq_len).unsqueeze(1)
        proposal_mask = (time_idx >= start) & (time_idx < end)  # (num_total, seq_len)

        # Feature extraction of internal frames of proposal (query-related frames)
        proposal_feats = torch.zeros(num_total, dim, device=device)
        valid_proposal = proposal_mask.sum(dim=1) > 0
        proposal_feats[valid_proposal] = (
            frames_expanded[valid_proposal] *
            proposal_mask[valid_proposal].unsqueeze(-1)
        ).sum(dim=1) / proposal_mask[valid_proposal].sum(dim=1, keepdim=True)

        # Feature extraction of external frames of proposal (query-irrelated frames)
        non_proposal_mask = ~proposal_mask
        non_proposal_feats = torch.zeros(num_total, dim, device=device)
        valid_non_proposal = non_proposal_mask.sum(dim=1) > 0

        # Handling the situation where there are non-proposal areas
        non_proposal_feats[valid_non_proposal] = (
            frames_expanded[valid_non_proposal] *
            non_proposal_mask[valid_non_proposal].unsqueeze(-1)
        ).sum(dim=1) / non_proposal_mask[valid_non_proposal].sum(dim=1, keepdim=True)

        # Handling the special case where the entire sequence is selected
        full_mask = (proposal_mask.sum(dim=1) == seq_len)
        non_proposal_feats[full_mask] = frames_expanded[full_mask].mean(dim=1)

        # Calculating Similarity
        word_cls_expanded = word_cls.repeat_interleave(num_proposal, dim=0)
        sim_pos = (F.cosine_similarity(proposal_feats, word_cls_expanded) /
                   self.semantic_temperature)
        sim_neg = (F.cosine_similarity(non_proposal_feats, word_cls_expanded) /
                   self.semantic_temperature)

        # Contrastive loss
        return F.margin_ranking_loss(
            sim_pos, sim_neg, torch.ones_like(sim_pos), margin=self.semantic_margin)

    def peak_aware_gauss_weight(self, props_len, center, width, effective_sigma=None):
        """PFA: peak-aware Gaussian temporal response of each proposal."""
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)  # [batch_size, 1]
        if effective_sigma is None:
            effective_sigma = torch.full_like(width, float(self.sigma))
        width = width.unsqueeze(-1).clamp(1e-2) / effective_sigma.unsqueeze(-1)

        # Calculate Gaussian weights
        w = 0.3989422804014327  # 1/sqrt(2*pi)
        gauss_weight = w / width * torch.exp(-(weight - center)**2 / (2 * width**2))
        gauss_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]

        # Compute the mask of the peak region (differentiable)
        # Using sigmoid smoothing instead of Boolean masking
        delta = weight - center
        mask = torch.sigmoid(
            (self.peak_beta() * width - torch.abs(delta)) * 1e3
        )  # [batch_size, props_len]

        # Increase the weight of the peak area to 1 (differentiable mixing)
        final_weight = gauss_weight * (1 - mask) + 1.0 * mask
        return final_weight

    def peak_beta(self):
        if self.peak_beta_parameterization == 'softplus':
            return F.softplus(self.range_width)
        return self.range_width

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            del pos
            effective_sigma = torch.full_like(w1, float(self.sigma_neg))
            return self.peak_aware_gauss_weight(
                props_len, c, w1, effective_sigma=effective_sigma)

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        curriculum_progress = min(
            epoch / self.negative_curriculum_epochs, 1)**self.gamma
        left_center = left_width * curriculum_progress * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * curriculum_progress * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.to(words_feat.device).unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1)
            masked_words.append(torch.zeros([words_feat.size(1)], dtype=torch.bool, device=words_feat.device))
            if l < 1:
                continue
            if not self.training and self.deterministic_eval_mask:
                if weights is None:
                    choices = np.arange(1, num_masked_words + 1)
                else:
                    # Stable content-first masking; lower token position breaks ties.
                    scores = weights[i, :l].detach().cpu().numpy()
                    choices = np.lexsort((np.arange(l), -scores))[:num_masked_words] + 1
            else:
                p = weights[i, :l].cpu().numpy() if weights is not None else None
                choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1

        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    mask = []
    for l in x_len:
        mask.append(torch.zeros([x.size(1)], dtype=torch.bool, device=x.device))
        mask[-1][:l] = 1
    return torch.stack(mask, 0)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1), ], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
