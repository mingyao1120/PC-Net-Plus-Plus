"""Fail-closed guard for the PC-Net++ evaluation scope."""

ALLOWED_FAMILIES = {'cpr', 'cf_marginal_set'}

# Config sections and model switches outside the paper's scope are rejected
# outright: the released entry points never produce them, and any manual edit
# that reintroduces them fails fast instead of silently changing the method.
ARCHIVED_SECTION_PREFIXES = (
    'distillation', 'calibrator', 'verifier', 'scorer', 'consensus',
    'response_counterfactual', 'query_generalization',
    'counterfactual_utility_training',
)

BACKBONE_INVARIANTS = {
    'hidden_size': 256,
    'num_props': 8,
    'num_iteration': 4,
    'proposal_generator_mode': 'slot_hungarian',
    'proposal_match_mode': 'per_sample',
    'proposal_feature_downsample': 4,
    'proposal_fusion_mode': 'center_width_learnable',
    'proposal_width_power': 1.0,
    'gaussian_mode': 'fixed',
    'proposal_kernel': 'gaussian',
    'peak_beta_parameterization': 'direct',
}


def validate_active_scope(args):
    archived = sorted(key for key in args
                      if isinstance(key, str) and
                      any(key.startswith(prefix)
                          for prefix in ARCHIVED_SECTION_PREFIXES))
    if archived:
        raise ValueError('archived experimental sections present: %s' % archived)

    model = args['model']['config']
    qvhighlights = args.get('dataset', {}).get('dataset') == 'QVHighlights'
    for key, expected in BACKBONE_INVARIANTS.items():
        if qvhighlights and key == 'proposal_feature_downsample':
            expected = 1
        actual = model.get(key)
        if actual != expected:
            raise ValueError('backbone invariant %s=%r, expected %r' % (
                key, actual, expected))
    fusion_alpha = float(model.get('proposal_fusion_alpha', 0.5))
    if not 0.0 < fusion_alpha < 1.0:
        raise ValueError('proposal_fusion_alpha must be in (0, 1)')

    candidates = args.get('evaluation', {}).get('selector_candidates', [])
    for candidate in candidates:
        family = candidate.get('family')
        if family not in ALLOWED_FAMILIES:
            raise ValueError('selector family is outside active scope: %r' % family)
        if candidate.get('mode', 'raw') != 'raw':
            raise ValueError('only raw CF scoring is active')
        if int(candidate.get('views', 1)) != 1:
            raise ValueError('multi-view selection is archived')
        if family == 'cpr':
            if int(candidate.get('boundary_topk', 0)) != 2:
                raise ValueError('CPR requires boundary_topk=2')
            if 'boundary_alpha' not in candidate:
                raise ValueError('CPR requires boundary_alpha')
        elif family == 'cf_marginal_set':
            if not qvhighlights:
                raise ValueError('marginal set deletion is QV-only')
            if float(candidate.get('fuse_alpha', 0.0)) < 0.0:
                raise ValueError('fuse_alpha must be non-negative')
