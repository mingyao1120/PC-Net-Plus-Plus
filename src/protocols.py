"""Experiment protocol switches.

The default is deliberately ``legacy_exact`` so old commands keep their historical
behaviour.  ``corrected`` enables the auditable TPAMI evaluation/training protocol.
"""


VALID_PROTOCOLS = ("legacy_exact", "corrected", "unified_paper")


def apply_protocol(args, name):
    if name not in VALID_PROTOCOLS:
        raise ValueError("unknown protocol: %s" % name)

    protocol = args.setdefault("protocol", {})
    protocol["mode"] = name
    model = args["model"]["config"]
    train = args.setdefault("train", {})
    dataset = args["dataset"]

    if name == "legacy_exact":
        protocol.setdefault("selection_split", "test")
        protocol.setdefault("strict_train_vocab", False)
        protocol.setdefault("deterministic_eval_mask", False)
        protocol.setdefault("score_masked_only", False)
        model.setdefault("proposal_match_mode", "global_batch")
        model.setdefault("correct_width_fusion", False)
        model.setdefault("skip_negative_eval", False)
        model.setdefault("correct_negative_peak_delta", False)
    else:
        selection_split = protocol.get("selection_split", "test")
        protocol.update({
            "selection_split": selection_split,
            "strict_train_vocab": True,
            "deterministic_eval_mask": True,
            "correct_negative_peak_delta": True,
            "score_masked_only": True,
        })
        model.update({
            "proposal_match_mode": "per_sample",
            "correct_width_fusion": True,
            "skip_negative_eval": True,
            "deterministic_eval_mask": True,
        })
        if name == "unified_paper":
            protocol.update({
                "strict_train_vocab": False,
                "score_masked_only": False,
            })
        if selection_split == "val" and not dataset.get("val_data"):
            raise ValueError("corrected protocol requires dataset.val_data")
        if (selection_split == "val" and
                dataset.get("val_data") == dataset.get("test_data") and
                not dataset.get("allow_val_equals_test", False)):
            raise ValueError("corrected protocol forbids val_data == test_data; create a video-disjoint val split")

    train.setdefault("num_workers", 16)
    train.setdefault("val_num_workers", 1)
    return args
