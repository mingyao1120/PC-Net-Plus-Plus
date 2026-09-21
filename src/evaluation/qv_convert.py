"""Convert generic runner predictions into the official QVHighlights format
and compute official MR/HD metrics in-process.

The highlight scores are derived exclusively from the model's reconstruction
Gaussian curves (max over proposals, interpolated to 75 clips, sigmoid) — no
HD ratings or MR window labels are consumed anywhere in this path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

# Official Moment-DETR standalone evaluator, vendored under standalone_eval/
# (files verified by SHA256 in load_official_evaluator).
DEFAULT_EVALUATOR_ROOT = Path(__file__).resolve().parent / "standalone_eval"


def load_official_evaluator(root=None):
    """Import the pinned standalone evaluator package and verify its hashes."""
    root = Path(root) if root is not None else DEFAULT_EVALUATOR_ROOT
    pinned = {
        "eval.py": "a5a74cde960345901693e7a6b383a412f8a02b7f809328054dc301fe3c2bfc59",
        "utils.py": "7cb2f9733ffb3fe22d0b4044ef7ba82679b85972a3e7e0ac88b3eee44904d637",
    }
    import hashlib
    for name, expected in pinned.items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                "QV evaluator hash mismatch for %s: %s" % (name, digest))
    # eval.py imports ``standalone_eval.utils``; make its parent importable.
    import sys
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    import standalone_eval.eval as module
    return module


def convert_records(records, num_clips=75, window_decimals=2):
    """Runner records -> official submission rows.

    ``window_decimals=2`` mirrors the original data-preparation convention so
    threshold ties are resolved from byte-for-byte compatible boundaries.
    Pass ``None`` only for an explicit full-precision diagnostic.
    """
    import torch
    import torch.nn.functional as functional

    submission = []
    for record in records:
        if "qid" not in record:
            raise ValueError("QV official evaluation requires qid in records")
        if "highlight_scores" in record:
            raw_saliency = torch.as_tensor(
                np.asarray(record["highlight_scores"]), dtype=torch.float32)
            saliency = functional.interpolate(
                raw_saliency.view(1, 1, -1), size=num_clips,
                mode="linear", align_corners=True).view(-1).sigmoid().tolist()
        else:
            curves = torch.as_tensor(
                np.asarray(record["gaussian_curves"]), dtype=torch.float32)
            saliency = functional.interpolate(
                curves.max(dim=0).values.view(1, 1, -1), size=num_clips,
                mode="linear", align_corners=True).view(-1).sigmoid().tolist()
        def boundary(value):
            value = float(value)
            return round(value, window_decimals) if window_decimals is not None else value

        submission.append({
            "qid": int(record["qid"]),
            "vid": record["vid"],
            "pred_relevant_windows": [
                [boundary(w[0]), boundary(w[1]), float(w[2])] if len(w) > 2
                else [boundary(w[0]), boundary(w[1]), 0.0]
                for w in record["pred_windows"]],
            "pred_saliency_scores": saliency,
        })
    return submission


def official_qv_metrics(records, gt_jsonl, evaluator_root=None, match_number=True):
    """Run the pinned official evaluator over runner records.

    Metric names follow the official evaluator.  ``MR-mAP`` and
    ``MR-full-mAP`` are the official average over IoU 0.5:0.05:0.95.  The
    former Unified-only score is retained under the unambiguous compatibility
    name ``MR-mAP-composite`` and must not be reported as official mAP Avg.
    """
    submission = convert_records(records)
    gt = [json.loads(line) for line in
          Path(gt_jsonl).read_text(encoding="utf8").splitlines() if line.strip()]
    module = load_official_evaluator(evaluator_root)
    eval_metrics = module.eval_submission(
        submission, gt, verbose=False, match_number=match_number)

    def flat(prefix, value):
        return {f"{prefix}-{k}": float(v) for k, v in value.items()}

    metrics = {}
    mr = eval_metrics.get("full", {}).get("MR-mAP", {})
    if mr:
        metrics["MR-mAP@0.5"] = float(mr["0.5"])
        metrics["MR-mAP@0.75"] = float(mr["0.75"])
        metrics["MR-full-mAP"] = float(mr["average"])
        metrics["MR-mAP"] = float(mr["average"])
        metrics["MR-mAP-composite"] = (
            0.5 * float(mr["average"])
            + 0.25 * float(mr["0.5"])
            + 0.25 * float(mr["0.75"]))
        for name, block in (("short", "MR-short-mAP"), ("middle", "MR-middle-mAP"),
                            ("long", "MR-long-mAP")):
            if name in eval_metrics:
                metrics[block] = float(eval_metrics[name]["MR-mAP"]["average"])
        r1 = eval_metrics["full"].get("MR-R1", {})
        if r1:
            metrics["MR-full-R1@0.3"] = float(r1["0.3"])
            metrics["MR-full-R1@0.5"] = float(r1["0.5"])
            metrics["MR-full-R1@0.7"] = float(r1["0.7"])
    for name, value in eval_metrics.items():
        if name.startswith("HL-min-"):
            tag = name[len("HL-min-"):]
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    metrics[f"HD-{tag}-{sub_key}"] = float(sub_value)
            else:
                metrics[f"HD-{tag}"] = float(value)
    return metrics
