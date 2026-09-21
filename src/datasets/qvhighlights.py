"""QVHighlights loader for the native PC-Net/PC-Net++ runner.

The annotation format is ``[vid, duration, windows, query, qid]``.  Training
remains timestamp-free inside PC-Net; the longest window is exposed only to the
legacy validation metric.  Official evaluation consumes every window from the
separate JSONL ground truth.
"""

import os
import pickle

import numpy as np

from .base import BaseDataset, build_collate_data


NUM_CLIPS = 75
CLIP_SECONDS = 2.0


def _longest_window(windows):
    if not windows:
        return None
    return max(windows, key=lambda window: float(window[1]) - float(window[0]))


def _window_saliency(windows, duration):
    values = np.zeros(NUM_CLIPS, dtype=np.float32)
    for start, end in windows:
        left = int(np.clip(np.floor(float(start) / CLIP_SECONDS), 0, NUM_CLIPS - 1))
        right = int(np.clip(np.ceil(float(end) / CLIP_SECONDS), 0, NUM_CLIPS))
        values[left:max(right, left + 1)] = 1.0
    return values


def _annotated_saliency(entry):
    values = np.zeros(NUM_CLIPS, dtype=np.float32)
    for clip_id, scores in zip(entry.get("clip_ids", []), entry.get("scores", [])):
        clip_id = int(clip_id)
        if 0 <= clip_id < NUM_CLIPS and scores:
            values[clip_id] = float(np.mean(scores))
    return values


class QVHighlights(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.frame_dim = args.get("frame_dim", args.get("frame_feat_dim"))
        self.word_dim = args.get("word_dim", args.get("word_feat_dim"))
        self.max_frames = args.get("max_num_frames", args.get("max_num_segments", NUM_CLIPS))
        self.collate_fn = build_collate_data(
            self.max_frames, args["max_num_words"], self.frame_dim, self.word_dim)
        self.text_backend = args.get("text_backend", "glove")
        self._clip_index = None
        if self.text_backend == "clip":
            path = args.get("clip_text_index")
            if not path or not os.path.isfile(path):
                raise FileNotFoundError("CLIP text index is missing: %r" % path)
            with open(path, "rb") as handle:
                self._clip_index = pickle.load(handle)
            encoded = []
            missing = []
            for row in self.data:
                qid = int(row[4])
                entry = self._clip_index.get(qid)
                if not isinstance(entry, dict) or "seq" not in entry:
                    missing.append(qid)
                    continue
                sequence = np.asarray(entry["seq"], dtype=np.float32)
                if sequence.ndim != 2 or sequence.shape[1] != self.word_dim:
                    raise ValueError(
                        "invalid CLIP sequence for qid=%s: %s, expected [T,%s]" %
                        (qid, sequence.shape, self.word_dim))
                if len(sequence) < 1:
                    raise ValueError("empty CLIP sequence for qid=%s" % qid)
                pool = np.asarray(entry.get("pool"), dtype=np.float32)
                if pool.shape != (self.word_dim,):
                    raise ValueError(
                        "invalid CLIP pooled feature for qid=%s: %s" %
                        (qid, pool.shape))
                weight_mode = args.get("clip_token_weight_mode", "uniform")
                if weight_mode == "uniform":
                    token_weights = np.ones(len(sequence), dtype=np.float32)
                elif weight_mode == "pool_similarity":
                    # CLIP-only semantic salience: favour contextual tokens that
                    # contribute most to the query embedding.  This consumes no
                    # temporal window or highlight annotation.
                    denom = (np.linalg.norm(sequence, axis=1) *
                             max(float(np.linalg.norm(pool)), 1e-6))
                    similarity = np.sum(sequence * pool[None], axis=1) / np.maximum(
                        denom, 1e-6)
                    scale = float(args.get("clip_token_weight_scale", 2.0))
                    token_weights = (scale * similarity).astype(np.float32)
                else:
                    raise ValueError("unknown clip_token_weight_mode: %s" %
                                     weight_mode)
                encoded.append({
                    "words_feat": np.concatenate((sequence[:1], sequence), axis=0),
                    "words_id": np.zeros(len(sequence), dtype=np.int64),
                    "words_clip": sequence,
                    "weights": token_weights,
                    "word_pos": np.full(len(sequence), 4, dtype=np.int64),
                    "words_clip_pool": pool,
                })
            if missing:
                raise KeyError("CLIP text index misses %d qids; first=%s" %
                               (len(missing), missing[:5]))
            self._encoded_queries = encoded
        self.qid_to_saliency = None
        path = args.get("qid2saliency")
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                self.qid_to_saliency = pickle.load(handle)

    def _load_frame_features(self, vid):
        import h5py

        paths = self.args["feature_path"]
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        features = []
        for path in paths:
            handle = self._get_h5(path)
            features.append(np.asarray(handle[vid], dtype=np.float32))
        length = min(feature.shape[0] for feature in features)
        return np.concatenate([feature[:length] for feature in features], axis=-1)

    def _build_words(self, sentence):
        import nltk

        words = []
        weights = []
        word_pos = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            if word not in self.keep_vocab:
                continue
            words.append(word)
            if "VB" in tag:
                weights.append(4)
                word_pos.append(2)
            elif "NN" in tag or "JJ" in tag or "RB" in tag:
                weights.append(2)
                word_pos.append(1 if "NN" in tag else 3)
            else:
                weights.append(1)
                word_pos.append(4)
        if not words:
            return [], [np.zeros(self.word_dim, dtype=np.float32)], [], []
        ids = [self.keep_vocab[word] for word in words]
        vectors = [self.vocab["id2vec"][self.vocab["w2id"][words[0]]].astype(np.float32)]
        vectors.extend(
            self.vocab["id2vec"][self.vocab["w2id"][word]].astype(np.float32)
            for word in words)
        return ids, vectors, weights, word_pos

    def __getitem__(self, index):
        vid, duration, windows, sentence, qid = self.data[index]
        duration = float(duration)
        target = _longest_window(windows) or [0.0, duration]
        encoded = self._encoded_queries[index]
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))
        saliency = (_annotated_saliency(self.qid_to_saliency[int(qid)])
                    if self.qid_to_saliency is not None and int(qid) in self.qid_to_saliency
                    else _window_saliency(windows, duration))
        return {
            "frames_feat": frames_feat,
            "words_feat": encoded["words_feat"],
            "words_id": encoded["words_id"],
            "weights": encoded["weights"],
            "word_pos": encoded["word_pos"],
            **({"words_clip": encoded["words_clip"]}
               if "words_clip" in encoded else {}),
            **({"words_clip_pool": encoded["words_clip_pool"]}
               if "words_clip_pool" in encoded else {}),
            "raw": [vid, duration, [float(target[0]), float(target[1])], sentence,
                    int(qid), [[float(start), float(end)] for start, end in windows]],
            "saliency_gt": saliency,
        }

    def collate_data(self, samples):
        import torch

        batch = self.collate_fn(samples)
        # Keep annotations outside net_input.  PC-Net-Core is trained as a
        # weakly-supervised model and must never consume HD ratings or MR windows.
        batch["saliency_gt_eval_only"] = torch.from_numpy(
            np.stack([sample["saliency_gt"] for sample in samples]).astype(np.float32))
        if samples and "words_clip" in samples[0]:
            net_input = batch["net_input"]
            length = net_input["words_id"].shape[1]
            target = np.zeros(
                (len(samples), length, self.word_dim), dtype=np.float32)
            for index, sample in enumerate(samples):
                keep = min(len(sample["words_clip"]), length)
                target[index, :keep] = sample["words_clip"][:keep]
            net_input["words_clip"] = torch.from_numpy(target)
            net_input["words_clip_pool"] = torch.from_numpy(np.stack(
                [sample["words_clip_pool"] for sample in samples]).astype(np.float32))
        return batch
