import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import load_json
import nltk

class BaseDataset(Dataset):
    # Per-process cache of open h5py file handles. Each DataLoader worker is a
    # separate process, so this cache lives once per worker. Reusing the handle
    # avoids re-opening / re-parsing the (large) HDF5 file on every __getitem__,
    # which is a major dataloader bottleneck. The parent (main) process never
    # calls __getitem__, so this stays empty there and is not pickled.
    _h5_cache = {}

    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self._sampled_features = None
        self._sampled_feature_ids = None
        sampled_cache = args.get('sampled_feature_cache')
        sampled_index = args.get('sampled_feature_index')
        if sampled_cache and sampled_index:
            if not os.path.isfile(sampled_cache) or not os.path.isfile(sampled_index):
                raise FileNotFoundError(
                    'sampled feature cache is incomplete; run '
                    '`scripts/build_feature_cache.py`: %s / %s' %
                    (sampled_cache, sampled_index))
            index_value = load_json(sampled_index)
            self._sampled_feature_ids = index_value['video_to_index']
            self._sampled_features = np.load(sampled_cache, mmap_mode='r')
            expected = (len(self._sampled_feature_ids), self.max_num_frames,
                        int(args['frame_dim']))
            if tuple(self._sampled_features.shape) != expected:
                raise ValueError('sampled feature cache shape %s != %s' % (
                    tuple(self._sampled_features.shape), expected))

        self.keep_vocab = dict()
        requested_vocab_size = args.get('vocab_size')
        # A non-positive/omitted value means "all words represented by this
        # vocabulary file".  Existing positive values retain exact checkpoint
        # compatibility (Ego4D=2604 and TACoS=1734 include the reserved id).
        if requested_vocab_size is None or int(requested_vocab_size) <= 0:
            requested_vocab_size = len(vocab['counter'])
        for w, _ in vocab['counter'].most_common(int(requested_vocab_size)):
            self.keep_vocab[w] = self.vocab_size

        # Tokenization and POS tagging used to run in every __getitem__ call,
        # i.e. once per sample per epoch in every worker.  Encode the immutable
        # query side once in the parent process; forked workers share these
        # arrays copy-on-write and only perform HDF5 feature reads.
        self._word_vectors = {
            word: np.asarray(vocab['id2vec'][vocab['w2id'][word]],
                             dtype=np.float32)
            for word in self.keep_vocab
        }
        self._encoded_queries = [self._encode_query(row[3]) for row in self.data]

    def _get_h5(self, path):
        # Never reuse an HDF5 handle inherited from another process after fork.
        key = (os.getpid(), os.path.realpath(path))
        fr = BaseDataset._h5_cache.get(key)
        if fr is None:
            fr = h5py.File(path, 'r')
            BaseDataset._h5_cache[key] = fr
        return fr

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _sample_frame_features(self, frames_feat):
        frames_feat = np.asarray(frames_feat, dtype=np.float32)
        if frames_feat.ndim != 2 or len(frames_feat) == 0:
            raise ValueError('video feature must be a non-empty [time, dim] array')
        num_clips = self.num_clips
        keep_idx = np.arange(0, num_clips + 1) / num_clips * len(frames_feat)
        keep_idx = np.round(keep_idx).astype(np.int64)
        keep_idx[keep_idx >= len(frames_feat)] = len(frames_feat) - 1
        starts, ends = keep_idx[:-1], keep_idx[1:]

        # Prefix-sum pooling replaces 200 Python slice/mean calls per sample.
        # Repeated boundaries retain the legacy behavior of selecting frame s.
        prefix = np.concatenate((
            np.zeros((1, frames_feat.shape[1]), dtype=np.float32),
            np.cumsum(frames_feat, axis=0, dtype=np.float32)), axis=0)
        lengths = ends - starts
        safe_ends = np.maximum(ends, starts + 1)
        pooled = (prefix[safe_ends] - prefix[starts]) / np.maximum(
            lengths, 1)[:, None]
        repeated = lengths == 0
        if np.any(repeated):
            pooled[repeated] = frames_feat[starts[repeated]]
        return np.asarray(pooled, dtype=np.float32)

    def _encode_query(self, sentence):
        weights = []
        word_pos = []
        words = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            if word not in self.keep_vocab:
                continue
            if 'NN' in tag:
                weights.append(2)
                word_pos.append(1)
            elif 'VB' in tag:
                weights.append(4)
                word_pos.append(2)
            elif 'JJ' in tag or 'RB' in tag:
                weights.append(2)
                word_pos.append(3)
            else:
                weights.append(1)
                word_pos.append(4)
            words.append(word)

        if not words:
            # Robust handling for a fully OOV rewritten query.  Id 0 is the
            # reserved class and the zero embedding is replaced by start_vec at
            # model entry, so this does not borrow any held-out vocabulary.
            zero = np.zeros(int(self.args['word_dim']), dtype=np.float32)
            return {
                'words_feat': np.stack((zero, zero)),
                'words_id': np.asarray([0], dtype=np.int64),
                'weights': np.asarray([1.0], dtype=np.float32),
                'word_pos': np.asarray([4], dtype=np.int64),
            }

        vectors = np.stack([self._word_vectors[word] for word in words])
        return {
            'words_feat': np.concatenate((vectors[:1], vectors), axis=0),
            'words_id': np.asarray(
                [self.keep_vocab[word] for word in words], dtype=np.int64),
            'weights': np.asarray(weights, dtype=np.float32),
            'word_pos': np.asarray(word_pos, dtype=np.int64),
        }

    @property
    def num_clips(self):
        return self.max_num_frames

    @property
    def vocab_size(self):
        return len(self.keep_vocab) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        duration = float(duration)
        encoded = self._encoded_queries[index]
        if self._sampled_features is not None:
            cache_index = self._sampled_feature_ids.get(str(vid))
            if cache_index is None:
                raise KeyError('video %s is absent from sampled feature cache' % vid)
            frames_feat = np.asarray(
                self._sampled_features[int(cache_index)], dtype=np.float32)
        else:
            frames_feat = self._sample_frame_features(self._load_frame_features(vid))

        return {
            'frames_feat': frames_feat,
            'words_feat': encoded['words_feat'],
            'words_id': encoded['words_id'],
            'weights': encoded['weights'],
            'word_pos': encoded['word_pos'],
            'raw': [vid, duration, timestamps, sentence]
        }
        

def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        frames_len = []
        words_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))
            words_len.append(min(len(sample['words_id']), max_num_words))

        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)
        words_feat = np.zeros([bsz, max(words_len) + 1, word_dim]).astype(np.float32)
        words_id = np.zeros([bsz, max(words_len)]).astype(np.int64)
        weights = np.zeros([bsz, max(words_len)]).astype(np.float32)
        word_pos = np.zeros([bsz, max(words_len)]).astype(np.int64)
        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)
            keep = min(len(sample['word_pos']), word_pos.shape[1])
            word_pos[i, :keep] = sample['word_pos'][:keep]

        batch.update({
            'net_input': {
                'frames_feat': torch.from_numpy(frames_feat),
                'frames_len': torch.from_numpy(np.asarray(frames_len)),
                'words_feat': torch.from_numpy(words_feat),
                'words_id': torch.from_numpy(words_id),
                'weights': torch.from_numpy(weights),
                'word_pos': torch.from_numpy(word_pos),
                'words_len': torch.from_numpy(np.asarray(words_len)),
                # Video duration is metadata, not temporal supervision.  It enables
                # length-aware aggregation without exposing the target timestamps.
                'duration': torch.from_numpy(np.asarray(
                    [float(sample['raw'][1]) for sample in samples], dtype=np.float32)),
            }
        })
        return batch

    return collate_data
