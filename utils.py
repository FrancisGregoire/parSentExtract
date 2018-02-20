from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from itertools import product
from six.moves import xrange

import numpy as np
import tensorflow as tf

from scipy.spatial.distance import cdist


EPSILON = 1e-8
START_VOCAB = ["_PAD", "_UNK"]
UNK_ID = 1
NORMALIZE_DIGIT = re.compile("\d")


class TrainingIterator(object):
    """Class used to train BiRNN."""
    def __init__(self, data, n_negative=1):
        self.data = data
        self.epoch_data = None
        self.n_negative = n_negative
        self.global_step = 0
        self.epoch_completed = 0
        self._index_in_epoch = 0
        self._generate_epoch_data()
        self.size = len(self.epoch_data)

    def _pad_batch(self, data):
        seq_length = np.array([(len(source), len(target)) for (source, target, _) in data])
        max_length = np.max(seq_length, axis=0)
        pad_source = np.zeros((len(data), max_length[0]), dtype=np.int32)
        pad_target = np.zeros((len(data), max_length[1]), dtype=np.int32)
        for i, (source, target, _) in enumerate(data):
            pad_source[i, :seq_length[i, 0]] = source
            pad_target[i, :seq_length[i, 1]] = target
        return pad_source, pad_target, data[:, 2]

    def _generate_epoch_data(self):
        pos = np.ones((len(self.data), 1))
        neg = np.zeros((len(self.data), 1))
        epoch_data = np.hstack((self.data, pos))
        for _ in xrange(self.n_negative):
            neg_data = np.copy(self.data)
            np.random.shuffle(neg_data[:, 1])
            index = np.where(self.data[:, 1] == neg_data[:, 1])[0]
            while len(index) > 0:
                rand_index = np.random.choice(len(self.data), len(index))
                neg_data[index, 1] = self.data[rand_index, 1]
                index = np.where(self.data[:, 1] == neg_data[:, 1])[0]
            neg_data = np.hstack((neg_data, neg))
            epoch_data = np.vstack((epoch_data, neg_data))
        self.epoch_data = epoch_data
        np.random.shuffle(self.epoch_data)

    def next_batch(self, batch_size):
        self.global_step += 1
        start = self._index_in_epoch
        if start + batch_size > self.size:
            self.epoch_completed += 1
            size_not_observed = self.size - start
            data_not_observed = self.epoch_data[start:self.size]
            self._generate_epoch_data()
            start = 0
            self._index_in_epoch = batch_size - size_not_observed
            end = self._index_in_epoch
            batch_data = np.concatenate((data_not_observed, self.epoch_data[start:end]), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            batch_data = self.epoch_data[start:end]
        return self._pad_batch(batch_data)


class EvalIterator(object):
    """Class used to evaluate BiRNN."""
    def __init__(self, data):
        self.data = data
        self.global_step = 0
        self.epoch_completed = 0
        self._index_in_epoch = 0
        self._generate_epoch_data()
        self.size = len(self.epoch_data)

    def _pad_batch(self, data):
        seq_length = np.array([(len(source), len(target)) for (source, target, _) in data])
        max_length = np.max(seq_length, axis=0)
        pad_source = np.zeros((len(data), max_length[0]), dtype=np.int32)
        pad_target = np.zeros((len(data), max_length[1]), dtype=np.int32)
        for i, (source, target, _) in enumerate(data):
            pad_source[i, :seq_length[i, 0]] = source
            pad_target[i, :seq_length[i, 1]] = target
        return pad_source, pad_target, data[:, 2]

    def _generate_epoch_data(self):
        source, target = zip(*self.data.tolist())
        epoch_data = list(zip(source, target, [1.0] * len(source)))
        epoch_data += [(source[i], target[j], 0.0) for i, j in product(xrange(len(source)), xrange(len(target)))
                       if target[i] != target[j]]
        self.epoch_data = np.array(epoch_data, dtype=object)

    def next_batch(self, batch_size):
        self.global_step += 1
        start = self._index_in_epoch
        if start + batch_size > self.size:
            self.epoch_completed += 1
            size_not_observed = self.size - start
            data_not_observed = self.epoch_data[start:self.size]
            start = 0
            self._index_in_epoch = batch_size - size_not_observed
            end = self._index_in_epoch
            batch_data = np.concatenate((data_not_observed, self.epoch_data[start:end]), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            batch_data = self.epoch_data[start:end]
        return self._pad_batch(batch_data)


class TestingIterator(object):
    """Class used to test BiRNN."""
    def __init__(self, data):
        self.data = data
        self.global_step = 0
        self.epoch_completed = 0
        self._index_in_epoch = 0
        self.size = len(self.data)

    def _pad_batch(self, data):
        seq_length = np.array([(len(source), len(target)) for (source, target, _) in data])
        max_length = np.max(seq_length, axis=0)
        pad_source = np.zeros((len(data), max_length[0]), dtype=np.int32)
        pad_target = np.zeros((len(data), max_length[1]), dtype=np.int32)
        for i, (source, target, _) in enumerate(data):
            pad_source[i, :seq_length[i, 0]] = source
            pad_target[i, :seq_length[i, 1]] = target
        return pad_source, pad_target, data[:, 2]

    def next_batch(self, batch_size):
        self.global_step += 1
        start = self._index_in_epoch
        if start + batch_size > self.size:
            self.epoch_completed += 1
            size_not_observed = self.size - start
            data_not_observed = self.data[start:self.size]
            start = 0
            self._index_in_epoch = batch_size - size_not_observed
            end = self._index_in_epoch
            batch_data = np.concatenate((data_not_observed, self.data[start:end]), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            batch_data = self.data[start:end]
        return self._pad_batch(batch_data)


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
    """Create vocabulary file from data file."""
    vocab = {}
    with open(data_path, mode="r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            for word in tokens:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w", encoding="utf-8") as vocab_file:
            for word in vocab_list:
                  vocab_file.write(word + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file."""
    if os.path.exists(vocabulary_path):
        with open(vocabulary_path, mode="r", encoding="utf-8") as vocab_file:
            rev_vocab = [line.strip() for line in vocab_file.readlines()]
        vocab = dict([(w, i) for (i, w) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file {} not found.".format(vocabulary_path))


def sentence_to_token_ids(sentence, vocabulary, max_sequence_length):
    """Convert a string to a list of integers representing token-ids."""
    words = sentence.strip().split()
    if len(words) > max_sequence_length:
        words = words[:max_sequence_length]
    return [vocabulary.get(w, UNK_ID) for w in words]


def read_data(source_path, target_path, source_vocab, target_vocab, max_seq_length=200):
    """Read parallel data and convert to token ids."""
    data = []
    with open(source_path, mode="r", encoding="utf-8") as source_file,\
         open(target_path, mode="r", encoding="utf-8") as target_file:
            for source, target in zip(source_file, target_file):
                source_data = sentence_to_token_ids(source, source_vocab, max_seq_length)
                target_data = sentence_to_token_ids(target, target_vocab, max_seq_length)
                data.append((source_data, target_data))
    return np.array(data, dtype=object)


def read_data_with_ref(source_path, target_path, ref_path):
    """Read sentences and parallel sentence references."""
    with open(source_path, "r", encoding="utf-8") as source_file,\
         open(target_path, "r", encoding="utf-8") as target_file:
            source_lines = [l for l in source_file]
            target_lines = [l for l in target_file]
    references = set()
    with open(ref_path, mode="r", encoding="utf-8") as ref_file:
        for l in ref_file:
            i, j = l.split()
            references.add((int(i), int(j)))
    return source_lines, target_lines, references


def sequence_length(sequence):
    """Returns a np array of the sequence lengths."""
    return np.sum(np.sign(sequence), axis=1, dtype=np.int32)


def l2_normalize(data):
    """Scale input vectors individually to unit norm."""
    if data.ndim == 1:
        data = data.reshape((1, -1))
    l2_norm = np.linalg.norm(data, axis=1) + EPSILON
    return np.divide(data, np.expand_dims(l2_norm, axis=1))


def read_pretrained_embeddings(embeddings_path, vocabulary):
    """"Read pretrained word embeddings."""
    with open(embeddings_path, mode="r", encoding="utf-8") as embeddings_file:
        # First line is the number of words and the size (as in word2vec).
        _, size = embeddings_file.readline().split()
        pretrained_embeddings = np.random.uniform(-0.1, 0.1, (len(vocabulary), int(size))).astype(np.float32)
        counter = 0
        for line in embeddings_file:
            word, features = line.split(" ", 1)
            if word in vocabulary:
                word_id = vocabulary[word]
                pretrained_embeddings[word_id] = features.split()
                counter += 1
    print("Found {} out of {} words in vocabulary.".format(counter, len(vocabulary)))
    return pretrained_embeddings


def get_pretrained_embeddings(source_embeddings_path, target_embeddings_path,
                              source_vocabulary, target_vocabulary, normalize=True):
    """"Wrapper to read source and target pretrained word embeddings."""
    source_pretrained_embeddings = read_pretrained_embeddings(source_embeddings_path,
                                                              source_vocabulary)
    target_pretrained_embeddings = read_pretrained_embeddings(target_embeddings_path,
                                                              target_vocabulary)
    if normalize:
        pretrained_embeddings = np.vstack((source_pretrained_embeddings, target_pretrained_embeddings))
        normed_pretrained_embeddings = l2_normalize(pretrained_embeddings)
        source_pretrained_embeddings = normed_pretrained_embeddings[:len(source_pretrained_embeddings), :]
        target_pretrained_embeddings = normed_pretrained_embeddings[-len(target_pretrained_embeddings):, :]
    return source_pretrained_embeddings, target_pretrained_embeddings


def f1_score(precision, recall):
    """Calculate the F1 score."""
    return (2 * precision * recall) / (precision + recall + EPSILON)


def top_k(source, targets, k=1):
    """Get indices of the top k closest target vectors with respect
       to the source vector.
    """
    source = np.expand_dims(source, 0)
    cosine_sim = 1 - cdist(source, targets, metric="cosine")
    cosine_sim[np.isnan(cosine_sim)] = 0
    return np.argsort(np.squeeze(cosine_sim))[::-1][:k]


def restore_model(sess, checkpoint_dir):
    """Restore full meta graph of saved TensorFlow model."""
    meta_graph = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".meta"):
            meta_graph.append(os.path.join(checkpoint_dir, file))
    meta_graph.sort()
    if meta_graph:
        saver = tf.train.import_meta_graph(meta_graph[-1])
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))


def reset_graph():
    """Close unclosed sessions and reset graph."""
    if "sess" in globals() and sess:
        sess.close()
    tf.reset_default_graph()