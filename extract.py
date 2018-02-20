from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from itertools import product
from six.moves import xrange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

import utils


tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints and meta graph.")

tf.flags.DEFINE_string("extract_dir", "",
                       "Directory containing the aligned articles to do "
                       "parallel sentence extraction.")

tf.flags.DEFINE_string("source_vocab_path", "",
                       "Path to source language vocabulary.")

tf.flags.DEFINE_string("target_vocab_path", "",
                       "Path to target language vocabulary.")

tf.flags.DEFINE_string("source_output_path", "",
                       "Path to the file containing the extracted sentences in "
                       "the source language.")

tf.flags.DEFINE_string("target_output_path", "",
                       "Path to the file containing the extracted sentences in "
                       "the target language.")

tf.flags.DEFINE_string("score_output_path", "",
                       "Path to the file containing the probability scores of "
                       "the extracted sentence pairs.")

tf.flags.DEFINE_string("source_language", "",
                       "Source language suffix used as file extension.")

tf.flags.DEFINE_string("target_language", "",
                       "Target language suffix used as file extension.")

tf.flags.DEFINE_float("decision_threshold", 0.99,
                      "Decision threshold to predict a positive label.")

tf.flags.DEFINE_integer("batch_size", 500,
                        "Batch size to use during evaluation.")

tf.flags.DEFINE_integer("max_seq_length", 100,
                        "Maximum number of tokens per sentence.")

tf.flags.DEFINE_boolean("use_greedy", True,
                        "Use greedy post-treatment to force one-to-one "
                        "alignments.")


FLAGS = tf.flags.FLAGS


def read_articles(source_path, target_path):
    """Read the articles in source and target languages."""
    with open(source_path, mode="r", encoding="utf-8") as source_file,\
         open(target_path, mode="r", encoding="utf-8") as target_file:
            source_sentences = [l for l in source_file]
            target_sentences = [l for l in target_file]
    return source_sentences, target_sentences


def inference(sess, data_iterator, probs_op, placeholders):
    """Get the predicted class {0, 1} of given sentence pairs."""
    x_source, source_seq_length,\
    x_target, target_seq_length,\
    labels = placeholders

    num_iter = int(np.ceil(data_iterator.size / FLAGS.batch_size))
    probs = []
    for step in xrange(num_iter):
        source, target, label = data_iterator.next_batch(FLAGS.batch_size)
        source_len = utils.sequence_length(source)
        target_len = utils.sequence_length(target)

        feed_dict = {x_source: source,
                     x_target: target,
                     labels: label,
                     source_seq_length: source_len,
                     target_seq_length: target_len}

        batch_probs = sess.run(probs_op, feed_dict=feed_dict)
        probs.extend(batch_probs.tolist())
    probs = np.array(probs[:data_iterator.size])
    return probs


def extract_pairs(sess, source_sentences, target_sentences,
                  source_sentences_ids, target_sentences_ids,
                  probs_op, placeholders):
    """Extract sentence pairs from a pair of articles in source and target languages.
       Returns a list of (source sentence, target sentence, probability score) tuples.
    """
    pairs = [(i, j) for i, j in product(range(len(source_sentences)),
                                        range(len(target_sentences)))]

    data = [(source_sentences_ids[i], target_sentences_ids[j], 1.0)
            for i, j in product(range(len(source_sentences)),
                                range(len(target_sentences)))]

    data_iterator = utils.TestingIterator(np.array(data, dtype=object))

    y_score = inference(sess, data_iterator, probs_op, placeholders)
    y_score = [(score, k) for k, score in enumerate(y_score)]
    y_score.sort(reverse=True)

    i_aligned = set()
    j_aligned = set()
    sentence_pairs = []
    for score, k in y_score:
        i, j = pairs[k]
        if score < FLAGS.decision_threshold or i in i_aligned or j in j_aligned:
            continue
        if FLAGS.use_greedy:
            i_aligned.add(i)
            j_aligned.add(j)
        sentence_pairs.append((source_sentences[i], target_sentences[j], score))
    return sentence_pairs


def main(_):
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required."
    assert FLAGS.extract_dir, "--extract_dir is required."
    assert FLAGS.source_vocab_path, "--source_vocab_path is required."
    assert FLAGS.target_vocab_path, "--target_vocab_path is required."
    assert FLAGS.source_output_path, "--source_output_path is required."
    assert FLAGS.target_output_path, "--target_output_path is required."
    assert FLAGS.score_output_path, "--score_output_path is required."
    assert FLAGS.source_language, "--source_language is required."
    assert FLAGS.target_language, "--target_language is required."

    # Read vocabularies.
    source_vocab, _ = utils.initialize_vocabulary(FLAGS.source_vocab_path)
    target_vocab, _ = utils.initialize_vocabulary(FLAGS.target_vocab_path)

    # Read source and target paths for sentence extraction.
    source_paths = []
    target_paths = []
    for file in os.listdir(FLAGS.extract_dir):
        if file.endswith(FLAGS.source_language):
            source_paths.append(os.path.join(FLAGS.extract_dir, file))
        elif file.endswith(FLAGS.target_language):
            target_paths.append(os.path.join(FLAGS.extract_dir, file))
    source_paths.sort()
    target_paths.sort()

    utils.reset_graph()
    with tf.Session() as sess:
        # Restore saved model.
        utils.restore_model(sess, FLAGS.checkpoint_dir)

        # Recover placeholders and ops for extraction.
        x_source = sess.graph.get_tensor_by_name("x_source:0")
        source_seq_length = sess.graph.get_tensor_by_name("source_seq_length:0")

        x_target = sess.graph.get_tensor_by_name("x_target:0")
        target_seq_length = sess.graph.get_tensor_by_name("target_seq_length:0")

        labels = sess.graph.get_tensor_by_name("labels:0")

        placeholders = [x_source, source_seq_length, x_target, target_seq_length, labels]

        probs = sess.graph.get_tensor_by_name("feed_forward/output/probs:0")

        with open(FLAGS.source_output_path, mode="w", encoding="utf-8") as source_output_file,\
             open(FLAGS.target_output_path, mode="w", encoding="utf-8") as target_output_file,\
             open(FLAGS.score_output_path, mode="w", encoding="utf-8") as score_output_file:

            for source_path, target_path in zip(source_paths, target_paths):
                # Read sentences from articles.
                source_sentences, target_sentences = read_articles(source_path, target_path)

                # Convert sentences to token ids sequences.
                source_sentences_ids = [utils.sentence_to_token_ids(sent, source_vocab, FLAGS.max_seq_length)
                                        for sent in source_sentences]
                target_sentences_ids = [utils.sentence_to_token_ids(sent, target_vocab, FLAGS.max_seq_length)
                                        for sent in target_sentences]

                # Extract sentence pairs.
                pairs = extract_pairs(sess, source_sentences, target_sentences,
                                      source_sentences_ids, target_sentences_ids,
                                      probs, placeholders)
                if not pairs:
                    continue
                for source_sentence, target_sentence, score in pairs:
                    source_output_file.write(source_sentence)
                    target_output_file.write(target_sentence)
                    score_output_file.write(str(score) + "\n")


if __name__ == "__main__":
    tf.app.run()