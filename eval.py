from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from itertools import product
from six.moves import xrange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_recall_curve

import utils


tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints and meta graph.")

tf.flags.DEFINE_string("source_test_path", "",
                       "Path to the file containing the source sentences to "
                       "test the model.")

tf.flags.DEFINE_string("target_test_path", "",
                       "Path to the file containing the target sentences to "
                       "test the model.")

tf.flags.DEFINE_string("reference_test_path", "",
                       "Path to the file containing the references to "
                       "test the model.")

tf.flags.DEFINE_string("source_vocab_path", "",
                       "Path to source language vocabulary.")

tf.flags.DEFINE_string("target_vocab_path", "",
                       "Path to target language vocabulary.")

tf.flags.DEFINE_integer("batch_size", 500,
                        "Batch size to use during evaluation.")

tf.flags.DEFINE_integer("max_seq_length", 100,
                        "Maximum number of tokens per sentence.")


FLAGS = tf.flags.FLAGS


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


def evaluate(sess, source_sentences, target_sentences, references,
             source_sentences_ids, target_sentences_ids,
             probs_op, placeholders):
    """"Evalute BiRNN at decision threshold value maximizing the area
        under the precison-recall curve.
    """
    pairs = [(i, j) for i, j in product(range(len(source_sentences)),
                                        range(len(target_sentences)))]

    data = [(source_sentences_ids[i], target_sentences_ids[j], 1.0) if (i, j) in references
            else (source_sentences_ids[i], target_sentences_ids[j], 0.0)
            for i, j in product(range(len(source_sentences)),
                                range(len(target_sentences)))]

    data_iterator = utils.TestingIterator(np.array(data, dtype=object))

    y_score = inference(sess, data_iterator, probs_op, placeholders)
    y_true = data_iterator.data[:, 2].astype(int)

    p, r, t = precision_recall_curve(y_true, y_score, pos_label=1)
    f1 = utils.f1_score(p, r)

    i = np.argmax(f1)
    print("Evaluation metrics at decision threshold = {:.4f}\n"
          "Precision = {:.2f}, Recall = {:.2f}, F1 = {:.2f}\n"
          "-------------------------------------------------"
          .format(p[i], 100*r[i], 100*f1[i], 100*t[i]))


def main(_):
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required."
    assert FLAGS.source_test_path, "--source_test_path is required."
    assert FLAGS.target_test_path, "--target_test_path is required."
    assert FLAGS.reference_test_path, "--reference_test_path is required."
    assert FLAGS.source_vocab_path, "--souce_vocab_path is required."
    assert FLAGS.target_vocab_path, "--target_vocab_path is required."

    # Read vocabularies.
    source_vocab, _ = utils.initialize_vocabulary(FLAGS.source_vocab_path)
    target_vocab, _ = utils.initialize_vocabulary(FLAGS.target_vocab_path)

    # Read test set.
    source_sentences, target_sentences, references = utils.read_data_with_ref(
        FLAGS.source_test_path,
        FLAGS.target_test_path,
        FLAGS.reference_test_path)

    # Convert sentences to token ids sequences.
    source_sentences_ids = [utils.sentence_to_token_ids(sent, source_vocab, FLAGS.max_seq_length)
                            for sent in source_sentences]
    target_sentences_ids = [utils.sentence_to_token_ids(sent, target_vocab, FLAGS.max_seq_length)
                            for sent in target_sentences]

    utils.reset_graph()
    with tf.Session() as sess:
        # Restore saved model.
        utils.restore_model(sess, FLAGS.checkpoint_dir)

        # Recover placeholders and ops for evaluation.
        x_source = sess.graph.get_tensor_by_name("x_source:0")
        source_seq_length = sess.graph.get_tensor_by_name("source_seq_length:0")

        x_target = sess.graph.get_tensor_by_name("x_target:0")
        target_seq_length = sess.graph.get_tensor_by_name("target_seq_length:0")

        labels = sess.graph.get_tensor_by_name("labels:0")

        placeholders = [x_source, source_seq_length, x_target, target_seq_length, labels]

        probs = sess.graph.get_tensor_by_name("feed_forward/output/probs:0")

        # Run evaluation.
        evaluate(sess,
                 source_sentences, target_sentences, references,
                 source_sentences_ids, target_sentences_ids,
                 probs, placeholders)


if __name__ == "__main__":
    tf.app.run()