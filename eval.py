from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

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

tf.flags.DEFINE_string("save_path", None,
                       "Path to file to write evaluation metrics.")


FLAGS = tf.flags.FLAGS


def inference(sess, data_iterator, probs_op, predicted_class_op, placeholders, threshold):
    """Get probability and predicted class of the examples in a test set."""
    x_source, source_seq_length,\
    x_target, target_seq_length,\
    labels = placeholders

    num_iter = int(np.ceil(data_iterator.size / FLAGS.batch_size))
    probs = []
    predicted_class = []
    for step in xrange(num_iter):
        source, target, label = data_iterator.next_batch(FLAGS.batch_size)
        source_len = utils.sequence_length(source)
        target_len = utils.sequence_length(target)

        feed_dict = {x_source: source, x_target: target, labels: label,
                     source_seq_length: source_len, target_seq_length: target_len}

        batch_probs, batch_predicted_class = sess.run([probs_op, predicted_class_op], feed_dict=feed_dict)
        probs.extend(batch_probs.tolist())
        predicted_class.extend(batch_predicted_class.tolist())
    probs = np.array(probs[:data_iterator.size])
    predicted_class = np.array(predicted_class[:data_iterator.size], dtype=np.int)
    return probs, predicted_class


def evaluate(sess, source_sentences, target_sentences, references,
             source_sentences_ids, target_sentences_ids,
             probs_op, predicted_class_op, placeholders):
    """"Evalute BiRNN at decision threshold value maximizing the area
        under the precison-recall curve.
    """
    pairs = [(i, j) for i, j in product(range(len(source_sentences)), range(len(target_sentences)))]

    data = [(source_sentences_ids[i], target_sentences_ids[j], 1.0) if (i, j) in references
            else (source_sentences_ids[i], target_sentences_ids[j], 0.0)
            for i, j in product(range(len(source_sentences)), range(len(target_sentences)))]

    data_iterator = EvalIterator(np.array(data, dtype=object))

    y_score, _ = inference(sess, data_iterator, probs_op, predicted_class_op,
                           placeholders, FLAGS.batch_size, 0.50)
    y_true = data_iterator.data[:, 2].astype(int)

    p, r, t = precision_recall_curve(y_true, y_score, pos_label=1)
    f1 = utils.f1_score(p, r)

    i = np.argmax(f1)
    p_star, r_star, f1_star, t_star = p[i], r[i], f1[i], t[i]

    print("Evaluation metrics if decision threshold = {:.4f}\n"
          "Precision = {:.2f}, Recall = {:.2f}, F1 = {:.2f}\n"
          "-------------------------------------------------"
          .format(t_star, 100*p_star, 100*r_star, 100*f1_star))

    if FLAGS.save_path:
        np.savez(save_path, precision_values=p, recall_values=r,
                 f1_values=f1, threshold_values=t,
                 precision_best=p_star , recall_best=r_star,
                 f1_best=f1_star, threshold_best=t_star)


def read_data_with_ref(source_path, target_path, ref_path):
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
    source_sentences, target_sentences, references = read_data_with_ref(
        FLAGS.source_test_path,
        FLAGS.target_test_path,
        FLAGS.reference_test_path)

    # Convert sentences to token ids sequences.
    source_sentences_ids = [utils.sentence_to_token_ids(sent, source_vocab, 100) for sent in source_sentences]
    target_sentences_ids = [utils.sentence_to_token_ids(sent, target_vocab, 100) for sent in target_sentences]

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
        predicted_class = sess.graph.get_tensor_by_name("feed_forward/output//predicted_class:0")

        # Run evaluation.
        evaluate_birnn_optimal(sess,
                               source_sentences, target_sentences, references,
                               source_sentences_ids, target_sentences_ids,
                               probs, predicted_class, placeholders)


if __name__ == "__main__":
    tf.app.run()