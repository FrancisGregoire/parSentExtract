from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from six.moves import xrange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

import utils
from model import Config, BiRNN


tf.flags.DEFINE_string("source_train_path", "",
                       "Path to the file containing the source sentences to "
                       "train the model.")

tf.flags.DEFINE_string("target_train_path", "",
                       "Path to the file containing the target sentences to "
                       "train the model.")

tf.flags.DEFINE_string("source_valid_path", "",
                       "Path to the file containing the source sentences to "
                       "evaluate the model.")

tf.flags.DEFINE_string("target_valid_path", "",
                       "Path to the file containing the target sentences to "
                       "evaluate the model.")

tf.flags.DEFINE_string("checkpoint_dir", "./tflogs",
                       "Directory to save checkpoints and summaries of the model.")

tf.flags.DEFINE_integer("source_vocab_size", 100000,
                        "Number of the most frequent words to keep in the source "
                        "vocabulary.")

tf.flags.DEFINE_integer("target_vocab_size", 100000,
                        "Number of the most frequent words to keep in target "
                        "vocabulary.")

tf.flags.DEFINE_float("learning_rate", 2e-4,
                      "Learning rate.")

tf.flags.DEFINE_float("max_gradient_norm", 5.0,
                      "Clip gradient to this norm.")

tf.flags.DEFINE_float("decision_threshold", 0.99,
                      "Decision threshold to predict a positive label.")

tf.flags.DEFINE_integer("embedding_size", 300,
                        "Size of each word embedding.")

tf.flags.DEFINE_integer("state_size", 300,
                        "Size of the recurrent state in the BiRNN encoder.")

tf.flags.DEFINE_integer("hidden_size", 128,
                        "Size of the hidden layer in the feed-forward neural "
                        "network.")

tf.flags.DEFINE_integer("num_layers", 1,
                        "Number of layers in the BiRNN encoder.")

tf.flags.DEFINE_string("source_embeddings_path", None,
                       "Pretrained embeddings to initialize the source embeddings "
                       "matrix.")

tf.flags.DEFINE_string("target_embeddings_path", None,
                       "Pretrained embeddings to initialize the target embeddings "
                       "matrix.")

tf.flags.DEFINE_boolean("fix_pretrained", False,
                        "If true fix pretrained embeddings.")

tf.flags.DEFINE_boolean("use_lstm", False,
                        "If true use LSTM cells. Otherwise use GRU cells.")

tf.flags.DEFINE_boolean("use_mean_pooling", False,
                        "If true use mean pooling for final sentence representation.")

tf.flags.DEFINE_boolean("use_max_pooling", False,
                        "If true use max pooling for final sentence representation.")

tf.flags.DEFINE_integer("batch_size", 128,
                        "Batch size to use during training.")

tf.flags.DEFINE_integer("num_epochs", 15,
                        "Number of epochs to train the model.")

tf.flags.DEFINE_integer("num_negative", 5,
                        "Number of negative examples to sample per pair of "
                        "parallel sentences in training dataset.")

tf.flags.DEFINE_float("keep_prob_input", 0.8,
                      "Keep probability for dropout applied at the embedding layer.")

tf.flags.DEFINE_float("keep_prob_output", 0.7,
                      "Keep probability for dropout applied at the prediction layer.")

tf.flags.DEFINE_integer("steps_per_checkpoint", 200,
                        "Number of steps to save a model checkpoint.")


FLAGS = tf.flags.FLAGS


def eval_epoch(sess, model, data_iterator, summary_writer):
    """Evaluate model for one epoch."""
    sess.run(tf.local_variables_initializer())
    num_iter = int(np.ceil(data_iterator.size / FLAGS.batch_size))
    epoch_loss = 0
    for step in xrange(num_iter):
        source, target, label = data_iterator.next_batch(FLAGS.batch_size)
        source_len = utils.sequence_length(source)
        target_len = utils.sequence_length(target)
        feed_dict = {model.x_source: source,
                     model.x_target: target,
                     model.labels: label,
                     model.source_seq_length: source_len,
                     model.target_seq_length: target_len,
                     model.decision_threshold: FLAGS.decision_threshold}
        loss_value, epoch_accuracy,\
        epoch_precision, epoch_recall = sess.run([model.mean_loss,
                                                  model.accuracy[1],
                                                  model.precision[1],
                                                  model.recall[1]],
                                                  feed_dict=feed_dict)
        epoch_loss += loss_value
        if step % FLAGS.steps_per_checkpoint == 0:
            summary = sess.run(model.summaries, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=data_iterator.global_step)
    epoch_loss /= step
    epoch_f1 = utils.f1_score(epoch_precision, epoch_recall)
    print("  Testing:  Loss = {:.6f}, Accuracy = {:.4f}, "
          "Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}"
          .format(epoch_loss, epoch_accuracy,
                  epoch_precision, epoch_recall, epoch_f1))


def main(_):
    assert FLAGS.source_train_path, ("--source_train_path is required.")
    assert FLAGS.target_train_path, ("--target_train_path is required.")

    # Create vocabularies.
    source_vocab_path = os.path.join(os.path.dirname(FLAGS.source_train_path),
                                     "vocabulary.source")
    target_vocab_path = os.path.join(os.path.dirname(FLAGS.source_train_path),
                                     "vocabulary.target")
    utils.create_vocabulary(source_vocab_path, FLAGS.source_train_path, FLAGS.source_vocab_size)
    utils.create_vocabulary(target_vocab_path, FLAGS.target_train_path, FLAGS.target_vocab_size)

    # Read vocabularies.
    source_vocab, rev_source_vocab = utils.initialize_vocabulary(source_vocab_path)
    target_vocab, rev_target_vocab = utils.initialize_vocabulary(target_vocab_path)

    # Read parallel sentences.
    parallel_data = utils.read_data(FLAGS.source_train_path, FLAGS.target_train_path,
                                    source_vocab, target_vocab)

    # Read validation data set.
    if FLAGS.source_valid_path and FLAGS.target_valid_path:
        valid_data = utils.read_data(FLAGS.source_valid_path, FLAGS.target_valid_path,
                                    source_vocab, target_vocab)

    # Initialize BiRNN.
    config = Config(len(source_vocab),
                    len(target_vocab),
                    FLAGS.embedding_size,
                    FLAGS.state_size,
                    FLAGS.hidden_size,
                    FLAGS.num_layers,
                    FLAGS.learning_rate,
                    FLAGS.max_gradient_norm,
                    FLAGS.use_lstm,
                    FLAGS.use_mean_pooling,
                    FLAGS.use_max_pooling,
                    FLAGS.source_embeddings_path,
                    FLAGS.target_embeddings_path,
                    FLAGS.fix_pretrained)

    model = BiRNN(config)

    # Build graph.
    model.build_graph()

    # Train  model.
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_iterator = utils.TrainingIterator(parallel_data, FLAGS.num_negative)
        train_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "train"), sess.graph)

        if FLAGS.source_valid_path and FLAGS.target_valid_path:
            valid_iterator = utils.EvalIterator(valid_data)
            valid_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "valid"), sess.graph)

        epoch_loss = 0
        epoch_completed = 0
        batch_completed = 0

        num_iter = int(np.ceil(train_iterator.size / FLAGS.batch_size * FLAGS.num_epochs))
        start_time = time.time()
        print("Training model on {} sentence pairs per epoch:".
            format(train_iterator.size, valid_iterator.size))

        for step in xrange(num_iter):
            source, target, label = train_iterator.next_batch(FLAGS.batch_size)
            source_len = utils.sequence_length(source)
            target_len = utils.sequence_length(target)
            feed_dict = {model.x_source: source,
                         model.x_target: target,
                         model.labels: label,
                         model.source_seq_length: source_len,
                         model.target_seq_length: target_len,
                         model.input_dropout: FLAGS.keep_prob_input,
                         model.output_dropout: FLAGS.keep_prob_output,
                         model.decision_threshold: FLAGS.decision_threshold}

            _, loss_value, epoch_accuracy,\
            epoch_precision, epoch_recall = sess.run([model.train_op,
                                                      model.mean_loss,
                                                      model.accuracy[1],
                                                      model.precision[1],
                                                      model.recall[1]],
                                                      feed_dict=feed_dict)
            epoch_loss += loss_value
            batch_completed += 1
            # Write the model's training summaries.
            if step % FLAGS.steps_per_checkpoint == 0:
                summary = sess.run(model.summaries, feed_dict=feed_dict)
                train_summary_writer.add_summary(summary, global_step=step)
            # End of current epoch.
            if train_iterator.epoch_completed > epoch_completed:
                epoch_time = time.time() - start_time
                epoch_loss /= batch_completed
                epoch_f1 = utils.f1_score(epoch_precision, epoch_recall)
                epoch_completed += 1
                print("Epoch {} in {:.0f} sec\n"
                      "  Training: Loss = {:.6f}, Accuracy = {:.4f}, "
                      "Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}"
                      .format(epoch_completed, epoch_time,
                              epoch_loss, epoch_accuracy,
                              epoch_precision, epoch_recall, epoch_f1))
                # Save a model checkpoint.
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=step)
                # Evaluate model on the validation set.
                if FLAGS.source_valid_path and FLAGS.target_valid_path:
                    eval_epoch(sess, model, valid_iterator, valid_summary_writer)
                # Initialize local variables for new epoch.
                batch_completed = 0
                epoch_loss = 0
                sess.run(tf.local_variables_initializer())
                start_time = time.time()

        print("Training done with {} steps.".format(num_iter))
        train_summary_writer.close()
        valid_summary_writer.close()


if __name__ == "__main__":
    tf.app.run()
