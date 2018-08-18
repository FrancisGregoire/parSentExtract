from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import get_pretrained_embeddings, reset_graph


class Config(object):
    """Utility class to store BiRNN hyperparameters."""
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 embedding_size,
                 state_size,
                 hidden_size,
                 num_layers,
                 learning_rate,
                 max_gradient_norm=5.0,
                 use_lstm=False,
                 use_mean_pooling=False,
                 use_max_pooling=False,
                 source_embeddings_path=None,
                 target_embeddings_path=None,
                 fix_pretrained=False):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.use_lstm = use_lstm
        self.use_mean_pooling = use_mean_pooling
        self.use_max_pooling = use_max_pooling
        self.source_embeddings_path = source_embeddings_path
        self.target_embeddings_path = target_embeddings_path
        self.fix_pretrained = fix_pretrained


class BiRNN(object):
    """BiRNN TensorFlow implementation based on https://arxiv.org/abs/1709.09783
       A Deep Neural Network Approach To Parallel Sentence Extraction
          Francis Grégoire, Philippe Langlais.
    """
    def __init__(self, config):
        self.config = config

    def update_config(self, config):
        self.config = config

    def build_graph(self):
        # Reset previous graph.
        reset_graph()

        # Placeholders.
        x_source = tf.placeholder(tf.int32,
                                  shape=[None, None],
                                  name="x_source")

        source_seq_length = tf.placeholder(tf.int32,
                                           shape=[None],
                                           name="source_seq_length")

        x_target = tf.placeholder(tf.int32,
                                  shape=[None, None],
                                  name="x_target")

        target_seq_length = tf.placeholder(tf.int32,
                                           shape=[None],
                                           name="target_seq_length")

        labels = tf.placeholder(tf.float32,
                                shape=[None],
                                name="labels")

        input_dropout = tf.placeholder_with_default(1.0,
                                                    shape=[],
                                                    name="input_dropout")

        output_dropout = tf.placeholder_with_default(1.0,
                                                     shape=[],
                                                     name="output_dropout")

        decision_threshold = tf.placeholder_with_default(0.5,
                                                         shape=[],
                                                         name="decision_threshold")

        # Embedding layer.
        with tf.variable_scope("embeddings"):
            if self.config.source_embeddings_path is not None and self.config.target_embeddings_path is not None:
                source_pretrained_embeddings,\
                target_pretrained_embeddings = get_pretrained_embeddings(
                    source_embeddings_path,
                    target_embeddings_path,
                    source_vocab,
                    target_vocab)
                assert source_pretrained_embeddings.shape[1] == target_pretrained_embeddings.shape[1]
                self.config.embedding_size = source_pretrained_embeddings.shape[1]
                if self.config.fix_pretrained:
                    source_embeddings = tf.get_variable(
                        name="source_embeddings_matrix",
                        shape=[self.config.source_vocab_size, self.config.embedding_size],
                        initializer=tf.constant_initializer(source_pretrained_embeddings),
                        trainable=False)
                    target_embeddings = tf.get_variable(
                        name="target_embeddings_matrix",
                        shape=[self.config.target_vocab_size, self.config.embedding_size],
                        initializer=tf.constant_initializer(target_pretrained_embeddings),
                        trainable=False)
                else:
                    source_embeddings = tf.get_variable(
                        name="source_embeddings_matrix",
                        shape=[self.config.source_vocab_size, self.config.embedding_size],
                        initializer=tf.constant_initializer(source_pretrained_embeddings))
                    target_embeddings = tf.get_variable(
                        name="target_embeddings_matrix",
                        shape=[self.config.target_vocab_size, self.config.embedding_size],
                        initializer=tf.constant_initializer(target_pretrained_embeddings))
            else:
                source_embeddings = tf.get_variable(
                    name="source_embeddings_matrix",
                    shape=[self.config.source_vocab_size, self.config.embedding_size])
                target_embeddings = tf.get_variable(
                    name="target_embeddings_matrix",
                    shape=[self.config.target_vocab_size, self.config.embedding_size])

            source_rnn_inputs = tf.nn.embedding_lookup(source_embeddings, x_source)
            target_rnn_inputs = tf.nn.embedding_lookup(target_embeddings, x_target)
            source_rnn_inputs = tf.nn.dropout(source_rnn_inputs,
                                              keep_prob=input_dropout,
                                              name="source_seq_embeddings")
            target_rnn_inputs = tf.nn.dropout(target_rnn_inputs,
                                              keep_prob=input_dropout,
                                              name="target_seq_embeddings")

        # BiRNN encoder.
        with tf.variable_scope("birnn") as scope:
            if self.config.use_lstm:
                cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.state_size, use_peepholes=True)
                cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.state_size, use_peepholes=True)
            else:
                cell_fw = tf.nn.rnn_cell.GRUCell(self.config.state_size)
                cell_bw = tf.nn.rnn_cell.GRUCell(self.config.state_size)

            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=output_dropout)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=output_dropout)

            if self.config.num_layers > 1:
                if self.config.use_lstm:
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.config.state_size,
                                                                                   use_peepholes=True)
                                                           for _ in range(self.config.num_layers)])
                    cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.config.state_size,
                                                                                   use_peepholes=True)
                                                           for _ in range(self.config.num_layers)])
                else:
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(self.config.state_size)
                                                           for _ in range(self.config.num_layers)])
                    cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(self.config.state_size)
                                                           for _ in range(self.config.num_layers)])

            with tf.variable_scope(scope):
                source_rnn_outputs, source_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=source_rnn_inputs,
                    sequence_length=source_seq_length,
                    dtype=tf.float32)

            with tf.variable_scope(scope, reuse=True):
                target_rnn_outputs, target_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=target_rnn_inputs,
                    sequence_length=target_seq_length,
                    dtype=tf.float32)

            self.config.state_size *= 2
            # Mean and max pooling only work for 1 layer BiRNN.
            if self.config.use_mean_pooling:
                source_final_state = self.mean_pooling(source_rnn_outputs, source_seq_length)
                target_final_state = self.mean_pooling(target_rnn_outputs, target_seq_length)
            elif self.config.use_max_pooling:
                source_final_state = self.max_pooling(source_rnn_outputs)
                target_final_state = self.max_pooling(target_rnn_outputs)
            else:
                source_final_state_fw, source_final_state_bw = source_final_state
                target_final_state_fw, target_final_state_bw = target_final_state
                if self.config.num_layers > 1:
                    source_final_state_fw = source_final_state_fw[-1]
                    source_final_state_bw = source_final_state_bw[-1]
                    target_final_state_fw = target_final_state_fw[-1]
                    target_final_state_bw = target_final_state_bw[-1]
                if self.config.use_lstm:
                    source_final_state_fw = source_final_state_fw.h
                    source_final_state_bw = source_final_state_bw.h
                    target_final_state_fw = target_final_state_fw.h
                    target_final_state_bw = target_final_state_bw.h
                source_final_state = tf.concat([source_final_state_fw, source_final_state_bw],
                                               axis=1)
                target_final_state = tf.concat([target_final_state_fw, target_final_state_bw],
                                               axis=1)

        # Feed-forward neural network.
        with tf.variable_scope("feed_forward"):
            h_multiply = tf.multiply(source_final_state, target_final_state)
            h_abs_diff = tf.abs(tf.subtract(source_final_state, target_final_state))

            W_1 = tf.get_variable(name="W_1",
                                  shape=[self.config.state_size, self.config.hidden_size])
            W_2 = tf.get_variable(name="W_2",
                                  shape=[self.config.state_size, self.config.hidden_size])
            b_1 = tf.get_variable(name="b_1",
                                  shape=[self.config.hidden_size],
                                  initializer=tf.constant_initializer(0.0))

            h_semantic = tf.tanh(tf.matmul(h_multiply, W_1) + tf.matmul(h_abs_diff, W_2) + b_1)

            W_3 = tf.get_variable(name="W_3",
                                  shape=[self.config.hidden_size, 1])
            b_2 = tf.get_variable(name="b_2",
                                  shape=[1],
                                  initializer=tf.constant_initializer(0.0))

            logits = tf.matmul(h_semantic, W_3) + b_2
            logits = tf.squeeze(logits,
                                name="logits")

            # Sigmoid output layer.
            with tf.name_scope("output"):
                probs = tf.sigmoid(logits,
                                   name="probs")
                predicted_class = tf.cast(tf.greater(probs, decision_threshold),
                                          tf.float32,
                                          name="predicted_class")

        # Loss.
        with tf.name_scope("cross_entropy"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=labels,
                name="cross_entropy_per_sequence")
            mean_loss = tf.reduce_mean(losses,
                                       name="cross_entropy_loss")

        # Optimization.
        with tf.name_scope("optimization"):
            global_step = tf.Variable(initial_value=0,
                                      trainable=False,
                                      name="global_step")
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            trainable_variables = tf.trainable_variables()
            gradients = tf.gradients(mean_loss, trainable_variables,
                                     name="gradients")
            clipped_gradients, global_norm = tf.clip_by_global_norm(
                gradients,
                clip_norm=self.config.max_gradient_norm,
                name="clipped_gradients")
            train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_variables),
                                                 global_step=global_step)

        # Evaluation metrics.
        accuracy = tf.metrics.accuracy(labels, predicted_class,
                                       name="accuracy")
        precision = tf.metrics.precision(labels, predicted_class,
                                         name="precision")
        recall = tf.metrics.recall(labels, predicted_class,
                                   name="recall")

        # Add summaries.
        tf.summary.scalar("loss", mean_loss)
        tf.summary.scalar("global_norm", global_norm)
        tf.summary.scalar("accuracy", accuracy[0])
        tf.summary.scalar("precision", precision[0])
        tf.summary.scalar("recall", recall[0])
        tf.summary.scalar("logits" + "/sparsity", tf.nn.zero_fraction(logits))
        tf.summary.histogram("logits" + "/activations", logits)
        tf.summary.histogram("probs", probs)

        # Add histogram for trainable variables.
        for var in trainable_variables:
            tf.summary.histogram(var.op.name, var)

        # Add histogram for gradients.
        for grad, var in zip(clipped_gradients, trainable_variables):
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradients", grad)

        # Assign placeholders and operations.
        self.x_source = x_source
        self.x_target = x_target
        self.source_seq_length = source_seq_length
        self.target_seq_length = target_seq_length
        self.labels = labels
        self.input_dropout = input_dropout
        self.output_dropout = output_dropout
        self.decision_threshold = decision_threshold
        self.train_op = train_op
        self.probs = probs
        self.predicted_class = predicted_class
        self.mean_loss = mean_loss
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def mean_pooling(rnn_outputs, seq_length):
        """Use mean pooling to obtain final sentence representation."""
        sum_rnn_outputs = tf.reduce_sum(tf.concat(rnn_outputs, axis=2), axis=1)
        seq_length = tf.expand_dims(tf.cast(seq_length, tf.float32), axis=1)
        return tf.divide(sum_rnn_outputs, seq_length)

    def max_pooling(rnn_outputs):
        """Use max pooling to obtain final sentence representation."""
        return tf.reduce_max(tf.concat(rnn_outputs, axis=2), axis=1)

    def restore_variables(self, sess, checkpoint_dir):
        """Restore previously saved trainable variable weights from the
           last checkpoint found in checkpoint_dir.
        """
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            raise ValueError("Can't load save path from checkpoint_dir.")
