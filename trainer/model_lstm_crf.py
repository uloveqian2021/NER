# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  :
@time   :20-12-9 下午9:04
@IDE    :PyCharm
@document   :data_process.py
"""
import tensorflow as tf


class MyModel(object):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size_char,
                 vocab_size_bio,
                 use_crf):

        self.inputs_ids = tf.placeholder(tf.int32, [None, None], name="inputs_seq")
        self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len")
        self.outputs_seq = tf.placeholder(tf.int32, [None, None], name='outputs_seq')
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

        with tf.variable_scope('embedding_layer'):
            embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size_char, embedding_dim], dtype=tf.float32)
            embedded = tf.nn.embedding_lookup(embedding_matrix, self.inputs_ids)

        with tf.variable_scope('encoder'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            ((rnn_fw_outputs, rnn_bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=embedded,
                sequence_length=self.inputs_seq_len,
                dtype=tf.float32
            )
            rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs)  # B * S1 * D

        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(rnn_outputs, vocab_size_bio,)  # B * S * V
            probs_seq = tf.nn.softmax(logits_seq)

            if not use_crf:
                preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq")  # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq, self.inputs_seq_len)
                preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, self.inputs_seq_len)

        self.outputs = preds_seq

        with tf.variable_scope('loss'):
            if not use_crf:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.outputs_seq)  # B * S
                masks = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32)  # B * S
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(self.inputs_seq_len, tf.float32)  # B
            else:
                loss = -log_likelihood / tf.cast(self.inputs_seq_len, tf.float32)  # B
            masks = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32)  # B * S
            equals = tf.reduce_sum(tf.cast(tf.equal(tf.cast(preds_seq, tf.int32), self.outputs_seq),
                                   tf.float32) * tf.cast(masks, tf.float32))
            acc = equals / tf.cast(tf.reduce_sum(masks), tf.float32)
        self.loss = tf.reduce_mean(loss)
        self.acc = acc

        with tf.variable_scope('opt'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
