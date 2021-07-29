# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-9 下午9:04
@IDE    :PyCharm
@document   :data_process.py
"""
import tensorflow as tf
from bert import modeling, optimization


class LoadBertModel(object):
    # bert层相当于提取上层特征

    def __init__(self, config_path, init_checkpoint, input_ids, input_mask, segment_ids, is_training):
        model = modeling.BertModel(config=modeling.BertConfig.from_json_file(config_path),
                                   is_training=is_training,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=False,
                                   # scope="bert"    # TODO
                                   )

        layer_logits = []
        for i, layer in enumerate(model.all_encoder_layers):
            layer_feature = tf.layers.dense(layer,
                                            1,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            name="layer_logit%d" % i)
            # layer_feature = tf.nn.relu(layer_feature)  # TODO tanh
            layer_logits.append(layer_feature)
        layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
        layer_dist = tf.nn.softmax(layer_logits)
        seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
        pooled_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
        pooled_output = tf.squeeze(pooled_output, axis=2)
        pooled_layer = pooled_output

        self.tvars = tf.trainable_variables()
        (self.assignment_map, _) = modeling.get_assignment_map_from_checkpoint(self.tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, self.assignment_map)
        # ===================
        # 分类采用     model.get_pooled_output()
        # 序列标注采用  model.get_sequence_output()
        # 多层融合采用　pooled_output
        # self.output_last2 = model.get_all_encoder_layers()[-2]
        self.get_sequence_output = pooled_layer
        # self.get_sequence_output = model.get_sequence_output()


class MyModel(object):

    def __init__(self,
                 bert_config,
                 init_checkpoint,
                 num_labels,
                 use_lstm,
                 use_crf,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps
                 ):

        self.inputs_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_seq")  # B * (S+2)
        self.inputs_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_mask")  # B * (S+2)
        self.inputs_segment = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_segment")  # B * (S+2)
        self.outputs_seq = tf.placeholder(shape=[None, None], dtype=tf.int32, name='outputs_seq')  # B * (S+2)
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)

        inputs_seq_len = tf.reduce_sum(self.inputs_mask, axis=-1)  # B

        bert_model = LoadBertModel(bert_config,
                                   init_checkpoint,
                                   self.inputs_ids,
                                   self.inputs_mask,
                                   self.inputs_segment,
                                   self.is_training
                                   )

        bert_outputs = bert_model.get_sequence_output  # B * (S+2) * D

        if not use_lstm:
            hiddens = bert_outputs
        else:
            bert_outputs = tf.nn.dropout(bert_outputs, keep_prob=self.keep_prob, seed=12345)

            with tf.variable_scope('bilstm'):
                cell_fw = tf.nn.rnn_cell.LSTMCell(256)
                cell_bw = tf.nn.rnn_cell.LSTMCell(256)
                ((rnn_fw_outputs, rnn_bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=bert_outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32
                )
                rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs)  # B * (S+2) * D
            hiddens = rnn_outputs

        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(hiddens, num_labels)  # B * (S+2) * V
            probs_seq = tf.nn.softmax(logits_seq, name='probs_seq')

            if not use_crf:
                preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq")  # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(
                    logits_seq, self.outputs_seq, inputs_seq_len)
                preds_seq, crf_scores = tf.contrib.crf.crf_decode(
                    logits_seq, transition_matrix, inputs_seq_len)

        self.outputs = preds_seq

        with tf.variable_scope('loss'):
            if not use_crf:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits_seq, labels=self.outputs_seq)  # B * S
                masks = tf.sequence_mask(inputs_seq_len, dtype=tf.float32)  # B * S
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(inputs_seq_len, tf.float32)  # B
            else:
                loss = -log_likelihood / tf.cast(inputs_seq_len, tf.float32)  # B
            equals = tf.reduce_sum(tf.cast(tf.equal(tf.cast(preds_seq, tf.int32), self.outputs_seq),
                                   tf.float32) * tf.cast(self.inputs_mask, tf.float32))
            acc = equals / tf.cast(tf.reduce_sum(self.inputs_mask), tf.float32)
        self.loss = tf.reduce_mean(loss)
        self.acc = acc

        with tf.variable_scope('opt'):
            params_of_bert = []
            params_of_other = []
            for var in tf.trainable_variables():
                vname = var.name
                if vname.startswith("bert"):
                    params_of_bert.append(var)
                else:
                    params_of_other.append(var)
            opt1 = tf.train.AdamOptimizer(learning_rate)
            opt2 = tf.train.AdamOptimizer(1e-3)
            # 实现　loss 对　相关参数的倒数
            gradients_bert = tf.gradients(loss, params_of_bert)
            gradients_other = tf.gradients(loss, params_of_other)
            # 梯度裁剪
            gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
            gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
            train_op_bert = opt1.apply_gradients(zip(gradients_bert_clipped, params_of_bert))
            train_op_other = opt2.apply_gradients(zip(gradients_other_clipped, params_of_other))

        if use_crf or use_lstm:
            self.train_op = (train_op_bert, train_op_other)
        else:
            self.train_op = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps, False)
