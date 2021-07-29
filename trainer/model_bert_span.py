# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-8 下午4:10
@IDE    :PyCharm
@document   :model_bert_span.py
"""
import tensorflow as tf
from bert import modeling, optimization
import logging as log

log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO)
log.info("start training!")


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
        log.info("model params:")
        params_num_all = 0
        for variable in tf.trainable_variables():
            params_num = 1
            for dim in variable.shape:
                params_num *= dim
            params_num_all += params_num
            log.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
        log.info("all params num: " + str(params_num_all))

        layer_logits = []
        for i, layer in enumerate(model.all_encoder_layers):
            layer_feature = tf.layers.dense(layer, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            name="layer_logit%d" % i)
            # layer_feature = tf.nn.relu(layer_feature)  # TODO tanh
            layer_logits.append(layer_feature)
        layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
        layer_dist = tf.nn.softmax(layer_logits)
        seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
        pooled_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
        pooled_output = tf.squeeze(pooled_output, axis=2)
        pooled_layer = pooled_output
        # char_bert_outputs = pooled_laRERyer[:, 1: max_seq_length - 1, :]  # [batch_size, seq_length, embedding_size]
        # char_bert_outputs = pooled_layer

        self.tvars = tf.trainable_variables()
        (self.assignment_map, _) = modeling.get_assignment_map_from_checkpoint(self.tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, self.assignment_map)
        # ===================
        # 分类采用     model.get_pooled_output()
        # 序列标注采用  model.get_sequence_output()
        # 多层融合采用　pooled_output
        # self.output_last2 = model.get_all_encoder_layers()[-2]
        self.get_sequence_output = pooled_layer


class MyModel(object):

    def __init__(self,
                 bert_config,
                 init_checkpoint,
                 num_labels,
                 use_lstm,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps
                 ):
        self.num_labels = num_labels
        self.inputs_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_seq")  # B * (S+2)
        self.inputs_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_mask")  # B * (S+2)
        self.inputs_segment = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_segment")  # B * (S+2)
        # O的标签
        self.tag_o1_inputs = tf.placeholder(shape=[None, None, None], dtype=tf.float32, name='tag_o1_label')  # (32, 164, 49)
        self.tag_o2_inputs = tf.placeholder(shape=[None, None, None], dtype=tf.float32, name='tag_o2_label')  # (32, 164, 49)

        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

        inputs_seq_len = tf.reduce_sum(self.inputs_mask, axis=-1)  # B

        bert_model = LoadBertModel(
            bert_config,
            init_checkpoint,
            self.inputs_ids,
            self.inputs_mask,
            self.inputs_segment,
            self.is_training,
        )
        # batch_size, length of sequence, diw
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

        with tf.variable_scope('dense'):

            # df = tf.cond(self.is_training, lambda: True, lambda: False)
            if use_lstm:
                # I.e., 0.1 dropout
                hiddens = tf.nn.dropout(hiddens, keep_prob=self.keep_prob, seed=12345)

            self.logit_o1 = tf.layers.dense(inputs=hiddens, units=self.num_labels)  # 全连接层
            self.logit_o2 = tf.layers.dense(inputs=hiddens, units=self.num_labels)  # 全连接层
            prob_o1 = tf.nn.sigmoid(self.logit_o1, name='score_o1')  # sigmoid 激活函数
            prob_o2 = tf.nn.sigmoid(self.logit_o2, name='score_o2')  # sigmoid 激活函数
            print('self.logit_o1', self.logit_o1)
            self.prob_o1 = prob_o1
            self.prob_o2 = prob_o2
            selection_loss_o1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_o1_inputs, logits=self.logit_o1)
            selection_loss_o2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_o2_inputs, logits=self.logit_o2)

        with tf.variable_scope('loss'):
            mask = tf.cast(self.inputs_mask, tf.float32)
            mask = tf.expand_dims(mask, 2)
            loss_op1 = tf.reduce_sum(selection_loss_o1, 2, keepdims=True)
            loss_op11 = tf.reduce_sum(loss_op1 * mask) / tf.reduce_sum(mask)

            loss_op2 = tf.reduce_sum(selection_loss_o2, 2, keepdims=True)
            loss_op22 = tf.reduce_sum(loss_op2 * mask) / tf.reduce_sum(mask)
            loss_op = loss_op1 + loss_op2

            self.loss = loss_op11 + loss_op22

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
            gradients_bert = tf.gradients(loss_op, params_of_bert)
            gradients_other = tf.gradients(loss_op, params_of_other)
            gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
            gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
            train_op_bert = opt1.apply_gradients(zip(gradients_bert_clipped, params_of_bert))
            train_op_other = opt2.apply_gradients(zip(gradients_other_clipped, params_of_other))
        if use_lstm:
            self.train_op = (train_op_bert, train_op_other)
        else:
            self.train_op = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps, False)

    @staticmethod
    def seq_maxpool(x):
        """
        seq是[None, seq_len, s_size]的格式，
        mask是[None, seq_len, 1]的格式，先除去mask部分，
        然后再做maxpooling。
        """
        seq, mask = x
        seq -= (1 - mask) * 1e10
        return tf.reduce_max(seq, 1, keepdims=True)

