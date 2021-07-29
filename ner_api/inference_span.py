# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-11-27 上午11:42
@IDE    :PyCharm
@document   :inference2.py
"""

import numpy as np
import tensorflow as tf
from tools.my_tokenizers import Tokenizer
import json
import codecs
from tools.my_scp import *
import datetime
from tools.config import *


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class NerModel(object):
    def __init__(self, model_name):
        # 启动会话
        self.max_len = 510
        self.tokenizer = OurTokenizer(self.load_vocab('config/vocab.txt'))
        self.label2id, self.id2label = self.load_vocabulary("./cluener_public/vocab_bioattr.txt")
        self.viteb = ViterbiDecoder(len(self.label2id))
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        self.__sess = tf.Session(config=session_config)

        # 载入数据
        self.saver = tf.train.import_meta_graph(model_name + '.meta')
        self.saver.restore(self.__sess, model_name)

        # 载入图结构
        self.__graph = tf.get_default_graph()

    def extraction_spo(self, text):
        if len(text) > 510:
            max_len = self.max_len
        else:
            max_len = len(text) + 2
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=max_len)
        input_mask = [1] * len(token_ids)
        token_ids = self.sequence_padding([token_ids], max_len)
        input_mask = self.sequence_padding([input_mask], max_len)
        segment_ids = self.sequence_padding([segment_ids], max_len)
        feed_dict = {'inputs_seq:0': token_ids,
                     'inputs_mask:0': input_mask,
                     'inputs_segment:0': segment_ids,
                     'is_train:0': False,
                     'keep_prob:0': 1.0}
        probs_seq, transitions_matrix = \
            self.__sess.run([self.__graph.get_tensor_by_name('projection/probs_seq:0'),
                             self.__graph.get_tensor_by_name('projection/transitions:0')],
                            feed_dict=feed_dict)
        # print(transitions_matrix)
        # print(len(transitions_matrix))     # 21
        # print(len(transitions_matrix[0]))  # 21
        # print(probs_seq[0])
        # print(len(probs_seq[0]))     # seq_length
        # print(len(probs_seq[0][0]))  # labels
        labels = self.viteb.decode(probs_seq[0], transitions_matrix)
        arguments, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    arguments.append([[i], self.id2label[label]])
                elif starting:
                    arguments[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(l[2:], text[w[0] - 1:w[-1]], w[0] - 1, w[-1] - 1) for w, l in arguments]
        # another method for crf decode
        # inputs_seq_len = tf.reduce_sum(input_mask, axis=-1)  # B
        # op = tf.contrib.crf.crf_decode(probs_seq, transitions_matrix, inputs_seq_len)
        # with tf.Session() as sess:
        #     preds_seq, crf_scores = sess.run(op)
        #     # print(len(preds_seq))
        #     # print(len(preds_seq[0]))
        #     arguments, starting = [], False
        #     for i, label in enumerate(preds_seq[0]):
        #         if label > 0:
        #             if label % 2 == 1:
        #                 starting = True
        #                 arguments.append([[i], self.id2label[label]])
        #             elif starting:
        #                 arguments[-1][0].append(i)
        #             else:
        #                 starting = False
        #         else:
        #             starting = False
        #
        #     return [(l[2:], text[w[0] - 1:w[-1]], w[0] - 1, w[-1] - 1) for w, l in arguments]

    @staticmethod
    def sequence_padding(inputs, length=None, padding=0):
        if length is None:
            length = max([len(x) for x in inputs])

        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[:length]
            pad_width[0] = (0, length - len(x))
            x = np.pad(x, pad_width, 'constant', constant_values=padding)
            outputs.append(x)

        return np.array(outputs)

    @staticmethod
    def load_schema(schema_path):
        id2predicate, predicate2id = json.load(open(schema_path, encoding='utf-8'))
        id2predicate = {int(i) - 1: j for i, j in id2predicate.items()}
        return id2predicate

    @staticmethod
    def load_vocab(dict_path):
        token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return token_dict

    @staticmethod
    def load_vocabulary(path):
        vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
        print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i
            i2w[i] = w
        return w2i, i2w

    def submit(self, test_data):
        data = []
        for d in test_data:
            labels = {}
            res = self.extraction_spo(d['text'])
            for itm in res:  # label是否存在
                if labels.get(itm[0]):  # 实体是否存在
                    if labels[itm[0]].get(itm[1]):  # 如果实体存在,则　追加
                        labels[itm[0]][itm[1]].append([itm[2], itm[3]])
                    else:  # 如果实体不存在，则
                        labels[itm[0]][itm[1]] = [[itm[2], itm[3]]]
                else:
                    labels[itm[0]] = {itm[1]: [[itm[2], itm[3]]]}
            data.append(json.dumps({"label": labels}, ensure_ascii=False))
        open("crf_{}_ner_predict.json".format(bert_type), "w", encoding='utf-8').write("\n".join(data))
        upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/", file_path="crf_{}_ner_predict.json".format(bert_type))


class ViterbiDecoder(object):
    """Viterbi解码算法基类
    """
    def __init__(self, num_labels, starts=[0], ends=[0]):
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes, trans):
        """nodes.shape=[seq_len, num_labels]
        """
        # 预处理
        nodes[0, self.non_starts] -= np.inf
        nodes[-1, self.non_ends] -= np.inf

        # 动态规划
        labels = np.arange(len(trans)).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        for l in range(1, len(nodes)):
            M = scores + trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)

        # 最优路径
        return paths[:, scores[:, 0].argmax()]


class SPO(tuple):
    def __init__(self, spo):
        self.spox = (
            spo[0],
            tuple(spo[1]),
            spo[2],
            spo[3]
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

#
# pretrain_model_path = 'bert_model'
# bert_path = pretrain_model_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/'
# config_path = bert_path + 'bert_config.json'
# checkpoint_path = None
# dict_path = 'bert_model/vocab.txt'
# schema_path = 'lic2019/all_50_schemas_me.json'
#
# from my_utils import load_data
#
# valid_data = load_data('cluener_public/test.json', is_train=False)
# if __name__ == "__main__":
#     # model_name = 'model/train_model_span_20201210.ckpt'
#     model_name = 'model/ner_{}_crf_{}.ckpt'.format(bert_type, datetime.datetime.now().strftime('%Y%m%d'))
#
#     model = NerModel(model_name=model_name, vocab_path=dict_path)
#
#     # text = '《离开》是由张宇谱曲，演唱'
#     text = '焦作市河阳酒精实业有限公司于2001年2月23日在孟州市工商行政管理局登记成立'
#     res = model.extraction_spo(text)
#     # print(res)
#     model.submit(valid_data)
    # X, Y, Z = 1e-10, 1e-10, 1e-10
    # pbar = tqdm()
    # for d in valid_data:
    #     R = set([SPO(spo) for spo in model.extraction_spo(d['text'])])
    #     T = set([SPO(spo) for spo in d['spo_list']])
    #     print('R', R)
    #     print('T', T)
    #     X += len(R & T)
    #     Y += len(R)
    #     Z += len(T)
    #     f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    #     pbar.update()
    #     pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
    #     # log.info('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
    # pbar.close()
