# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-11-27 上午11:42
@IDE    :PyCharm
@document   :inference2.py
"""

import numpy as np
import tensorflow
from tools.my_tokenizers import Tokenizer
import json
import codecs
from tools.my_scp import *
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
    def __init__(self, model_name, vocab_path=None, schema_path=None):
        # 启动会话
        self.max_len = 510
        self.tokenizer = OurTokenizer(self.load_vocab('config/vocab.txt'))
        labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        predicate2id = {}
        id2predicate = {}
        for i, l in enumerate(labels):
            predicate2id[l] = i
            id2predicate[i] = l
        self.id2predicate = id2predicate
        session_config = tensorflow.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        self.__sess = tensorflow.Session(config=session_config)

        # 载入数据
        self.saver = tensorflow.train.import_meta_graph(model_name + '.meta')
        self.saver.restore(self.__sess, model_name)

        # 载入图结构
        self.__graph = tensorflow.get_default_graph()

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
        spoes = []
        object_preds1, object_preds2 = \
            self.__sess.run([self.__graph.get_tensor_by_name('dense/score_o1:0'),
                             self.__graph.get_tensor_by_name('dense/score_o2:0')],
                            feed_dict=feed_dict)

        start = np.where(object_preds1[0] > 0.5)
        end = np.where(object_preds2[0] > 0.5)
        for _start, predicate1 in zip(*start):
            for _end, predicate2 in zip(*end):
                if _start <= _end and predicate1 == predicate2:
                    spoes.append((predicate1, text[_start - 1:_end], int(_start - 1), int(_end - 1)))
                    break
        return [(self.id2predicate[p], o, i1, i2) for p, o, i1, i2 in spoes]

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
            # print(labels)
            data.append(json.dumps({"label": labels}, ensure_ascii=False))
        open("span_{}_ner_predict.json".format(bert_type), "w", encoding='utf-8').write("\n".join(data))
        upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/", file_path="span_{}_ner_predict.json".format(bert_type))


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




# from my_utils import load_data
#
# valid_data = load_data('cluener_public/test.json', is_train=False)
# if __name__ == "__main__":
#     model_name = '/wang/my_program/ner_robert_large_span_20201216.ckpt'
#     # model_name = 'model/ner_{}_span_{}.ckpt'.format(bert_type, datetime.datetime.now().strftime('%Y%m%d'))
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
