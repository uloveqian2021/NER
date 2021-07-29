# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-9 下午9:04
@IDE    :PyCharm
@document   :data_process.py
"""
import json
import numpy as np
# import unicodedata
from tqdm import tqdm
import logging as log

log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO)
labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
predicate2id = {}
id2predicate = {}
for i, l in enumerate(labels):
    predicate2id[l] = i
    id2predicate[i] = l
print(id2predicate)
print(predicate2id)


# 加载数据集合
def load_data(file_name, is_train=True, debug=False):
    D = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            l = json.loads(line)
            spo_list = []
            if is_train:
                text = l['text']
                label = l['label']  # {"company": {"美联银行": [[0, 3]], "花旗": [[11, 12]], "富国": [[13, 14]]}}
                for k, v in label.items():
                    for kk, vv in v.items():  # {"美联银行": [[0, 3]], "花旗": [[11, 12]], "富国": [[13, 14]]}
                        for vvv in vv:
                            spo_list.append((k, kk, vvv[0], vvv[1]))  # (label, entity, span)

            else:
                text = l['text']
            D.append({'text': text,
                      'spo_list': spo_list
                      })
    if debug:
        return D[:len(D) // 4]
    return D


def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class DataGenerator(object):
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)
            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False, predicate2id=None, max_len=None, tokenizer=None):
        T0, T1, T2, M1, O1, O2 = [], [], [], [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=max_len)

            # 整理三元组 {s: [(o, p)]}
            spoes = []
            for l, e, idx_s, idx_e in d['spo_list']:
                p = predicate2id[l]
                s_idx = idx_s
                e_idx = idx_e
                o = (s_idx + 1, e_idx + 1, p)
                spoes.append(o)

            if spoes:
                # mask
                input_mask = [1] * len(token_ids)  # TODO
                o1 = np.zeros((len(token_ids), len(predicate2id)))
                o2 = np.zeros((len(token_ids), len(predicate2id)))
                for idx1, idx2, lb in spoes:
                    # print(idx1, idx2, len(o1))
                    o1[idx1][lb] = 1
                    o2[idx2][lb] = 1
                # 构建batch
                T0.append(d['text'])
                T1.append(token_ids)
                T2.append(segment_ids)
                M1.append(input_mask)
                # K1.append([start])
                # K2.append([end])
                O1.append(o1)
                O2.append(o2)
                if len(T1) == self.batch_size or is_end:
                    T1 = sequence_padding(T1)
                    T2 = sequence_padding(T2)
                    M1 = sequence_padding(M1)
                    O1 = sequence_padding(O1)
                    O2 = sequence_padding(O2)
                    # K1, K2, K3 = np.array(K1), np.array(K2), np.array(K3)
                    yield T0, T1, M1, T2, O1, O2
                    T0, T1, T2, M1, S1, S2, K3, O1, O2, = [], [], [], [], [], [], [], [], []


def closed_single(text, model, sess, tokenizer, max_len, id2predicate):
    # tokens = tokenizer.tokenize(text, maxlen=max_len)
    # mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=max_len)
    input_mask = [1] * len(token_ids)
    token_ids = sequence_padding([token_ids], max_len)
    input_mask = sequence_padding([input_mask], max_len)
    segment_ids = sequence_padding([segment_ids], max_len)
    feed_dict = {model.inputs_ids: token_ids,
                 model.inputs_mask: input_mask,
                 model.inputs_segment: segment_ids,
                 model.is_training: False,
                 model.keep_prob: 1.0}
    spoes = []
    object_o1_preds, object_o2_preds = sess.run([model.prob_o1, model.prob_o2], feed_dict)
    start = np.where(object_o1_preds[0] > 0.5)
    end = np.where(object_o2_preds[0] > 0.5)
    for _start, predicate1 in zip(*start):
        for _end, predicate2 in zip(*end):
            if _start <= _end and predicate1 == predicate2:
                spoes.append((predicate1, text[_start-1:_end], _start-1, _end-1))
                break
    return [(id2predicate[p], o, i1, i2) for p, o, i1, i2 in spoes]


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


def evaluate(data, model, sess, tokenizer, max_len, id2predicate, limit=None):
    if limit:
        data = data[:limit]
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in closed_single(d['text'], model, sess, tokenizer, max_len, id2predicate)])
        T = set([SPO(spo) for spo in d['spo_list']])
        # print('R:', R)
        # print('T:', T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        # pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
        log.info('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))

    pbar.close()
    return f1, precision, recall


# res = load_data('cluener_public/train.json', is_train=True)
# print(res)
