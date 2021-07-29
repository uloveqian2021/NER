# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-9 下午9:04
@IDE    :PyCharm
@document   :data_process.py for bilstm+crf
"""
import json
import numpy as np
from tqdm import tqdm
import logging as log

log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO)


# 加载数据集合
def load_data(file_name, is_train=False, debug=False):
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


def load_vocabulary(path):
    vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w


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
    def __iter__(self, random=False, w2i_bio=None, max_len=None, w2i_char=None):
        batch_token_ids, batch_seq_length, batch_segment_ids, batch_labels = [], [], [], []
        for is_end, d in self.sample(random):

            # token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=max_len)
            text = d['text']
            if len(text) > max_len:
                text = d['text'][:max_len]
            token_ids = list(text)
            labels = ['O'] * len(token_ids)
            for argument in d['spo_list']:  # (label, entity, span)
                start_index = argument[2]+1
                end_index = argument[3]+1

                if end_index < len(token_ids):
                    labels[start_index] = 'B-' + argument[0]
                    for i in range(1, len(argument[1])):
                        labels[start_index + i] = 'I-' + argument[0]
            labels = [w2i_bio[itm] for itm in labels]
            token_ids = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in token_ids]
            batch_seq_length.append(len(token_ids))
            batch_token_ids.append(token_ids)
            # batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_seq_length = np.array(batch_seq_length)
                # batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield batch_token_ids, batch_seq_length, batch_labels
                batch_token_ids, batch_seq_length, batch_segment_ids, batch_labels = [], [], [], []


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


# {'O': 0, 'B-address': 1, 'I-address': 2, 'B-book': 3, 'I-book': 4, 'B-company': 5, 'I-company': 6, 'B-game': 7, 'I-game': 8, 'B-government': 9, 'I-government': 10, 'B-movie': 11, 'I-movie': 12,
# 'B-name': 13, 'I-name': 14, 'B-organization': 15, 'I-organization': 16, 'B-position': 17, 'I-position': 18, 'B-scene': 19, 'I-scene': 20} {0: 'O', 1: 'B-address', 2: 'I-address', 3: 'B-book',
# 4: 'I-book', 5: 'B-company', 6: 'I-company', 7: 'B-game', 8: 'I-game', 9: 'B-government', 10: 'I-government', 11: 'B-movie', 12: 'I-movie', 13: 'B-name', 14: 'I-name', 15: 'B-organization',
# 16: 'I-organization', 17: 'B-position', 18: 'I-position', 19: 'B-scene', 20: 'I-scene'}


def extract_entities(text,  model, sess, w2i_char, max_len, id2label):
    """arguments抽取函数
    """
    if len(text) > max_len:
        text = text[:max_len]
    token_ids = list(text)
    token_ids = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in token_ids]
    seq_length = [len(token_ids)]
    # input_mask = [1] * len(token_ids)

    token_ids = sequence_padding([token_ids], max_len)
    # input_mask = sequence_padding([input_mask], max_len)
    seq_length = np.array(seq_length)
    feed_dict = {model.inputs_ids: token_ids,
                 model.inputs_seq_len: seq_length,
                 # model.inputs_segment: segment_ids,
                 model.keep_prob: 1.0}
    preds_seq = sess.run(model.outputs, feed_dict)[0]
    arguments, starting = [], False
    for i, label in enumerate(preds_seq):
        if label > 0:
            if label % 2 == 1:
                starting = True
                arguments.append([[i], id2label[label]])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return [(l[2:], text[w[0]-1:w[-1]], w[0]-1, w[-1]-1) for w, l in arguments]


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
        R = set([SPO(spo) for spo in extract_entities(d['text'], model, sess, tokenizer, max_len, id2predicate)])
        T = set([SPO(spo) for spo in d['spo_list']])

        print('R:', R)
        print('T:', T)
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
