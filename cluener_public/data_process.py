# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : wangbingqian@boe.com.cn
@time   :20-12-9 下午9:04
@IDE    :PyCharm
@document   :data_process.py
"""
import json


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


attr_set = set()
vocab_set = {}
file1 = open('train/input.seq.char', 'w', encoding='utf-8')
file2 = open('train/output.seq.attr', 'w', encoding='utf-8')
file3 = open('train/output.seq.bio', 'w', encoding='utf-8')
file4 = open('train/output.seq.bioattr', 'w', encoding='utf-8')

datas = load_data('train.json', is_train=True)
print(datas[:10])
for itm in datas:
    text = itm['text']
    seq = list(text)
    for s in seq:
        if s in vocab_set:
            vocab_set[s] += 1
        else:
            vocab_set[s] = 1
    seq_bio = ['O']*len(text)
    seq_bio_attr = ['O']*len(text)
    seq_attr = ['null']*len(text)
    spo_list = itm['spo_list']
    for lb, et, ss, ee in spo_list:
        attr_set.add(lb)
        seq_bio[ss] = 'B'
        seq_attr[ss] = lb
        seq_bio_attr[ss] = 'B-'+lb
        for i in range(1, len(et)):
            seq_bio[ss+i] = 'I'
            seq_attr[ss+i] = lb
            seq_bio_attr[ss+i] = 'I-' + lb

    print(seq)
    print(seq_bio)
    print(seq_attr)
    print(seq_bio_attr)
    print('********'*3)
    assert len(seq_bio_attr) == len(seq)
    assert len(seq_bio) == len(seq)
    assert len(seq_attr) == len(seq)
    file1.write(' '.join(seq) + '\n')
    file2.write(' '.join(seq_attr) + '\n')
    file3.write(' '.join(seq_bio) + '\n')
    file4.write(' '.join(seq_bio_attr) + '\n')

file1 = open('test/input.seq.char', 'w', encoding='utf-8')
file2 = open('test/output.seq.attr', 'w', encoding='utf-8')
file3 = open('test/output.seq.bio', 'w', encoding='utf-8')
file4 = open('test/output.seq.bioattr', 'w', encoding='utf-8')

datas = load_data('dev.json', is_train=True)
print(datas[:10])
for itm in datas:
    text = itm['text']
    seq = list(text)
    for s in seq:
        if s in vocab_set:
            vocab_set[s] += 1
        else:
            vocab_set[s] = 1
    seq_bio = ['O']*len(text)
    seq_bio_attr = ['O']*len(text)
    seq_attr = ['null']*len(text)
    spo_list = itm['spo_list']
    for lb, et, ss, ee in spo_list:
        seq_bio[ss] = 'B'
        seq_attr[ss] = lb
        seq_bio_attr[ss] = 'B-'+lb
        for i in range(1, len(et)):
            seq_bio[ss+i] = 'I'
            seq_attr[ss+i] = lb
            seq_bio_attr[ss+i] = 'I-' + lb

    print(seq)
    print(seq_bio)
    print(seq_attr)
    print(seq_bio_attr)
    print('********'*3)
    assert len(seq_bio_attr) == len(seq)
    assert len(seq_bio) == len(seq)
    assert len(seq_attr) == len(seq)
    file1.write(' '.join(seq) + '\n')
    file2.write(' '.join(seq_attr) + '\n')
    file3.write(' '.join(seq_bio) + '\n')
    file4.write(' '.join(seq_bio_attr) + '\n')

file5 = open('vocab_attr.txt', 'w', encoding='utf-8')
for attr in attr_set:
    file5.write(attr + '\n')

file6 = open('vocab_bioattr.txt', 'w', encoding='utf-8')


def prepare_label2():
    import re
    text = """
    地址（address）: 544
    书名（book）: 258
    公司（company）: 479
    游戏（game）: 281
    政府（government）: 262
    电影（movie）: 307
    姓名（name）: 710
    组织机构（organization）: 515
    职位（position）: 573
    景点（scene）: 288
    """

    a = re.findall(r"（(.*?)）", text.strip())
    print(a)
    label2id = {"O": 0}
    index = 1
    for i in a:
        label2id["B-" + i] = index
        label2id["I-" + i] = index + 1
        # label2id["M_" + i] = index + 2
        # label2id["E_" + i] = index + 3
        index += 2
    for k, v in label2id.items():
        file6.write(k + '\n')

    # open("label2id.json", "w").write(json.dumps(label2id, ensure_ascii=False, indent=2))

prepare_label2()


# file1 = open('test2/input.seq.char', 'w', encoding='utf-8')
# file2 = open('test2/output.seq.attr', 'w', encoding='utf-8')
# file3 = open('test2/output.seq.bio', 'w', encoding='utf-8')
# file4 = open('test2/output.seq.bioattr', 'w', encoding='utf-8')

datas = load_data('test.json', is_train=False)
print(datas[:10])
for itm in datas:
    text = itm['text']
    seq = list(text)
    for s in seq:
        if s in vocab_set:
            vocab_set[s] += 1
        else:
            vocab_set[s] = 1

vocabs = sorted(vocab_set.items(), key=lambda x: x[1], reverse=True)

file1 = open('vocab_char.txt', 'w', encoding='utf-8')
for v in vocabs:
    file1.write(v[0] + '\n')
