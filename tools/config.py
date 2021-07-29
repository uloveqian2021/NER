# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-16 下午4:23
@IDE    :PyCharm
@document   :config.py
"""
# online, bert_type, batch_size, max_len, lr = False, 'robert_base', 8, 128, 2e-5
online, bert_type, batch_size, max_len, lr = False, 'robert_base', 8, 64, 1e-5
# online, bert_type, batch_size, max_len, lr = False, 'chinese_rbt3', 4, 128, 1e-4
debug = False

if online:
    pretrain_model_path = '/data/datasets/tmp_data/pretrain_model/'
else:
    pretrain_model_path = '/wang/pretrain_model/'
if bert_type == 'robert_large':
    bert_path = pretrain_model_path + 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/'
    # bert_path = pretrain_model_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'robert_base':
    bert_path = pretrain_model_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'user_base':
    bert_path = pretrain_model_path + 'user_base/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_uer_chinese.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'user_large':
    bert_path = pretrain_model_path + 'user_large/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_uer_24_chinese.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'simbert':
    bert_path = pretrain_model_path + 'chinese_simbert_L-12_H-768_A-12/'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    config_path = bert_path + 'bert_config.json'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'albert':
    bert_path = pretrain_model_path + 'albert_base/'
    config_path = bert_path + 'albert_config.json'
    checkpoint_path = bert_path + 'model.ckpt-best'
    dict_path = bert_path + 'vocab_chinese.txt'
elif bert_type == 'chinese_rbt3':
    bert_path = pretrain_model_path + 'chinese_rbt3_L-3_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'nezha_large':
    bert_path = pretrain_model_path + 'NEZHA-Large-WWM/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'model.ckpt-346400'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'nezha_base':
    bert_path = pretrain_model_path + 'NEZHA-Base/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'model.ckpt-900000'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'wonezha':
    bert_path = pretrain_model_path + 'chinese_wonezha_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'ernie':
    bert_path = pretrain_model_path + 'ernie-512/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
if bert_type in ['robert_large', 'robert_base', 'simbert', 'user_large', 'ernie', 'roberta', 'chinese_rbt3']:
    bert_type_ = 'bert'
elif bert_type in ['nezha_large', 'nezha_base', 'nezha_wwm']:
    bert_type_ = 'nezha'
