# 命名实体识别
命名实体识别（Named Entity Recognition, NER），也称序列标注（Sequence Tagging）
可用于问答、对话任务中的实体和槽位识别、也可用于OCR后文本信息的结构化处理

## 实现方案
* 方案1: BiLSTM/BERT + CRF (已完成)
* 方案2: BiLSTM/BERT + SPAN (已完成)  效果：f1: 0.80654, precision: 0.81038, recall: 0.80273, best f1: 0.80654

## 目前支持标签种类

地址（address）、书名（book）、公司（company）、游戏（game）、政府（government）、电影（movie）、姓名（name）、组织机构（organization）、职位（position）、景点（scene）


## 调用方法

'''
from ner_api.inference_crf import NerModel

model = NerModel(model_name='model/ner_robert_base_crf_20201225.ckpt')

model = NerModel(model_name='model/ner_robert_base_span_20201216.ckpt')
res = model.extraction_spo(text)
'''

##  开发日志

2020/12/25 完成第一版线上部署代码

## 依赖

- tensorflow-gpu==1.14.0
- tensorflow==1.14.0


## 数据来源
* https://github.com/CLUEbenchmark/CLUENER2020
* 技术报告： https://arxiv.org/abs/2001.04351
