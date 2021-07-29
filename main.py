# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-25 下午5:49
@IDE    :PyCharm
@document   :main.py
"""
# from ner_api.inference_span import NerModel
from ner_api.inference_crf import NerModel
# model = NerModel(model_name='model/ner_robert_base_crf_20201225.ckpt')
model = NerModel(model_name='model/ner_robert_base_span_20201216.ckpt')
if __name__ == "__main__":
    # text = '《离开》是由张宇谱曲，演唱'
    # text = '焦作市河阳酒精实业有限公司于2001年2月23日在孟州市工商行政管理局登记成立'
    text = '从官方网站提供的PDF公告文档可以发现PSP专用的字样，如果PC版《伊苏7》永远不会问世，'
    res = model.extraction_spo(text)
    print(res)
