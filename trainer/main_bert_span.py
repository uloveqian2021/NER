# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-9 下午9:04
@IDE    :PyCharm
@document   :data_process.py
"""

import tensorflow as tf
# import tokenization
from trainer.my_utils import *
from trainer.model_bert_span import MyModel as model_fn
from tools.my_tokenizers import Tokenizer
from tools.my_scp import *
from tools.config import *
import datetime
import logging as log
import codecs
log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO)
log.info("start training!")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("mode", 'train', "The input datadir.", )
flags.DEFINE_string("data_dir", '../lic2019', "The input data dir. Should con ""for the task.")
flags.DEFINE_string("output_dir", './model', "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("bert_config_file", config_path, "The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string("init_checkpoint", checkpoint_path, "Initial checkpoint  BERT model).")
flags.DEFINE_string("vocab_file", dict_path, "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")

flags.DEFINE_integer("max_seq_length", max_len, "The maximum total input sequence length after WordPiece tokenization.")
flags.DEFINE_integer("batch_size", batch_size, "Total batch size for training.")
flags.DEFINE_float("learning_rate", lr, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_epochs", 30, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. ""E.g., 0.1 = 10% of training.")


# 验证集评估
def evaluate_val(valid_data, model, sess, tokenizer, max_len, id2predicate, limit=None):
    f1, precision, recall = evaluate(valid_data, model, sess, tokenizer, max_len, id2predicate, limit)
    best_test_f1 = model.best_dev_f1.eval()
    if f1 > best_test_f1:
        tf.assign(model.best_dev_f1, f1).eval()  # 赋值操作  将f1值赋给model.best_dev_f1
        print('precision: %.5f, recall: %.5f ,f1: %.5f,' % (precision, recall, f1))
    test_f1 = model.best_dev_f1.eval()
    log.info('precision: %.5f, recall: %.5f ,f1: %.5f, best_f1:%.5f' % (precision, recall, f1, test_f1))
    return f1 > best_test_f1


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


def main(_):

    token_dict = {}
    with codecs.open(FLAGS.vocab_file, 'r', 'utf8') as reader:
        for line in reader:
            # token = line.strip()
            token = line.strip().split('\t')[0]
            token_dict[token] = len(token_dict)
    tokenizer = OurTokenizer(token_dict, do_lower_case=True)
    train_data = load_data('cluener_public/train.json', True, debug=debug)
    valid_data = load_data('cluener_public/dev.json', True, debug=debug)

    train_D = data_generator(train_data, FLAGS.batch_size)
    train_examples = train_data
    num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model = model_fn(bert_config=FLAGS.bert_config_file,
                     init_checkpoint=FLAGS.init_checkpoint,
                     num_labels=len(predicate2id),
                     use_lstm=True,
                     learning_rate=FLAGS.learning_rate,
                     num_train_steps=num_train_steps,
                     num_warmup_steps=num_warmup_steps,
                     )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 模型保存路径
    checkpoint_path = os.path.join('model', 'ner_{}_span_{}.ckpt'.format(bert_type, datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path0 = os.path.join('model', 'ner_{}_span_{}.ckpt.data-00000-of-00001'.format(bert_type, datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path1 = os.path.join('model', 'ner_{}_span_{}.ckpt.index'.format(bert_type, datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path2 = os.path.join('model', 'ner_{}_span_{}.ckpt.meta'.format(bert_type, datetime.datetime.now().strftime('%Y%m%d')))
    # ===============================
    # 加载bert_config文件
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state('model')

        # ===============================
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            try:
                print('mode_path %s' % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            except:
                pass
        # ============================
        for j in range(FLAGS.num_train_epochs):  # 30
            print('j', j)
            eval_los = 0.0
            count = 0
            step = 0
            for (batch_text, batch_token_ids, batch_mask, batch_segment_ids,
                 batch_object_label_o1, batch_object_label_o2) \
                    in train_D.__iter__(random=True,
                                        predicate2id=predicate2id,
                                        max_len=FLAGS.max_seq_length,
                                        tokenizer=tokenizer):
                count = count + 1
                feed = {model.inputs_ids: batch_token_ids,
                        model.inputs_mask: batch_mask,
                        model.inputs_segment: batch_segment_ids,
                        model.tag_o1_inputs: batch_object_label_o1,
                        model.tag_o2_inputs: batch_object_label_o2,
                        model.is_training: True,
                        model.keep_prob: 0.8
                        }
                step = step + 1
                loss, _ = sess.run([model.loss, model.train_op], feed)

                eval_los = loss + eval_los
                los = eval_los / count
                if step % 40 == 0:
                    log.info('epoch:{}, step:{}, loss:{}'.format(j, step, los))

            best = evaluate_val(valid_data, model, sess, tokenizer, FLAGS.max_seq_length, id2predicate)
            if best:
                saver.save(sess, checkpoint_path)
            if online:
                upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/lic2019/", file_path=checkpoint_path0)
                upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/lic2019/", file_path=checkpoint_path1)
                upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/lic2019/", file_path=checkpoint_path2)
                upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/lic2019/", file_path='model/checkpoint')


if __name__ == "__main__":
    tf.app.run()
    upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/lic2019/", file_path='model/')
