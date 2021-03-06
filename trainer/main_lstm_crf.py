# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-9 下午9:04
@IDE    :PyCharm
@document   :data_process.py
"""
import tensorflow as tf
from trainer.my_utils_v3 import *
from trainer.model_lstm_crf import MyModel as model_fn
# from tools.my_tokenizers import Tokenizer, load_vocab
from tools.my_scp import *
import datetime
import logging as log
import codecs


log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO)
log.info("start training!")

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# online, bert_type, batch_size, max_len, lr = True, 'robert_base', 32, 256, 2e-5
online, bert_type, batch_size, max_len, lr = False, 'robert_base', 16, 128, 2e-5
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
    bert_type = 'bert'
elif bert_type in ['nezha_large', 'nezha_base', 'nezha_wwm']:
    bert_type = 'nezha'

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


def main(_):
    w2i_char, i2w_char = load_vocabulary("./cluener_public/vocab_char.txt")
    w2i_bio, i2w_bio = load_vocabulary("./cluener_public/vocab_bioattr.txt")
    train_data = load_data('cluener_public/train.json', True, debug=debug)
    valid_data = load_data('cluener_public/dev.json', True, debug=debug)

    train_D = data_generator(train_data, FLAGS.batch_size)
    train_examples = train_data
    num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    model = model_fn(hidden_dim=300,
                     embedding_dim=300,
                     vocab_size_char=len(w2i_char),
                     vocab_size_bio=len(w2i_bio),
                     use_crf=True,
                     # learning_rate=FLAGS.learning_rate,
                     # num_train_steps=num_train_steps,
                     # num_warmup_steps=num_warmup_steps,
                     )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 模型保存路径
    checkpoint_path = os.path.join('model', 'train_model_lstm_{}.ckpt'.format(datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path0 = os.path.join('model', 'train_model_lstm_{}.ckpt.data-00000-of-00001'.format(datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path1 = os.path.join('model', 'train_model_lstm_{}.ckpt.index'.format(datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path2 = os.path.join('model', 'train_model_lstm_{}.ckpt.meta'.format(datetime.datetime.now().strftime('%Y%m%d')))
    # ===============================
    # 加载bert_config文件
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        # saver = tf.compat.v1.train.Saver([v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name],
        #                                   max_to_keep=2)
        ckpt = tf.compat.v1.train.get_checkpoint_state('model')

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
            for (batch_token_ids, batch_seq_length, batch_labels) \
                    in train_D.__iter__(random=True,
                                        w2i_bio=w2i_bio,
                                        max_len=FLAGS.max_seq_length,
                                        w2i_char=w2i_char):
                if j == 0 and step == 0:
                    log.info("###### shape of a batch #######")
                    log.info("input_seq: " + str(batch_token_ids.shape))
                    log.info("input_seq_len: " + str(batch_seq_length.shape))
                    log.info("output_seq: " + str(batch_labels.shape))
                    log.info("###### preview a sample #######")
                    log.info("input_seq:" + " ".join([i2w_char[i] for i in batch_token_ids[0]]))
                    log.info("input_seq_len :" + str(batch_seq_length[0]))
                    log.info("output_seq: " + " ".join([i2w_bio[i] for i in batch_labels[0]]))
                    log.info("###############################")

                count = count + 1
                feed = {model.inputs_ids: batch_token_ids,
                        model.inputs_seq_len: batch_seq_length,
                        model.outputs_seq: batch_labels,
                        model.keep_prob: 0.8
                        }
                step = step + 1
                loss, acc, _ = sess.run([model.loss, model.acc, model.train_op], feed)

                eval_los = loss + eval_los
                los = eval_los / count
                if step % 40 == 0:
                    log.info('epoch:{}, step:{}, acc:{}, loss:{}'.format(j, step, acc, los))

            best = evaluate_val(valid_data, model, sess, w2i_char, FLAGS.max_seq_length, i2w_bio)
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
