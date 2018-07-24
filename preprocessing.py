"""
  preprocessing
  author: xuqh
  18-7-23 下午4:25
  description: 将所有文本输入，生成字级别的lookup table和笔画级别的lookup table
"""
import codecs
import os
import pickle
from collections import defaultdict

import numpy  as np
import tensorflow as tf


#############################################################################################
#   构造字典
#############################################################################################

class Vocab:
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


class OrderVocab:
    def __init__(self, token2order=None, index2order=None):

        self._token2order = token2order or []
        self._index2order = index2order or {}

    def feed(self, token, order):
        # allocate new index for this token
        if token not in self._index2order:
            index = len(self._index2order)
            self._index2order[token] = order
            self._token2order.append(order)

    @property
    def size(self):
        return len(self._index2order)

    def token(self, index):
        return self._token2order[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2order.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._index2order, self._token2order), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2order, index2order = pickle.load(f)

        return cls(token2order, index2order)


def preprocess(data_dir, save_dir, max_word_length, eos):
    """
    逐行遍历data_dir中数据，构造字的Lookup table和笔画的lookup table
    :param data_dir:
    :param save_dir:
    :param max_word_length:
    :param eos:
    :return: void
    """
    actual_max_word_length = 0
    actual_max_word = ""
    stroke_order_dir = "./cn_data"
    stroke_file_name = os.path.join(stroke_order_dir, "cns11643_strokeorder.txt")
    stroke_order = StrokeOrder()
    stroke_order.load_stroke_order(file_name=stroke_file_name)

    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab
    char_vocab.feed('|')  # <unk> is at index 3 in char vocab

    word_vocab = Vocab()
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab

    stroke_order_vocab = OrderVocab()
    stroke_order_vocab.feed('|', [1, 3, 2])

    fnames = ["train.txt"]
    fnames = [os.path.join(data_dir, fname + '.txt') for fname in fnames]
    # fnames=["train.txt","test.txt","valid.txt"]
    dataset = tf.data.TextLineDataset(fnames)

    # 每行最起码70个字符，否则不给算！注意：这里每一行是一个tensor只能用tensorflow封装的方法，或者使用py_func自己封装
    # 封装需要指定输入参数，和输出的数据类型
    def filter_func(line):
        return len(line.strip()) > 70

    # dataset = dataset.filter(lambda line: tf.py_func(filter_func, [line], [tf.bool]))
    iterator = dataset.make_one_shot_iterator()
    next = iterator.get_next()

    with tf.Session() as sess:
        print("开始解析啦！！！")
        num = 1
        while True:
            try:
                line = sess.run(next)
            except tf.errors.OutOfRangeError:
                break
                # 读入进来是个bytes类型的的人改版你

            line = str(line, encoding='utf-8')
            # 清洗
            line = line.strip()
            line = line.replace('}', '')
            line = line.replace('{', '')
            line = line.replace('|', '')
            line = line.replace('<unk>', ' | ')

            if num % 1000 == 0:
                print("round : ", num)
                print("char vocab size: ", char_vocab.size)
                print("stroke order vocab size: ", stroke_order_vocab.size)
                print("word vocab size: ", word_vocab.size)
                print("actual max word length: ", actual_max_word_length)
                print("actual max word : ", actual_max_word)
            num += 1

            if eos:
                line.replace(eos, "")
            for charac in line:
                # 每读一个单词，需要 1.更新词汇表 2.更新字母表（笔画表） 3.更新文本对应的词汇列表 4.更新文本对应的字母列表

                order_str = stroke_order.get_order(charac)
                if len(order_str) > max_word_length - 2:  # space for 'start' and 'end' chars
                    charac = charac[:max_word_length - 2]

                # feed 返回的是下标
                word_vocab.feed(charac)

                stroke_array = [char_vocab.feed(c) for c in '{' + order_str + '}']
                stroke_order_vocab.feed(charac, stroke_array)
                if actual_max_word_length < len(stroke_array):
                    actual_max_word = charac
                actual_max_word_length = max(actual_max_word_length, len(stroke_array))

            # 分隔符号也是要加入到字典中的
            if eos:
                word_vocab.feed(eos)
                stroke_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                stroke_order_vocab.feed(eos, stroke_array)
        word_vocab.save(os.path.join(save_dir, "word_vocab"))
        char_vocab.save(os.path.join(save_dir, "char_vocab"))
        stroke_order_vocab.save(os.path.join(save_dir, "order_vocab"))


class StrokeOrder:
    """笔画顺序类"""

    def __init__(self):
        self._stroke_order = defaultdict(str)

    def load_stroke_order(self, file_name):
        """
        从文件中获取笔画顺序
        :param file_name:
        :return:{字：笔画列表}
        """
        with codecs.open(filename=file_name, mode="r", encoding="utf-8") as f:

            for line_id, line in enumerate(f):
                if line_id < 2: continue

                line = line.strip()
                items = line.split()
                charac, order = items[0], items[1]
                self._stroke_order[charac] = self.map_order(order)

    @property
    def stroke_order(self):
        return self._stroke_order

    def get_order(self, key):
        """
        获取笔画；非汉字返回符号本身
        :param key:
        :return:
        """
        return self._stroke_order[key] if key in self._stroke_order else key

    def map_order(self, order):
        """
        12345是字典中会出现的字，并且和order中的12345含义完全不同（这一点和英文不同，英文在order和文本中出现的字母是同一个字母）
        :param order:
        :return:
        """
        return "".join(["<{}>".format(x) for x in order])


#############################################################################################
#   构造tensorflow 计算图的embedding
#############################################################################################
#############################################################################################
def make_order_embeddings(max_word_length, order_arr):
    """
    根据笔顺表生成具有最大字长约束的笔顺embeddings
    :param max_word_length:
    :param order_arr:
    :return:
    """
    order_arr = [
        row + [0] * (max_word_length - len(row)) if len(row) <= max_word_length
        else row[:max_word_length - 1] + [row[-1]]
        for row in order_arr
        ]
    order_arr = np.array(order_arr)
    order_embeddings = tf.convert_to_tensor(order_arr)
    return order_embeddings


# 构造tensorflow计算图的batch 输入
#############################################################################################
if __name__ == '__main__':
    # data_dir = "./test_data"
    data_dir = "./data"
    save_dir = "./save"
    # save_dir = "./test_save"
    # max_word_length = 20
    # preprocess(data_dir, save_dir, max_word_length, eos='+')
    word_vocab = Vocab().load(os.path.join(save_dir, "word_vocab"))
    char_vocab = Vocab().load(os.path.join(save_dir, "char_vocab"))
    order_vocab = OrderVocab().load(os.path.join(save_dir, "order_vocab"))
    arr = order_vocab._index2order
    max_word_length = 20
    order_embeddings = make_order_embeddings(max_word_length, arr)
    input_ids = tf.convert_to_tensor([1,2,4,5], dtype=tf.int32)
    lookups = tf.nn.embedding_lookup(order_embeddings, input_ids)
    with tf.Session() as sess:
        print(sess.run(lookups))
