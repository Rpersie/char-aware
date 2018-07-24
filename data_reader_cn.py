from __future__ import division
from __future__ import print_function

import codecs
import collections
import os
from collections import defaultdict

import numpy as np


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
            np.pickle.dump((self._token2index, self._index2token), f, np.pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = np.pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_word_length, eos='+'):
    """
    加载数据，获取1.字母字典 2.单词字典 3.单词embedding 4.字母embedding 5.最大字符长度
    :param data_dir:
    :param max_word_length:
    :param eos:
    :return:
    """
    # 笔画数据
    stroke_order_dir = "./data"
    stroke_file_name = os.path.join(stroke_order_dir, "cns11643_strokeorder.txt")
    stroke_order = StrokeOrder()
    stroke_order.load_stroke_order(file_name=stroke_file_name)

    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab

    word_vocab = Vocab()
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab

    actual_max_word_length = 0

    # 带初始值的字典
    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                # 清洗
                line = line.replace('}', '').replace('{', '').replace('|', '')
                line = line.replace('<unk>', ' | ')
                if eos:
                    line = line.replace(eos, '')
                for charac in line:
                    # 每读一个单词，需要 1.更新词汇表 2.更新字母表（笔画表） 3.更新文本对应的词汇列表 4.更新文本对应的字母列表
                    # todo 修改为最长笔画的获取

                    order_str = stroke_order.get_order(charac)
                    if len(order_str) > max_word_length - 2:  # space for 'start' and 'end' chars
                        charac = charac[:max_word_length - 2]

                    # feed 返回的是下标
                    word_tokens[fname].append(word_vocab.feed(charac))

                    stroke_array = [char_vocab.feed(c) for c in '{' + order_str + '}']
                    char_tokens[fname].append(stroke_array)
                    actual_max_word_length = max(actual_max_word_length, len(stroke_array))

                if eos:
                    word_tokens[fname].append(word_vocab.feed(eos))

                    stroke_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                    char_tokens[fname].append(stroke_array)

    assert actual_max_word_length <= max_word_length

    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    print('number of tokens in train.txt:', len(word_tokens['train']))
    print('number of tokens in valid.txt:', len(word_tokens['valid']))
    print('number of tokens in test.txt:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in ('train', 'valid', 'test'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])

        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)

        for i, stroke_array in enumerate(char_tokens[fname]):
            char_tensors[fname][i, :len(stroke_array)] = stroke_array

    return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length


class DataReader:
    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps):
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]

        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()

        x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


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


if __name__ == '__main__':

    _, _, wt, ct, _ = load_data('./data', 65)


    count = 0
    for x, y in DataReader(wt['valid.txt'], ct['valid.txt'], 20, 35).iter():
        count += 1

        if count > 0:
            break

            # data_dir="data"
            # stroke_file_name=os.path.join(data_dir,"cns11643_strokeorder.txt")
            # stroke_order=StrokeOrder()
            # stroke_order.load_stroke_order(file_name=stroke_file_name)
            # stroke_order=stroke_order.stroke_order
            # print(stroke_order["酒"])
