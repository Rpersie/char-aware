from __future__ import division
from __future__ import print_function

import codecs
import collections
import os

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
    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab

    word_vocab = Vocab()
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab

    actual_max_word_length = 0

    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)

    for fname in ('train.txt', 'valid.txt', 'test.txt'):
        print('reading', fname)
        with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                # 清洗
                line = line.replace('}', '').replace('{', '').replace('|', '')
                line = line.replace('<unk>', ' | ')
                if eos:
                    line = line.replace(eos, '')

                for word in line.split():
                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length - 2]

                    word_tokens[fname].append(word_vocab.feed(word))

                    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                    char_tokens[fname].append(char_array)

                    actual_max_word_length = max(actual_max_word_length, len(char_array))

                if eos:
                    word_tokens[fname].append(word_vocab.feed(eos))

                    char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                    char_tokens[fname].append(char_array)

    assert actual_max_word_length <= max_word_length

    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    print('number of tokens in train.txt:', len(word_tokens['train.txt']))
    print('number of tokens in valid.txt:', len(word_tokens['valid.txt']))
    print('number of tokens in test.txt:', len(word_tokens['test.txt']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in ('train.txt', 'valid.txt', 'test.txt'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])

        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname][i, :len(char_array)] = char_array

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


if __name__ == '__main__':

    _, _, wt, ct, _ = load_data('data', 65)
    print(wt.keys())

    count = 0
    for x, y in DataReader(wt['valid.txt'], ct['valid.txt'], 20, 35).iter():
        count += 1
        print(x, y)
        if count > 0:
            break
