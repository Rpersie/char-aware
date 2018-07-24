"""
data_reader 测试文件
"""
import codecs
import os

import tensorflow as tf

from data_reader_cn import load_data


class TestReader(tf.test.TestCase):
    def setUp(self):
        self._string_data = "\n".join(
            [" 床前明d，。12月光？",
             " 概率论与数理统计",
             " 张宇线性代数"])
        tmpdir = tf.test.get_temp_dir()
        for suffix in "train.txt", "valid.txt", "test.txt":
            filename = os.path.join("%s.txt" % suffix)
            with codecs.open(os.path.join(tmpdir, filename), 'w', 'utf-8') as f:
                print(tmpdir)
                f.write(self._string_data)

    def test_load_data(self):
        word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length = load_data(tf.test.get_temp_dir(),
                                                                                               100)
        print("word_vocab=", word_vocab)
        print("char_vocab=", char_vocab)
        print("word_tensors=", word_tensors)
        print("char_tensors=", char_tensors)

        # def test_stroke_order(self):
        #     data_dir = "data"
        #     stroke_file_name = os.path.join(data_dir, "cns11643_strokeorder.txt")
        #     stroke_order = StrokeOrder()
        #     stroke_order.load_stroke_order(file_name=stroke_file_name)
        #     stroke_order = stroke_order.stroke_order
        #     print(stroke_order["酒"])

        # def test_cn(self):
        #     str_cn="给他以虚妄的想象"
        #     for ch in str_cn:
        #         print(ch)
