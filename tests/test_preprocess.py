import unittest

from src.utils.preprocess import replace_symbol

test_sentences = [
    "비틀즈 [SEP]조지 해리슨 [SEP] 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.",
    '““비"틀‘‘즈 [SEPˈ]조지 해＇리ˈ"슨 [SEP] ’〈Some′thin‘‘g〉는＇＇ 조지 해ˈ′리“슨”이 ’쓰ˈ고 ＇＇비"틀′즈가 1969년“ 앨범 《′Abbey Road》에 ”ˈ담은 "노래다.',
]

test_answers = [
    "비틀즈 [SEP]조지 해리슨 [SEP] 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.",
    "''비'틀''즈 [SEP']조지 해'리''슨 [SEP] '〈Some'thin''g〉는'' 조지 해''리'슨'이 '쓰'고 ''비'틀'즈가 1969년' 앨범 《'Abbey Road》에 ''담은 '노래다.",
]


class PreprocessTester(unittest.TestCase):
    def test_replace_symbol(self):
        for sentence, answer in zip(test_sentences, test_answers):
            generate_sentence = replace_symbol(sentence)
            self.assertEqual(generate_sentence, answer)
