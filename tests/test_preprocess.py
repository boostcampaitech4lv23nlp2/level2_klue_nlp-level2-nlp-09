import unittest

from src.utils.preprocess import replace_symbol

test_quote_symbol_map = {"\"`'‘‘’“”＇ˈ′": "'"}
test_bracket_symbol_map = {"[[": "<", "]]": ">", "\[《〈「˹｢⟪≪<⌜『«": "<", "\]》〉」˼｣⟫≫>⌟»": ">", "（{": "(", "）}": ")"}
test_sentences = [
    "비틀즈 [SEP] 조지 해리슨 [SEP] 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.{}",
]

test_quote_answers = [
    "비틀즈 [SEP] 조지 해리슨 [SEP] 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.{}",
]

test_bracket_answers = [
    "비틀즈 [SEP] 조지 해리슨 [SEP] <Something>는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 <Abbey Road>에 담은 노래다.()",
]


class PreprocessTester(unittest.TestCase):
    def test_replace_symbol(self):
        for sentence, answer in zip(test_sentences, test_quote_answers):
            generate_sentence = replace_symbol(sentence, test_quote_symbol_map)
            self.assertEqual(generate_sentence, answer)

    def test_bracket_symbol(self):
        for sentence, answer in zip(test_sentences, test_bracket_answers):
            generate_sentence = replace_symbol(sentence, test_bracket_symbol_map)
            self.assertEqual(generate_sentence, answer)
