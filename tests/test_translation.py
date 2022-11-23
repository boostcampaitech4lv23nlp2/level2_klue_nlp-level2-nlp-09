import unittest

from src.utils.representation import translation

test_sentences = [
    "백한성(白漢成, 水原鶴人, 1899년 6월 15일 조선 충청도 공주 출생 ~ 1971년 10월 13일 대한민국 서울에서 별세.)은 대한민국의 정치가이며 법조인이다.",
    '1904년 7월 1일, ""툰-운트 스포트버라인 바이어 04 레버쿠젠"" (Turn- und Spielverein Bayer 04 Leverkusen)의 이름으로 창단되었다.',
    "헌강왕(憲康王, ~ 886년, 재위: 875년 ~ 886년)은 신라의 제49대 왕이다.",
    "쇼니 씨(少弐氏)의 8대 당주로 쇼니 요리히사(少弐頼尚)의 둘째 아들이다.",
    "버턴 릭터(Burton Richter, 1931년 3월 22일 ~ 2018년 7월 18일)는 노벨 물리학상을 받은 미국의 물리학자이다.",
    "유한굉(劉漢宏, Liu Hanhong, ~ 887년)은 중국 당나라 말기에 활약했던 군벌로, 당초에는 당나라에 반기를 들었으나, 후에 당나라의 관직을 받고 의승군 절도사(義勝軍節度使, 본거지는 지금의 저장 성 사오싱 시)로서 절강 동부 일대를 지배하였다.",
]

chiense_sentences = [
    "백한성(백한성, 수원학인, 1899년 6월 15일 조선 충청도 공주 출생 ~ 1971년 10월 13일 대한민국 서울에서 별세.)은 대한민국의 정치가이며 법조인이다.",
    '1904년 7월 1일, ""툰-운트 스포트버라인 바이어 04 레버쿠젠"" (Turn- und Spielverein Bayer 04 Leverkusen)의 이름으로 창단되었다.',
    "헌강왕(헌강왕, ~ 886년, 재위: 875년 ~ 886년)은 신라의 제49대 왕이다.",
    "쇼니 씨(소이씨)의 8대 당주로 쇼니 요리히사(소이뢰상)의 둘째 아들이다.",
    "버턴 릭터(Burton Richter, 1931년 3월 22일 ~ 2018년 7월 18일)는 노벨 물리학상을 받은 미국의 물리학자이다.",
    "유한굉(유한굉, Liu Hanhong, ~ 887년)은 중국 당나라 말기에 활약했던 군벌로, 당초에는 당나라에 반기를 들었으나, 후에 당나라의 관직을 받고 의승군 절도사(의승군절도사, 본거지는 지금의 저장 성 사오싱 시)로서 절강 동부 일대를 지배하였다.",
]


class TranlsationTester(unittest.TestCase):
    def test_run(self):
        pass

    def test_chinese(self):
        for sentence, answer in zip(test_sentences, chiense_sentences):
            translation_sentence = translation(sentence, method="chinese")
            self.assertEqual(translation_sentence, answer)
