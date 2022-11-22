import unittest

from src.utils.representation import representation

test_objects = [
    [
        "〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.",
        "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}",
        "{'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}",
    ],
    [
        "K리그2에서 성적 1위를 달리고 있는 광주FC는 지난 26일 한국프로축구연맹으로부터 관중 유치 성과와 마케팅 성과를 인정받아 ‘풀 스타디움상’과 ‘플러스 스타디움상’을 수상했다.",
        "{'word': '광주FC', 'start_idx': 21, 'end_idx': 24, 'type': 'ORG'}",
        "{'word': '한국프로축구연맹', 'start_idx': 34, 'end_idx': 41, 'type': 'ORG'}",
    ],
    [
        "균일가 생활용품점 (주)아성다이소(대표 박정부)는 코로나19 바이러스로 어려움을 겪고 있는 대구광역시에 행복박스를 전달했다고 10일 밝혔다.",
        "{'word': '아성다이소', 'start_idx': 13, 'end_idx': 17, 'type': 'ORG'}",
        "{'word': '박정부', 'start_idx': 22, 'end_idx': 24, 'type': 'PER'}",
    ],
    [
        "백한성(白漢成, 水原鶴人, 1899년 6월 15일 조선 충청도 공주 출생 ~ 1971년 10월 13일 대한민국 서울에서 별세.)은 대한민국의 정치가이며 법조인이다.",
        "{'word': '백한성', 'start_idx': 0, 'end_idx': 2, 'type': 'PER'}",
        "{'word': '조선 충청도 공주', 'start_idx': 28, 'end_idx': 36, 'type': 'LOC'}",
    ],
    [
        "KBS 전주방송총국(KBS 全州放送總局)은 전라북도 지역을 대상으로 하는 한국방송공사의 지역 방송 총국이다.",
        "{'word': 'KBS 전주방송총국', 'start_idx': 0, 'end_idx': 9, 'type': 'ORG'}",
        "{'word': 'KBS 全州放送總局', 'start_idx': 11, 'end_idx': 20, 'type': 'ORG'}",
    ],
]

none_answers = [
    "비틀즈 [SEP]조지 해리슨 [SEP] 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.",
    "광주FC [SEP]한국프로축구연맹 [SEP] K리그2에서 성적 1위를 달리고 있는 광주FC는 지난 26일 한국프로축구연맹으로부터 관중 유치 성과와 마케팅 성과를 인정받아 ‘풀 스타디움상’과 ‘플러스 스타디움상’을 수상했다.",
    "아성다이소 [SEP]박정부 [SEP] 균일가 생활용품점 (주)아성다이소(대표 박정부)는 코로나19 바이러스로 어려움을 겪고 있는 대구광역시에 행복박스를 전달했다고 10일 밝혔다.",
    "백한성 [SEP]조선 충청도 공주 [SEP] 백한성(白漢成, 水原鶴人, 1899년 6월 15일 조선 충청도 공주 출생 ~ 1971년 10월 13일 대한민국 서울에서 별세.)은 대한민국의 정치가이며 법조인이다.",
    "KBS 전주방송총국 [SEP]KBS 全州放送總局 [SEP] KBS 전주방송총국(KBS 全州放送總局)은 전라북도 지역을 대상으로 하는 한국방송공사의 지역 방송 총국이다.",
]


chinese_answers = [
    "비틀즈 [SEP]조지 해리슨 [SEP] 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.",
    "광주FC [SEP]한국프로축구연맹 [SEP] K리그2에서 성적 1위를 달리고 있는 광주FC는 지난 26일 한국프로축구연맹으로부터 관중 유치 성과와 마케팅 성과를 인정받아 ‘풀 스타디움상’과 ‘플러스 스타디움상’을 수상했다.",
    "아성다이소 [SEP]박정부 [SEP] 균일가 생활용품점 (주)아성다이소(대표 박정부)는 코로나19 바이러스로 어려움을 겪고 있는 대구광역시에 행복박스를 전달했다고 10일 밝혔다.",
    "백한성 [SEP]조선 충청도 공주 [SEP] 백한성(백한성, 수원학인, 1899년 6월 15일 조선 충청도 공주 출생 ~ 1971년 10월 13일 대한민국 서울에서 별세.)은 대한민국의 정치가이며 법조인이다.",
    "KBS 전주방송총국 [SEP]KBS 전주방송총국 [SEP] KBS 전주방송총국(KBS 전주방송총국)은 전라북도 지역을 대상으로 하는 한국방송공사의 지역 방송 총국이다.",
]


class RepresentationTester(unittest.TestCase):
    def test_none(self):
        for example_object, answer in zip(test_objects, none_answers):
            sentence, subject, object = example_object
            generate_text = representation(subject, object, sentence, entity_method=None)
            self.assertEqual(generate_text, answer)

    def test_chinese(self):
        for example_object, answer in zip(test_objects, chinese_answers):
            sentence, subject, object = example_object
            generate_text = representation(
                subject, object, sentence, entity_method=None, translation_methods=["chinese"]
            )
            self.assertEqual(generate_text, answer)
