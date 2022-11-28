import os
import pickle as pickle


def label_to_num(label):
    num_label = []
    dict_label_to_num = {
        "no_relation": 0,
        "relation": 1,
    }
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    dict_num_to_label = {
        0: "no_relation",
        1: "relation",
    }
    for v in label:
        origin_label.append(dict_num_to_label[v])
    return origin_label
