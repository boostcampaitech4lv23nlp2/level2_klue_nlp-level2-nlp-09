import os
import pickle as pickle


def label_to_num(label):
    num_label = []
    dict_label_to_num = {
        "org:top_members/employees": 0,
        "org:members": 1,
        "org:product": 2,
        "per:title": 3,
        "org:alternate_names": 4,
        "per:employee_of": 5,
        "org:place_of_headquarters": 6,
        "per:product": 7,
        "org:number_of_employees/members": 8,
        "per:children": 9,
        "per:place_of_residence": 10,
        "per:alternate_names": 11,
        "per:other_family": 12,
        "per:colleagues": 13,
        "per:origin": 14,
        "per:siblings": 15,
        "per:spouse": 16,
        "org:founded": 17,
        "org:political/religious_affiliation": 18,
        "org:member_of": 19,
        "per:parents": 20,
        "org:dissolved": 21,
        "per:schools_attended": 22,
        "per:date_of_death": 23,
        "per:date_of_birth": 24,
        "per:place_of_birth": 25,
        "per:place_of_death": 26,
        "org:founded_by": 27,
        "per:religion": 28,
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
        0: "org:top_members/employees",
        1: "org:members",
        2: "org:product",
        3: "per:title",
        4: "org:alternate_names",
        5: "per:employee_of",
        6: "org:place_of_headquarters",
        7: "per:product",
        8: "org:number_of_employees/members",
        9: "per:children",
        10: "per:place_of_residence",
        11: "per:alternate_names",
        12: "per:other_family",
        13: "per:colleagues",
        14: "per:origin",
        15: "per:siblings",
        16: "per:spouse",
        17: "org:founded",
        18: "org:political/religious_affiliation",
        19: "org:member_of",
        20: "per:parents",
        21: "org:dissolved",
        22: "per:schools_attended",
        23: "per:date_of_death",
        24: "per:date_of_birth",
        25: "per:place_of_birth",
        26: "per:place_of_death",
        27: "org:founded_by",
        28: "per:religion",
    }
    for v in label:
        origin_label.append(dict_num_to_label[v])
    return origin_label
