from typing import Tuple

import hanja

from . import replace_symbol


def translation(sentence: str, method: str = None) -> str:
    assert method in [
        None,
        "chinese",
    ], "입력하신 method는 없습니다."
    if method is None:
        return sentence

    if method == "chinese":
        return hanja.translate(sentence, "substitution")


def extraction(entity: str) -> dict:
    """
    Args:
        entity (str): subject or object

    Returns:
        Dict[int,int,str,str]: return dict containing entity information
    """
    entity_type = entity[:-1].split(",")[-1].split(":")[1]
    entity_length = len(entity.split(","))
    start_idx = int(entity.split(",", entity_length - 3)[entity_length - 3].split(",")[0].split(":")[1])
    end_idx = int(entity.split(",", entity_length - 3)[entity_length - 3].split(",")[1].split(":")[1])
    entity_word = "".join(entity.split(",", entity_length - 3)[: entity_length - 3]).split(":")[1]
    entity_word = entity_word.replace("'", "").strip()
    entity_type = entity_type.replace("'", "").strip()

    entity_dict = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "entity_type": entity_type,
        "entity_word": entity_word,
    }

    return entity_dict


def unpack_entity_dict(start_idx, end_idx, entity_type, entity_word):
    return start_idx, end_idx, entity_word, entity_type


def entity_representation(subject_dict: dict, object_dict: dict, sentence: str, method: str = None) -> str:
    """
    Args:
        subject (str): subject dictionary
        object (str):  object dictionary
        sentence (str): single sentence
        method (_type_, optional): entity representation. Defaults to None.

    Returns:
        str: single sentence
    """

    assert method in [
        None,
        "entity_mask",
        "entity_marker",
        "entity_marker_punct",
        "typed_entity_marker",
        "typed_entity_marker_punct",
    ], "입력하신 method는 없습니다."

    sub_start_idx, sub_end_idx, subject, subject_entity = unpack_entity_dict(**subject_dict)
    obj_start_idx, obj_end_idx, object, object_entity = unpack_entity_dict(**object_dict)

    # entity representation

    # baseline code
    if method is None:
        temp = subject + " [SEP] " + object + " [SEP] " + sentence

    # entity mask
    elif method == "entity_mask":

        if sub_start_idx < obj_start_idx:
            temp = (
                sentence[:sub_start_idx]
                + f"[SUBJ-{subject_entity}] "
                + sentence[sub_end_idx + 1 : obj_start_idx]
                + f"[OBJ-{object_entity}] "
                + sentence[obj_end_idx + 1 :]
            )
        else:
            temp = (
                sentence[:obj_start_idx]
                + f"[OBJ-{object_entity}] "
                + sentence[obj_end_idx + 1 : sub_start_idx]
                + f"[SUBJ-{subject_entity}] "
                + sentence[sub_end_idx + 1 :]
            )

    # entity marker
    elif method == "entity_marker" or method == "entity_marker_punct":

        if sub_start_idx < obj_start_idx:

            temp = (
                sentence[:sub_start_idx]
                + "[E1] "
                + sentence[sub_start_idx : sub_end_idx + 1]
                + " [/E1] "
                + sentence[sub_end_idx + 1 : obj_start_idx]
                + "[E2] "
                + sentence[obj_start_idx : obj_end_idx + 1]
                + " [/E2] "
                + sentence[obj_end_idx + 1 :]
            )
        else:
            temp = (
                sentence[:obj_start_idx]
                + "[E1] "
                + sentence[obj_start_idx : obj_end_idx + 1]
                + " [/E1] "
                + sentence[obj_end_idx + 1 : sub_start_idx]
                + "[E2] "
                + sentence[sub_start_idx : sub_end_idx + 1]
                + " [/E2] "
                + sentence[sub_end_idx + 1 :]
            )

        # entity marker punct
        if method == "entity_marker_punct":

            temp = temp.replace("[E1]", "@")
            temp = temp.replace("[/E1]", "@")
            temp = temp.replace("[E2]", "#")
            temp = temp.replace("[/E2]", "#")

    # typed entity marker
    elif method == "typed_entity_marker" or method == "typed_entity_marker_punct":
        subject = subject.replace("'", "").upper()
        object = object.replace("'", "").upper()

        if sub_start_idx < obj_start_idx:
            temp = (
                sentence[:sub_start_idx]
                + f"<S:{subject_entity}> "
                + sentence[sub_start_idx : sub_end_idx + 1]
                + f" </S:{subject_entity}> "
                + sentence[sub_end_idx + 1 : obj_start_idx]
                + f"<O:{object_entity}> "
                + sentence[obj_start_idx : obj_end_idx + 1]
                + f" </O:{object_entity}> "
                + sentence[obj_end_idx + 1 :]
            )
        else:
            temp = (
                sentence[:obj_start_idx]
                + f"<O:{object_entity}> "
                + sentence[obj_start_idx : obj_end_idx + 1]
                + f" </O:{object_entity}> "
                + sentence[obj_end_idx + 1 : sub_start_idx]
                + f"<S:{subject_entity}> "
                + sentence[sub_start_idx : sub_end_idx + 1]
                + f" </S:{subject_entity}> "
                + sentence[sub_end_idx + 1 :]
            )

        # typed entity marker punct
        if method == "typed_entity_marker_punct":

            temp = temp.replace(f"<S:{subject_entity}>", f"@ * {subject_entity.lower()} *")
            temp = temp.replace(f"</S:{subject_entity}>", "@")
            temp = temp.replace(f"</O:{object_entity}>", "#")
            temp = temp.replace(f"<O:{object_entity}>", f"# ∧ {object_entity.lower()} ∧")

    return temp


def representation(
    subject: str,
    object: str,
    sentence: str,
    entity_method: str = None,
    translation_methods: list = [None],
    is_replace=False,
) -> str:
    """
    Args:
        subject (str): subject dictionary
        object (str):  object dictionary
        sentence (str): single sentence
        entity_method (str, optional): entity representation. Defaults to None.
        translation_methods (list, optional): translation methods: (None, chinese)
        is_replace (bool, optional) replace symbol methods. Defaults to False.(True, False)

    Returns:
        str: single sentence
    """

    subject_dict = extraction(subject)
    object_dict = extraction(object)

    tmp = entity_representation(subject_dict, object_dict, sentence, method=entity_method)

    for translation_method in translation_methods:
        tmp = translation(tmp, method=translation_method)

    if is_replace:
        tmp = replace_symbol(tmp)

    return tmp
