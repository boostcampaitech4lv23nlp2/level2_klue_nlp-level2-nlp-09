def entity_representation(subject: str, object: str, sentence: str, method: str = None) -> str:
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

    # subject object extraction

    subject_entity = subject[:-1].split(",")[-1].split(":")[1]
    object_entity = object[:-1].split(",")[-1].split(":")[1]

    subject_length = len(subject.split(","))
    object_length = len(object.split(","))

    sub_start_idx = int(subject.split(",", subject_length - 3)[subject_length - 3].split(",")[0].split(":")[1])
    sub_end_idx = int(subject.split(",", subject_length - 3)[subject_length - 3].split(",")[1].split(":")[1])
    subject = "".join(subject.split(",", subject_length - 3)[: subject_length - 3]).split(":")[1]

    obj_start_idx = int(object.split(",", object_length - 3)[object_length - 3].split(",")[0].split(":")[1])
    obj_end_idx = int(object.split(",", object_length - 3)[object_length - 3].split(",")[1].split(":")[1])
    object = "".join(object.split(",", object_length - 3)[: object_length - 3]).split(":")[1]

    subject_entity = subject_entity.replace("'", "").strip()
    object_entity = object_entity.replace("'", "").strip()

    # entity representation

    # baseline code
    if method is None:
        temp = subject + " [SEP]" + object + " [SEP] " + sentence

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
