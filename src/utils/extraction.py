from typing import Tuple


def extraction(subject: str, object: str) -> Tuple[int, int, int, str, str, str, str, str]:
    """
    Args:
        subject (str): subject
        object (str): object

    Returns:
        Tuple[int,int,int,str,str,str,str,str]: return subject object idx or subject object
    """
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

    return sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx, subject, object, subject_entity, object_entity
