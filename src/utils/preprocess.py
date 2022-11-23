import re

symbol_map = {"\"`'‘‘’“”＇ˈ′": "'"}


def remove_symbol():
    pass


def replace_symbol(sentence):
    for key, value in symbol_map.items():
        sentence = re.sub(f"[{key}]", value, sentence)
    return sentence


def remove_language():
    pass
