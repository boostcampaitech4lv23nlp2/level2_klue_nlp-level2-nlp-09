import re

default_symbol_map = {
    "\"`'‘‘’“”＇ˈ′＇": "'",
    "\[\[": "<",
    "\]\]": ">",
    "[\[《〈「˹｢⟪≪<⌜『«]": "<",
    "[\]》〉」˼｣⟫≫>⌟»]": ">",
    "[（{]": "(",
    "[）}]": ")",
    "[？]": "?",
    "[㎿]": "mw",
    "\(주\)": "㈜",
    "[⅓⅔³]": "",
    "[，]": ",",
    "[１]": "1",
    "[～∼]": "~",
    "[▲△▴▵•°]": "▲",
    "[。]": ".",
    "[：]": ":",
    "[∙⋅∙⋅·‧ㆍ⸱･]": "·",
    "[–]": "-",
    "[―—]": " ",
    "[_]": " ",
    "[％]": "%",
    "\(;": "(",
    "[／]": "/",
    "[：]": ":",
    "[★☆]": "",
    "[®]": "",
    "[☎]": "",
    "[\u0400-\u04FF]": "",
}


def remove_symbol():
    pass


def replace_symbol(sentence, custom_symbol_map=None):
    symbol_map = default_symbol_map
    if custom_symbol_map:
        symbol_map = custom_symbol_map

    for key, value in symbol_map.items():
        sentence = re.sub(key, value, sentence)
    sentence = re.sub("(<SEP>)", "[SEP]", sentence)
    return sentence


def remove_language():
    pass
