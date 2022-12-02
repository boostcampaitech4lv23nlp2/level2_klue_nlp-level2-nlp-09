from typing import Optional

from dataclasses import dataclass, field
from datetime import datetime

from dataclasses_json import DataClassJsonMixin
from marshmallow import fields
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class ModelArgument(YamlDataClassConfig):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = None
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
