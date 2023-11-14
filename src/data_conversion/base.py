# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
from typing_extensions import Type
from pathlib import Path
import os
import yaml
from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo

from src.general import LOGGER


def log_or_raise_exc(msg: str, exception: Type[Exception], strict: bool = True):
    if strict:
        raise exception(msg)
    LOGGER.warning(msg)


def validate_path(path: str, directory: bool = False, strict: bool = True):
    path = Path(path)
    if not path.exists():
        log_or_raise_exc(f"Path {path} not found.", FileNotFoundError, strict)
        return False
    if directory:
        if not path.is_dir():
            log_or_raise_exc(f"Path {path} is not a directory", NotADirectoryError, strict)
            return False
    else:
        if not path.is_file():
            log_or_raise_exc(f"Path {path} is not a file", ValueError, strict)
            return False
    return True


class BaseManager:
    def __init__(self):
        self.logger = LOGGER


class BaseArgs(BaseModel):
    @classmethod
    def load_args(cls, yaml_path):
        with open(yaml_path, "r") as file:
            args = yaml.load(file, Loader=yaml.SafeLoader)
        return cls(**args)


class COCOArgs(BaseArgs):
    root: str                           # Data root directory path. Absolute path.
    train_dir: str = 'train'            # Directory containing training images. Relative path to 'root'.
    val_dir: str = 'val'                # Directory containing validation images. Relative path to 'root'.
    test_dir: str = 'test'              # Directory containing test images. Relative path to 'root'.
    anno_dir: str = 'annotations'       # Directory containing annotations. Relative path to 'root'.
    anno_train: str = 'train.json'      # Training images annotation file. Relative path to 'anno_dir'.
    anno_val: str = 'val.json'          # Validation images annotation file. Relative path to 'anno_dir'.
    anno_test: str = ''                 # Test images annotation file. Relative path to 'anno_dir'.

    is_img_same_dir: bool = False       # Whether all images are in same directory
    data_dir: str = ''                  # Optional. Directory containing all images. Relative path to 'root'.
                                        # Valid when is_img_same_dir is True.
    anno_data: str = ''                 # Optional. All images annotation file. Relative path to 'anno_dir'.
    split_ratio: float = 0.8            # Split ratio. Valid when 'anno_data' is set.
    shuffle: bool = False               # Shuffle. Valid when 'anno_data' is set. Shuffle when split data.
    seed: int = 0                       # Random seed used to shuffle data. Valid when 'shuffle' is True.

    @field_validator('root')
    @classmethod
    def root_not_empty(cls, val: str, info: ValidationInfo):
        if not val:
            raise ValidationError("Field 'root' must be given.")
        return val

    def model_post_init(self, __context) -> None:
        self.train_dir = os.path.join(self.root, self.train_dir)
        self.val_dir = os.path.join(self.root, self.val_dir)
        self.test_dir = os.path.join(self.root, self.test_dir)
        self.anno_dir = os.path.join(self.root, self.anno_dir)
        self.data_dir = os.path.join(self.root, self.data_dir)

        self.anno_train = os.path.join(self.anno_dir, self.anno_train)
        self.anno_val = os.path.join(self.anno_dir, self.anno_val)
        self.anno_test = os.path.join(self.anno_dir, self.anno_test)
        self.anno_data = os.path.join(self.anno_dir, self.anno_data)

    def make_dirs(self):
        for folder in (self.root, self.train_dir, self.val_dir, self.test_dir):
            folder = Path(folder)
            if not folder.exists():
                folder.mkdir()


class YOLOArgs(BaseArgs):
    root: str           # Data root directory path. Absolute path.
    anno_train: str     # Directory containing training images. Relative path to 'root'.
    anno_val: str       # Directory containing validation images. Relative path to 'root'.
    anno_test: str      # Directory containing test images. Relative path to 'root'.

    is_img_same_dir: bool = False   # Whether all annotations are in same file.
    anno_data: str = ''     # Optional. All images annotation file. Relative path to 'root'.
    # List including all categories' names, or text file relative path to 'root',
    # which contains categories information. One line for one category.
    names: Union[list, str] = ""
    nc: int = 0             # Number of classes

    @field_validator('root')
    @classmethod
    def root_not_empty(cls, val: str, info: ValidationInfo):
        if not val:
            raise ValidationError("Field 'root' must be given.")
        return val

    def model_post_init(self, __context) -> None:
        self.anno_train = os.path.join(self.root, self.anno_train)
        self.anno_val = os.path.join(self.root, self.anno_val)
        self.anno_test = os.path.join(self.root, self.anno_test)
        self.anno_data = os.path.join(self.root, self.anno_data)
        if isinstance(self.names, str):
            self.names = os.path.join(self.root, self.names)

    def make_dirs(self):
        root = Path(self.root)
        if not validate_path(self.root, strict=False):
            root.mkdir()
        (root / "labels").mkdir()
        (root / "images").mkdir()
        (root / "annotations").mkdir()


class LabelmeArgs(BaseArgs):
    root: str
    train_dir: str
    val_dir: str
    test_dir: str
    is_img_same_dir: bool = False
    data_dir: str

    @field_validator('root')
    @classmethod
    def root_not_empty(cls, val: str, info: ValidationInfo):
        if not val:
            raise ValidationError("Field 'root' must be given.")
        return val

    def model_post_init(self, __context) -> None:
        self.train_dir = os.path.join(self.root, self.train_dir)
        self.val_dir = os.path.join(self.root, self.val_dir)
        self.test_dir = os.path.join(self.root, self.test_dir)
        self.data_dir = os.path.join(self.root, self.data_dir)

    def make_dirs(self):
        for folder in (self.root, self.train_dir, self.val_dir, self.test_dir, self.data_dir):
            folder = Path(folder)
            if not folder.exists():
                folder.mkdir()
