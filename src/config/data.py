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

import yaml
from pathlib import Path
from typing import Dict, Any


class DatasetConfig:
    def __init__(self, cfg_path: str):
        with open(cfg_path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        self.dataset_cfg = self.process_dataset_cfg(data_dict)
        self.dataset_name = self.dataset_cfg['dataset_name']
        self.root = self.dataset_cfg['root']
        self.train = self.dataset_cfg['train']
        self.val = self.dataset_cfg['val']
        self.test = self.dataset_cfg['test']
        self.num_cls = self.dataset_cfg['nc']
        self.cls_names = self.dataset_cfg['names']

    def process_dataset_cfg(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data_dict, dict):
            return data_dict

        def _join(root, sub):
            if not isinstance(root, str) or not isinstance(sub, str):
                return False
            root = Path(root)
            sub = Path(sub)
            joined_path = sub
            if not sub.parent.samefile(root):
                joined_path = root / sub
            return str(joined_path.resolve())

        data_dict['train'] = _join(data_dict['root'], data_dict['train'])
        data_dict['val'] = _join(data_dict['root'], data_dict['val'])
        data_dict['test'] = _join(data_dict['root'], data_dict['test'])
        return data_dict
