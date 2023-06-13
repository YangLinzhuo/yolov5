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

from collections import namedtuple
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from tqdm import tqdm

from src.dataset.common import TQDM_BAR_FORMAT, NUM_THREADS, HELP_URL
from src.dataset.common import load_images_and_labels, get_hash, verify_image_label


CacheResult = namedtuple("CacheResult", ["n_found", "n_missing", "n_empty", "n_corrupt", "n_total"])


class Cache:
    # Cache for YOLOv5 train_loader/val_loader
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self,
                 data_path: str,
                 img_files: Optional[List[str]] = None,
                 label_files: Optional[List[str]] = None,
                 augment: bool = False,
                 prefix: str = ''):
        self.path = data_path
        self.prefix = prefix
        self.augment = augment
        if img_files is not None and label_files is not None:
            self.img_files, self.label_files = img_files, label_files
        else:
            self.img_files, self.label_files = load_images_and_labels(data_path, prefix)
        self.cache_path = self.get_cache_path()
        try:
            cache, exists = np.load(self.cache_path, allow_pickle=True).item(), True  # load dict
            try:
                assert cache['version'] == self.cache_version  # matches current version
                assert cache['hash'] == get_hash(self.label_files + self.img_files)  # identical hash
            except Exception as e:
                exit(f"[ERROR] {e}, please remove cache file in dataset path: rm -f {Path(self.path).parent}/*.cache*")
        except Exception:
            cache, exists = self.cache_labels(Path(self.cache_path), prefix), False  # run cache ops
        self.cache = cache
        self.exists = exists
        self.display_cache()

    def get_cache_path(self) -> str:
        path = self.path
        p = path if not isinstance(path, list) else path[0]
        p = Path(p)
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        return str(cache_path)

    def cache_labels(self, path: Path = Path('./labels.cache'), prefix: str = '') -> Dict[str, Any]:
        # Cache dataset labels, check images and read shapes
        x: Dict[str, Any] = {}  # dict
        img_infos: Dict[str, Any] = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.img_files),
                        bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file is not None:
                    img_infos[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            print('\n'.join(msgs), flush=True)
        if nf == 0:
            print(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}', flush=True)
        x['img_infos'] = img_infos
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = CacheResult(nf, nm, ne, nc, len(self.img_files))
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(str(path.resolve()), x, allow_pickle=True)  # type: ignore # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            print(f'{prefix}New cache created: {path}', flush=True)
        except Exception as e:
            print(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}',
                  flush=True)  # not writeable
        return x

    def display_cache(self):
        # Display cache
        result: CacheResult = self.get_cache_result()
        if self.exists:
            d = f"Scanning {self.cache_path}... {result.n_found} images, " \
                f"{result.n_missing + result.n_empty} backgrounds, {result.n_corrupt} corrupt"
            # display cache results
            tqdm(None, desc=self.prefix + d, total=result.n_total, initial=result.n_total, bar_format=TQDM_BAR_FORMAT)
            if self.cache.get('msg', None) is not None:
                msg = self.cache.get('msg', '')
                if msg:
                    print('\n'.join(self.cache['msgs']), flush=True)  # display warnings
        assert result.n_found > 0 or not self.augment, f'{self.prefix}No labels found in {self.cache_path}, ' \
                                           f'can not start training. {HELP_URL}'

    def get_cache_result(self) -> CacheResult:
        # found, missing, empty, corrupt, total
        return self.cache['result']
