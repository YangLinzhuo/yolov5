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
import ctypes
import math
import random
import psutil
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
from tqdm import tqdm

from src.dataset.common import TQDM_BAR_FORMAT, NUM_THREADS
from src.dataset.common import load_images_and_labels, HELP_URL, img2label_paths
from src.dataset.cache import Cache


class Dataset:
    def __init__(self,
                 data_path: str,
                 batch_size: int = 16,
                 img_size: int = 640,
                 cache_images: str = '',
                 rect: bool = False,
                 augment: bool = False,
                 single_cls: bool = False,
                 min_items: int = 0,
                 stride: int = 32,
                 pad: float = 0.0,
                 prefix: str = ''):
        self.batch_size = batch_size
        self.img_size = img_size
        self.cache_images = cache_images
        self.rect = rect
        self.augment = augment
        self.single_cls = single_cls
        self.min_items = min_items
        self.prefix = prefix
        self.stride = stride
        self.pad = pad
        self.is_cache = False
        self.cache_ready = False
        self.img_files, self.label_files = load_images_and_labels(data_path, prefix)
        self.cache = Cache(data_path, augment=augment,
                           img_files=self.img_files, label_files=self.label_files, prefix=prefix)
        # Update
        # type of labels:  tuple(Nx5 ndarray)
        # type of shapes: tuple(tuple(int, int))
        img_infos: Dict[str, Any] = self.cache.cache['img_infos']
        labels, shapes, self.segments = zip(*img_infos.values())
        n_labels = len(np.concatenate(labels, 0))  # number of labels
        assert n_labels > 0 or not augment, f'{prefix}All labels empty in {self.cache.cache_path}, ' \
                                      f'can not start training. {HELP_URL}'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.img_files = list(img_infos.keys())  # update
        self.label_files = img2label_paths(self.img_files)  # update

        self.filter_images(min_items)
        self.create_indices(batch_size)
        self.update_labels(single_cls)
        self.rectangular_training(rect, img_size, stride, pad)
        self.load_images_to_cache(cache_images)


    def filter_images(self, min_items: int = 0):
        # Filter images
        cache_result = self.cache.get_cache_result()
        n_total = cache_result.n_total
        if min_items > 0:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            print(f'{self.prefix}{n_total - len(include)}/{n_total} images filtered from self', flush=True)
            self.img_files = [self.img_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

    def create_indices(self, batch_size: int = 16):
        # Create indices
        n_images = len(self.shapes)  # number of images
        batch_index = np.floor(np.arange(n_images) / batch_size).astype(int)  # batch index
        self.n_batches = batch_index[-1] + 1  # number of batches
        self.batch_index = batch_index  # batch index of image
        self.n_images = n_images
        self.indices = range(n_images)

    def update_labels(self, single_cls: bool = False):
        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

    def rectangular_training(self, rect: bool = False, img_size: int = 640, stride: int = 32, pad: float = 0.0):
        # Rectangular Training
        if rect:
            # Sort by aspect ratio
            shapes = self.shapes  # wh
            aspect_ratios = shapes[:, 1] / shapes[:, 0]  # aspect ratio
            irect = aspect_ratios.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = shapes[irect]  # wh
            aspect_ratios = aspect_ratios[irect]

            # Set training image shapes
            shapes = [[1, 1]] * self.n_batches
            for i in range(self.n_batches):
                ari = aspect_ratios[self.batch_index == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

    def check_cache_ram(self, safety_margin: float = 0.1, prefix: str = ''):
        # Check image caching requirements vs available memory
        b, gb = 0., 1 << 30  # bytes of cached images, bytes per gigabytes
        n_total = self.cache.get_cache_result().n_total
        n = min(n_total, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.img_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * n_total / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            print(f"{prefix}{mem_required / gb:.1f}GB RAM required, "
                  f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                  f"{'caching images ✅' if cache else 'not caching images ⚠️'}", flush=True)
        return cache

    def load_images_to_cache(self, cache_images: str = '', prefix: str = ''):
        # Cache images into RAM/disk for faster training
        is_cache = cache_images != ''
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            is_cache = False
        n_total = len(self.shapes)  # number of images
        self.imgs: List[Optional[np.ndarray]] = [None] * n_total
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.img_files]
        if is_cache:
            self.is_cache = True
            if cache_images == 'ram':
                # Create share memory object for images
                import multiprocessing
                img_channel = 3
                imgs_share_base = multiprocessing.Array(ctypes.c_uint8,
                                                   n_total * self.img_size * self.img_size * img_channel,
                                                   lock=False)
                imgs_hw0_share_base = multiprocessing.Array(ctypes.c_int32,
                                                       n_total * 3,    # (height, width, channel)
                                                       lock=False)
                imgs_hw_share_base = multiprocessing.Array(ctypes.c_int32,
                                                      n_total * 3,     # (height, width, channel)
                                                      lock=False)
                self.imgs_share = np.frombuffer(imgs_share_base, dtype=ctypes.c_uint8)
                self.imgs_share = self.imgs_share.reshape((n_total, self.img_size * self.img_size * img_channel))
                self.imgs_hw0_share = np.frombuffer(imgs_hw0_share_base, dtype=ctypes.c_int32)
                self.imgs_hw0_share = self.imgs_hw0_share.reshape((n_total, 3))
                self.imgs_hw_share = np.frombuffer(imgs_hw_share_base, dtype=ctypes.c_int32)
                self.imgs_hw_share = self.imgs_hw_share.reshape((n_total, 3))
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.img_hw0, self.img_hw = [None] * n_total, [None] * n_total
            fcn = self.cache_images_to_disk if self.cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n_total))
            pbar = tqdm(enumerate(results), total=n_total, bar_format=TQDM_BAR_FORMAT)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    # self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    img, img_hw0, img_hw = x
                    self.imgs_share[i][:img.nbytes] = img.flatten()
                    self.imgs_hw0_share[i] = (*img_hw0, 3)
                    self.imgs_hw_share[i] = (*img_hw, 3)
                    # b += self.imgs[i].nbytes
                    b += img.nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()
            self.cache_ready = True

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.img_files[i]))

    def load_image(self, i):
        if self.is_cache and self.cache_images == 'ram' and self.cache_ready:
            # load from shared memory
            img_hw0: np.ndarray = self.imgs_hw0_share[i]
            img_hw: np.ndarray = self.imgs_hw_share[i]
            nbytes = np.prod(img_hw)
            img: np.ndarray = self.imgs_share[i][:nbytes]
            img = img.reshape(tuple(img_hw))
            return img, tuple(img_hw0)[:2], tuple(img_hw)[:2]
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        img, img_file, npy_file = self.imgs[i], self.img_files[i], self.npy_files[i]
        if img is None:  # not cached in RAM
            if npy_file.exists():  # load npy
                img = np.load(str(npy_file.resolve()))
            else:  # read image
                img = cv2.imread(img_file)  # BGR
                assert img is not None, f'Image Not Found {img_file}'
            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                img = cv2.resize(img, (round(w0 * r), round(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # im, hw_original, hw_resized
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized
