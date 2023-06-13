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

import glob
import math
import os
import random
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import cv2
import psutil
import numpy as np
from tqdm import tqdm

from src.augmentations import (Albumentations, augment_hsv, copy_paste,
                               letterbox, load_image, load_samples, mixup,
                               pastein, random_perspective)
from src.general import xyn2xy, xywhn2xyxy, xyxy2xywhn, empty
from src.dataset.common import (HELP_URL, IMG_FORMATS, VID_FORMATS, PIN_MEMORY,
                                TQDM_BAR_FORMAT, NUM_THREADS, IMG_TUPLE)
from src.dataset.common import img2label_paths, get_hash, verify_image_label
from src.dataset.dataset import Dataset


class Dataloader:
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 16,
                 img_size: int = 640,
                 cache_images: str = '',
                 rect: bool = False,
                 augment: bool = False,
                 single_cls: bool = False,
                 min_items: int = 0,
                 stride: int = 32,
                 pad: float = 0.0,
                 rank: int = 0,
                 prefix: str = ''):
        self.dataset = dataset
        self.batch_size = batch_size
        self.img_size = img_size
        self.cache_images = cache_images
        self.rect = rect
        self.augment = augment
        self.single_cls = single_cls
        self.min_items = min_items
        self.stride = stride
        self.pad = pad
        self.rank = rank
        self.prefix = prefix
        self.albumentations = Albumentations(size=img_size) if augment else None
        self.filter_images(min_items)
        self.create_indices()
        self.update_labels(single_cls)

    def filter_images(self, min_items: int = 0):
        # Filter images
        dataset = self.dataset
        cache_result = dataset.cache.get_cache_result()
        n_total = cache_result.n_total
        if min_items > 0:
            include = np.array([len(x) >= min_items for x in dataset.labels]).nonzero()[0].astype(int)
            print(f'{self.prefix}{n_total - len(include)}/{n_total} images filtered from dataset', flush=True)
            dataset.img_files = [dataset.img_files[i] for i in include]
            dataset.label_files = [dataset.label_files[i] for i in include]
            dataset.labels = [dataset.labels[i] for i in include]
            dataset.segments = [dataset.segments[i] for i in include]
            dataset.shapes = dataset.shapes[include]  # wh

    def create_indices(self):
        # Create indices
        dataset = self.dataset
        n_images = len(dataset.shapes)  # number of images
        batch_index = np.floor(np.arange(n_images) / self.batch_size).astype(int)  # batch index
        self.n_batches = batch_index[-1] + 1  # number of batches
        self.batch_index = batch_index  # batch index of image
        self.n_images = n_images
        self.indices = range(n_images)

    def update_labels(self, single_cls: bool = False):
        # Update labels
        dataset = self.dataset
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(dataset.labels, dataset.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                dataset.labels[i] = label[j]
                if segment:
                    dataset.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                dataset.labels[i][:, 0] = 0

    def rectangular_training(self):
        # Rectangular Training
        dataset = self.dataset
        if self.rect:
            # Sort by aspect ratio
            shapes = dataset.shapes  # wh
            aspect_ratios = shapes[:, 1] / shapes[:, 0]  # aspect ratio
            irect = aspect_ratios.argsort()
            dataset.img_files = [dataset.img_files[i] for i in irect]
            dataset.label_files = [dataset.label_files[i] for i in irect]
            dataset.labels = [dataset.labels[i] for i in irect]
            dataset.segments = [dataset.segments[i] for i in irect]
            dataset.shapes = shapes[irect]  # wh
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

            self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(int) \
                                * self.stride

    def check_cache_ram(self, safety_margin=0.1, prefix=''):
        # Check image caching requirements vs available memory
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.dataset.img_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            print(f"{prefix}{mem_required / gb:.1f}GB RAM required, "
                  f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                  f"{'caching images ✅' if cache else 'not caching images ⚠️'}", flush=True)
        return cache

    def load_images_to_cache(self):
        # Cache images into RAM/disk for faster training
        cache_images = True
        if self.cache_images == 'ram' and not self.check_cache_ram(prefix=self.prefix):
            cache_images = False
        n_total = len(self.dataset.shapes)  # number of images
        self.ims = [None] * n_total
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.dataset.img_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n_total, [None] * n_total
            fcn = self.cache_images_to_disk if self.cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n_total))
            pbar = tqdm(enumerate(results), total=n_total, bar_format=TQDM_BAR_FORMAT, disable=self.rank > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.dataset.img_files[i]))

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.dataset.img_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized
