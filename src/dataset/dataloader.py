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

import math
import random
from typing import Union

import cv2
import numpy as np

from mindspore.dataset import vision
from mindspore.dataset.vision import Inter

from src.augmentations import (Albumentations, augment_hsv, copy_paste,
                               letterbox, load_image, load_samples, mixup,
                               pastein, random_perspective)
from src.general import xyn2xy, xywhn2xyxy, xyxy2xywhn, empty
from src.dataset.common import IMG_TUPLE
from src.dataset.dataset import Dataset
from src.config.args import TrainConfig, EvalConfig
from src.config.hyp import Hyp


class LoadImagesAndLabels:
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    def __init__(self,
                 dataset: Dataset,
                 hyp: Hyp,
                 opt: Union[TrainConfig, EvalConfig],
                 image_weights: bool = False):
        self.dataset = dataset
        self.hyp = hyp
        self.opt: Union[TrainConfig, EvalConfig] = opt
        if isinstance(opt.img_size, list):
            self.img_size = opt.img_size[0]
        else:
            self.img_size = opt.img_size
        self.augment = self.opt.augment
        self.image_weights = image_weights
        self.rect = False if image_weights else opt.rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        self.albumentations = Albumentations(size=self.img_size) if self.augment else None

    def __len__(self):
        return len(self.dataset.img_files)

    def __getitem__(self, index):
        dataset = self.dataset
        index = dataset.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp.mosaic
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = np.zeros((3, 2))

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp.mixup:
                img, labels = mixup(img, labels, * self.load_mosaic(random.randint(0, dataset.n_images - 1)))
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            # final letterboxed shape
            shape = dataset.batch_shapes[dataset.batch_index[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = np.array([[h0, w0],
                               [h / h0, w / w0],
                               [pad[0], pad[1]]])  # (3, 2), for COCO mAP rescaling

            labels = dataset.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp.degrees,
                                                 translate=hyp.translate,
                                                 scale=hyp.scale,
                                                 shear=hyp.shear,
                                                 perspective=hyp.perspective)

        num_labels = len(labels)  # number of labels
        if num_labels:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Augment imagespace
            img, labels = self.albumentations(img, labels)
            num_labels = len(labels)  # number of labels

            # Augment colorspace
            augment_hsv(img, hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v)

            # Flip up-down
            if random.random() < hyp.flipud:
                img = np.flipud(img)
                if num_labels:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp.fliplr:
                img = np.fliplr(img)
                if num_labels:
                    labels[:, 1] = 1 - labels[:, 1]

            if random.random() < hyp.paste_in:
                sample_labels, sample_images, sample_masks = [], [], []
                while len(sample_labels) < 30:
                    sample_labels_, sample_images_, sample_masks_ = \
                        load_samples(self, random.randint(0, len(dataset.labels) - 1))
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    if empty(sample_labels):
                        break
                labels = pastein(img, labels, sample_labels, sample_images, sample_masks)

        _labels_out = np.zeros((num_labels, 6))
        if num_labels:
            _labels_out[:, 1:] = labels

        # create fixed label, avoid dynamic shape problem.
        labels_out = np.full((self.hyp.max_box_per_img, 6), -1, dtype=np.float32)
        if num_labels:
            labels_out[:min(num_labels, self.hyp.max_box_per_img), :] = \
                _labels_out[:min(num_labels, self.hyp.max_box_per_img), :]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return IMG_TUPLE(img, labels_out, dataset.img_files[index], shapes)

    def load_mosaic(self, index):
        dataset = self.dataset
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(dataset.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            # Load image
            img, _, (h, w) = dataset.load_image(idx)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = dataset.labels[idx].copy(), dataset.segments[idx].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp.copy_paste)
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp.degrees,
                                           translate=self.hyp.translate,
                                           scale=self.hyp.scale,
                                           shear=self.hyp.shear,
                                           perspective=self.hyp.perspective,
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    @staticmethod
    def collate_fn(img, label, path, shapes, batch_info):
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return IMG_TUPLE(
            np.stack(img, 0).astype(np.float32),
            np.stack(label, 0).astype(np.float32),
            path,
            np.stack(shapes, 0)
        )

    @staticmethod
    def collate_fn4(img, label, path, shapes, batch_info):
        n = len(img) // 4
        img4, label4, path4, _ = [], [], path[:n], shapes[:n]

        ho = np.array([[0., 0, 0, 1, 0, 0]])
        wo = np.array([[0., 0, 1, 0, 0, 0]])
        s = np.array([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:

                _resize_shape = (img[i].shape[1] * 2, img[i].shape[2] * 2)
                # (c,h,w) -> (h,w,c) -> (c,h,w)
                im = vision.Resize(_resize_shape, Inter.BILINEAR)(img[i].transpose(1, 2, 0)).transpose(2, 0, 1)
                # im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear',
                #      align_corners=False)[0].type(img[i].type())
                l = label[i]
            else:
                im = np.concatenate((np.concatenate((img[i], img[i + 1]), 1),
                                     np.concatenate((img[i + 2], img[i + 3]), 1)), 2)
                l = np.concatenate((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return np.stack(img4, 0).astype(np.float32), np.stack(label4, 0).astype(np.float32), path4

def create_dataloader(dataset: Dataset,
                      hyp: Hyp,
                      opt: Union[TrainConfig, EvalConfig],
                      shuffle: bool = True,
                      image_weights: bool = False,
                      num_parallel_workers: int = 8,
                      drop_remainder: bool = True,
                      is_training: bool = True):
    import mindspore.dataset as de
    if opt.rect and shuffle:
        print('[WARNING] --rect is incompatible with DataLoader shuffle, setting shuffle=False', flush=True)
        shuffle = False
        # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    cv2.setNumThreads(2)
    de.config.set_seed(1236517205 + opt.rank)
    dataset = LoadImagesAndLabels(dataset, hyp=hyp, opt=opt, image_weights=image_weights)
    dataset_column_names = ["img", "label_out", "img_files", "shapes"]
    print(f"[INFO] Num parallel workers: [{num_parallel_workers}]", flush=True)
    if opt.rank_size > 1:
        ds = de.GeneratorDataset(dataset, column_names=dataset_column_names,
                                 num_parallel_workers=num_parallel_workers, shuffle=shuffle,
                                 num_shards=opt.rank_size, shard_id=opt.rank)
    else:
        ds = de.GeneratorDataset(dataset, column_names=dataset_column_names,
                                 num_parallel_workers=num_parallel_workers, shuffle=shuffle)
    print(f"[INFO] Batch size: {opt.batch_size}", flush=True)
    ds = ds.batch(opt.batch_size,
                  per_batch_map=LoadImagesAndLabels.collate_fn4 if opt.quad else LoadImagesAndLabels.collate_fn,
                  input_columns=dataset_column_names,
                  drop_remainder=drop_remainder)
    if is_training:
        ds = ds.project(columns=["img", "label_out"])
    else:
        ds = ds.repeat(opt.epochs)

    per_epoch_size = int(len(dataset) / opt.batch_size / opt.rank) if drop_remainder else \
        math.ceil(len(dataset) / opt.batch_size / opt.rank_size)

    return ds, dataset, per_epoch_size
