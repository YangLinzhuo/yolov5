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
import hashlib
import os
from collections import namedtuple
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import ExifTags, Image, ImageOps

from src.general import segments2boxes


def get_os_cpu_count() -> int:
    os_cpu_count = os.cpu_count()
    if os_cpu_count is None:
        return 1
    return os_cpu_count


# Parameters
HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
PIN_MEMORY = str(os.getenv('PIN_MEMORY', "True")).lower() == 'true'  # global pin_memory for dataloaders
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
NUM_THREADS = min(8, max(1, get_os_cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
IMG_TUPLE = namedtuple('IMG_TUPLE', ['img', 'labels', 'img_path', 'shapes'])


def get_hash(paths: List[str]) -> str:
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def img2label_paths(img_paths: List[str]) -> List[str]:
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


def load_images_and_labels(path, prefix: str = '') -> Tuple[List[str], List[str]]:
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # file
                with open(p) as t:
                    lines = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in lines]  # to global path
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')
        img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        assert img_files, f'{prefix}No images found'
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {path}: {e}\n{HELP_URL}') from e
    # Check cache
    label_files = img2label_paths(img_files)  # labels
    return img_files, label_files


def get_orientation_key():
    # Get orientation exif tag
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            return orientation
    return None


ImgShape = Tuple[int, int]


def exif_size(img: Image.Image) -> ImgShape:
    # Returns exif-corrected PIL size
    s: ImgShape = img.size  # (width, height)
    try:
        rotation = img.getexif()[get_orientation_key()]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


# img_file, label, shape, segments, n_missing, n_found, n_empty, n_corrupt, msg
VerifiedImageInfo = Tuple[Optional[str],
                          Optional[np.ndarray],
                          Optional[ImgShape],
                          Optional[List[np.ndarray]],
                          int, int, int, int, str]


def verify_image_label(args: Tuple[str, str, str]) -> VerifiedImageInfo:
    # Verify one image-label pair
    img_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(img_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        img_format = im.format
        if img_format is None:
            raise RuntimeError("The format of image file is None")
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert img_format.lower() in IMG_FORMATS, f'invalid image format {img_format}'
        if img_format.lower() in ('jpg', 'jpeg'):
            with open(img_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(img_file)).save(img_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {img_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb_lst = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb_lst):  # is segment
                    classes = np.array([x[0] for x in lb_lst], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb_lst]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {img_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return img_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {img_file}: ignoring corrupt image/label: {e}'
        return None, None, None, None, nm, nf, ne, nc, msg
