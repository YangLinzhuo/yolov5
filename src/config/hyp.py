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
from dataclasses import dataclass


@dataclass
class Hyp:
    # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
    # Hyperparameters for low-augmentation COCO training from scratch
    # python train.py --batch 64 --cfg yolov5n6.yaml --weights '' --data coco.yaml --img 640 --epochs 300 --linear
    # See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials
    lr0: float = 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
    lrf: float = 0.01  # final OneCycleLR learning rate (lr0 * lrf)
    momentum: float = 0.937  # SGD momentum/Adam beta1
    weight_decay: float = 0.0005  # optimizer weight decay 5e-4
    warmup_epochs: float = 3.0  # warmup epochs (fractions ok)
    warmup_momentum: float = 0.8  # warmup initial momentum
    warmup_bias_lr: float = 0.1  # warmup initial bias lr
    box: float = 0.05  # box loss gain
    cls: float = 0.5  # cls loss gain
    cls_pw: float = 1.0  # cls BCELoss positive_weight
    obj: float = 1.0  # obj loss gain (scale with pixels)
    obj_pw: float = 1.0  # obj BCELoss positive_weight
    iou_t: float = 0.20  # IoU training threshold
    anchor_t: float = 4.0  # anchor-multiple threshold
    # anchors: 3  # anchors per output layer (0 to ignore)
    fl_gamma: float = 0.0  # focal loss gamma (efficientDet default gamma=1.5)
    hsv_h: float = 0.015  # image HSV-Hue augmentation (fraction)
    hsv_s: float = 0.7  # image HSV-Saturation augmentation (fraction)
    hsv_v: float = 0.4  # image HSV-Value augmentation (fraction)
    degrees: float = 0.0  # image rotation (+/- deg)
    translate: float = 0.1  # image translation (+/- fraction)
    scale: float = 0.5  # image scale (+/- gain)
    shear: float = 0.0  # image shear (+/- deg)
    perspective: float = 0.0  # image perspective (+/- fraction), range 0-0.001
    flipud: float = 0.0  # image flip up-down (probability)
    fliplr: float = 0.5  # image flip left-right (probability)
    mosaic: float = 1.0  # image mosaic (probability)
    mixup: float = 0.0  # image mixup (probability)
    copy_paste: float = 0.0  # segment copy-paste (probability)
    paste_in: float = 0.0  # image copy paste (probability), use 0 for faster training
    max_box_per_img: int = 160  # label mask
    enable_clip_grad: bool = False
    bn_eps: float = 0.001
    bn_momentum: float = 0.03
    label_smoothing: float = 0.0


def get_hyp(hyp_path: str) -> Hyp:
    with open(hyp_path, 'r') as file:
        hyp = yaml.load(file, Loader=yaml.SafeLoader)  # load hyps
    return Hyp(**hyp)
