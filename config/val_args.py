from __future__ import annotations
import ast
from argparse import ArgumentParser
from typing import Union
from typing_extensions import Literal

from pydantic import BaseModel


MS_MODE_NAME = Literal["graph", "pynative"]
DEVICE_NAME = Literal["Ascend", "GPU", "CPU"]
TASK_NAME = Literal["train", "val", "test", "speed", "study"]


class EnvConfig(BaseModel):
    rank: int = 0
    rank_size: int = 1
    project_dir: str = ''
    save_dir: str = './'


class DatasetConfig(BaseModel):
    dataset_name: str = 'coco'
    root: str
    train: str  # 118287 images
    val: str  # 5000 images
    test: str  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794
    names: Union[list, dict]
    nc: int
    is_coco: bool = True
    cls_start_idx: int = 1
    cls_map: list = []


class DataConfig(BaseModel):
    data: str = 'config/data/coco.yaml'
    batch_size: int = 32
    img_size: int = 640
    grid_size: int = 32
    per_epoch_size: int = -1


class BasicConfig(BaseModel):
    ms_mode: MS_MODE_NAME = 'graph'
    device_target: DEVICE_NAME = 'Ascend'
    cfg: str = 'config/network/yolov5s.yaml'
    hyp: str = 'config/data/hyp.scratch-low.yaml'
    half_precision: bool = False


class InferBasicConfig(BaseModel):
    conf_thres: float = 0.001
    iou_thres: float = 0.65
    task: TASK_NAME = 'val'
    single_cls: bool = False
    augment: bool = False
    verbose: bool = False


class ValConfig(EnvConfig, DataConfig, BasicConfig, InferBasicConfig):
    distributed_eval: bool = False
    weights: str = './EMA_yolov5s_300.ckpt'
    rect: bool = False
    save_txt: bool = False
    save_hybrid: bool = False
    save_conf: bool = False
    save_json: bool = True
    project: str = './run_test'
    exist_ok: bool = False
    trace: bool = False
    plots: bool = True
    transfer_format: bool = True
    result_view: bool = False
    recommend_threshold: bool = False


def get_val_args():
    # Basic
    parser = ArgumentParser(description="val.py")
    parser.add_argument('--ms_mode', type=str, default='graph', help='train mode, graph/pynative')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--cfg', type=str, default='config/network/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='config/data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='config/data/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all device')
    parser.add_argument('--half_precision', type=ast.literal_eval, default=False, help='whether use fp16')

    # Infer basic
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')

    # Val
    parser.add_argument('--distributed_eval', type=ast.literal_eval, default=False, help='Distribute test or not')
    parser.add_argument('--weights', type=str, default='./EMA_yolov5s_300.ckpt', help='model.ckpt path(s)')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--save_txt', type=ast.literal_eval, default=False, help='save results to *.txt')
    parser.add_argument('--save_hybrid', type=ast.literal_eval, default=False,
                        help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save_conf', type=ast.literal_eval, default=False,
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save_json', type=ast.literal_eval, default=True,
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', type=str, default='./run_test', help='save to project/name')
    parser.add_argument('--exist_ok', type=ast.literal_eval, default=False,
                        help='existing project/name ok, do not increment')
    parser.add_argument('--trace', type=ast.literal_eval, default=False, help='trace model')
    parser.add_argument('--plots', type=ast.literal_eval, default=True, help='enable plot')
    parser.add_argument('--transfer_format', type=ast.literal_eval, default=True,
                        help='whether transform data format to coco')
    parser.add_argument('--result_view', type=ast.literal_eval, default=False, help='view the eval result')
    parser.add_argument('--recommend_threshold', type=ast.literal_eval, default=False,
                        help='recommend threshold in eval')
    val_args = parser.parse_args()
    return ValConfig(**vars(val_args))
