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
# =======================================================================================

from __future__ import annotations

import codecs
import glob
import json
import os
import time
from collections import namedtuple
from pathlib import Path

import yaml
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, ops
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from pycocotools.coco import COCO

from config.args import get_args_eval
from src.coco_visual import CocoVisualUtil
from src.dataset import create_dataloader
from src.general import LOGGER, AllReduce, empty
from src.general import COCOEval as COCOeval
from src.general import (Synchronize, SynchronizeManager, box_iou, check_file,
                         check_img_size, coco80_to_coco91_class, colorstr,
                         increment_path, xywh2xyxy, xyxy2xywh, WRITE_FLAGS, FILE_MODE)
from src.metrics import (ConfusionMatrix, ap_per_class, non_max_suppression, scale_coords)
from src.network.yolo import Model
from src.plots import output_to_target, plot_images, plot_study_txt
from third_party.yolo2coco.yolo2coco import YOLO2COCO


class Dict(dict):
    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, item):
        return self.__getitem__(item)


class COCOResult:
    def __init__(self, eval_result=None):
        self.stats: np.ndarray | None = None
        self.stats_str: str = ''
        self.category_stats: list[np.ndarray] = []
        self.category_stats_strs: list[str] = []
        if eval_result is not None:
            self.stats = eval_result.stats  # np.ndarray
            self.stats_str = eval_result.stats_str  # str
            self.category_stats = eval_result.category_stats  # List[np.ndarray]
            self.category_stats_strs = eval_result.category_stats_strs  # List[str]

    def get_map(self):
        if self.stats is None:
            return -1
        return self.stats[0]

    def get_map50(self):
        if self.stats is None:
            return -1
        return self.stats[1]


class MetricStatistics:
    def __init__(self):
        self.mp = 0.  # mean precision
        self.mr = 0.  # mean recall
        self.map50 = 0.  # mAP@50
        self.map = 0.  # mAP@50:95
        self.loss_box = 0.
        self.loss_obj = 0.
        self.loss_cls = 0.

        self.pred_json = []
        self.pred_stats = []    # (correct, conf, pred_cls, target_cls)
        self.tp = np.array(0)  # true positive
        self.fp = np.array(0)  # false positive
        self.precision = np.array(0)
        self.recall = np.array(0)
        self.f1 = np.array(0)
        self.ap = np.array(0)  # average precision(AP)
        self.ap50 = np.array(0)  # average precision@50(AP@50)
        self.ap_cls = np.array(0)  # average precision(AP) of each class

        self.seen = 0
        self.confusion_matrix = None
        self.nt = None


    def __iter__(self):
        for _, val in vars(self).items():
            yield val

    def set_loss(self, loss):
        self.loss_box, self.loss_obj, self.loss_cls = loss.tolist()

    def get_loss_tuple(self):
        return self.loss_box, self.loss_obj, self.loss_cls

    def set_mean_stats(self):
        self.mp = np.mean(self.precision)
        self.mr = np.mean(self.recall)
        self.map50 = np.mean(self.ap50)
        self.map = np.mean(self.ap)

    def get_mean_stats(self):
        return self.mp, self.mr, self.map50, self.map

    def get_map(self):
        return self.map

    def compute_ap_per_class(self, plot=False, save_dir='.', names=()):
        tp, conf, pred_class, target_cls = self.pred_stats
        result = ap_per_class(tp, conf, pred_class, target_cls, plot=plot, save_dir=save_dir, names=names)
        # result: tp, fp, p, r, f1, ap, unique_classes.astype(int)
        self.tp, self.fp = result.tp, result.fp
        self.precision, self.recall, self.f1 = result.precision, result.recall, result.f1
        self.ap = result.ap
        self.ap_cls = result.unique_class
        # AP@0.5, AP@0.5:0.95
        self.ap50 = self.ap[:, 0]
        self.ap = np.mean(self.ap, axis=1)
        self.set_mean_stats()

    def get_ap_per_class(self, idx):
        return self.precision[idx], self.recall[idx], self.ap50[idx], self.ap[idx]


class TimeStatistics:
    def __init__(self):
        self.infer_duration = 0.
        self.nms_duration = 0.
        self.metric_duration = 0.

    def total_duration(self):
        return self.infer_duration + self.nms_duration + self.metric_duration

    def get_tuple(self):
        duration_tuple = namedtuple('Duration', ['infer', 'nms', 'metric', 'total'])
        return duration_tuple(self.infer_duration, self.nms_duration, self.metric_duration, self.total_duration())


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(np.bool_)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i, threshold in enumerate(iouv):
        x = np.where((iou >= threshold) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return np.array(correct).astype(np.bool_)


def load_checkpoint_to_yolo(model, ckpt_path):
    param_dict = ms.load_checkpoint(ckpt_path)
    new_params = {}
    for k, v in param_dict.items():
        if k.startswith("model.") or k.startswith("updates"):
            new_params[k] = v
        if k.startswith("ema.ema."):
            k = k[len("ema.ema."):]
            new_params[k] = v
    ms.load_param_into_net(model, new_params)
    LOGGER.info(f"load ckpt from \"{ckpt_path}\" success.")


class EvalManager:
    def __init__(self, opt, half_precision=False, compute_loss=None):
        self.opt = opt
        self.half_precision = half_precision
        self.is_coco = False
        self.dataset_cfg: dict
        self.hyper_params = None
        if isinstance(opt.data, str):
            self.is_coco = opt.data.endswith('coco.yaml')
            with open(opt.data) as f:
                self.dataset_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        elif isinstance(opt.data, dict):
            self.dataset_cfg = opt.data
        else:
            raise TypeError("The type of opt.data must be str or dict.")
        if self.opt.single_cls:
            self.dataset_cfg['nc'] = 1
        self.confusion_matrix = ConfusionMatrix(nc=self.dataset_cfg['nc'])
        self._process_dataset_cfg()
        self.project_dir: str = ''
        self.save_dir: str = './'
        self.model = None
        self.training = False
        self.img_size = self.opt.img_size
        self.batch_size = self.opt.batch_size
        self.grid_size = None
        self.cls_map: list[int] = []
        self.compute_loss = compute_loss
        self.synchronize = Synchronize(self.opt.rank_size)
        self.reduce_sum = AllReduce()

    @staticmethod
    def save_json(pred_json, save_path):
        with os.fdopen(os.open(save_path, WRITE_FLAGS, FILE_MODE), 'w') as file:
            json.dump(pred_json, file)

    def save_map(self, coco_result):
        dataset_cfg = self.dataset_cfg
        s = f"\n{len(glob.glob(os.path.join(self.save_dir, 'labels/*.txt')))} labels saved to " \
            f"{os.path.join(self.save_dir, 'labels')}" if self.opt.save_txt else ''
        LOGGER.info(f"Results saved to {self.save_dir}, {s}")
        with os.fdopen(os.open("class_map.txt", WRITE_FLAGS, FILE_MODE), "w") as file:
            file.write(f"COCO map:\n{coco_result.stats_str}\n")
            if coco_result.category_stats_strs:
                for idx, category_str in enumerate(coco_result.category_stats_strs):
                    file.write(f"\nclass {dataset_cfg['names'][idx]}:\n{category_str}\n")

    def get_val_anno(self):
        dataset_cfg = self.dataset_cfg
        opt = self.opt
        data_dir = Path(dataset_cfg["val"]).parent
        anno_json = os.path.join(data_dir, "annotations/instances_val2017.json")
        if opt.transfer_format and not os.path.exists(anno_json):
            # data format transfer if annotations does not exists
            LOGGER.info("Transfer annotations from yolo to coco format.")
            transformer = YOLO2COCO(data_dir, output_dir=data_dir,
                                    class_names=dataset_cfg["names"], class_map=self.cls_map,
                                    mode='val', annotation_only=True)
            transformer()
        return anno_json

    def eval_coco(self, anno_json, pred_json, dataset=None):
        LOGGER.info("Start evaluating mAP...")
        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval_result = COCOeval(anno, pred, 'bbox')
        if self.is_coco and dataset is not None:
            eval_result.params.imgIds = [int(Path(x).stem) for x in dataset.img_files]  # image IDs to evaluate
        eval_result.evaluate()
        eval_result.accumulate()
        eval_result.summarize(category_ids=-1)
        coco_result = COCOResult(eval_result)
        LOGGER.info("Finish evaluating mAP.")
        return coco_result

    def visualize_coco(self, anno_json, pred_json_path):
        LOGGER.info("Start visualization result.")
        dataset_cfg = self.dataset_cfg
        dataset_coco = COCO(anno_json)
        coco_visual = CocoVisualUtil()
        eval_types = ["bbox"]
        config = {"dataset": "coco"}
        data_dir = Path(dataset_cfg["val"]).parent
        img_path_name = os.path.splitext(os.path.basename(dataset_cfg["val"]))[0]
        im_path_dir = os.path.join(data_dir, "images", img_path_name)
        with open(pred_json_path, 'r') as f:
            result = json.load(f)
        result_files = coco_visual.results2json(dataset_coco, result, "./results.pkl")
        coco_visual.coco_eval(Dict(config), result_files, eval_types, dataset_coco, im_path_dir=im_path_dir,
                              score_threshold=None,
                              recommend_threshold=self.opt.recommend_threshold)

    def print_stats(self, metric_stats, time_stats):
        total_time_fmt_str = 'Total time: {:.1f}/{:.1f}/{:.1f}/{:.1f} s ' \
                             'inference/NMS/Metric/total {:g}x{:g} image at batch-size {:g}'
        speed_fmt_str = 'Speed: {:.1f}/{:.1f}/{:.1f}/{:.1f} ms ' \
                        'inference/NMS/Metric/total per {:g}x{:g} image at batch-size {:g}'
        img_size, batch_size = self.img_size, self.batch_size
        total_time = (*time_stats.get_tuple(), img_size, img_size, batch_size)  # tuple
        speed = tuple(x / metric_stats.seen * 1E3 for x in total_time[:4]) + (img_size, img_size, batch_size)  # tuple
        LOGGER.info(speed_fmt_str.format(*speed))
        LOGGER.info(total_time_fmt_str.format(*total_time))
        return speed

    def eval(self, model=None, dataset=None, dataloader=None, cur_epoch=None):
        opt = self.opt
        self._create_dirs(cur_epoch)
        model = self._config_model(model)
        dataloader, dataset, per_epoch_size = self._config_dataset(dataset, dataloader)
        if opt.v5_metric:
            LOGGER.info("Testing with YOLOv5 AP metric...")

        dataset_cfg = self.dataset_cfg
        dataset_cfg['names'] = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))
        start_idx = 1
        self.cls_map = coco80_to_coco91_class() if self.is_coco else list(range(start_idx, 1000 + start_idx))

        # Test
        metric_stats, time_stats = self._eval(model, dataloader, per_epoch_size)
        self._compute_map_stats(metric_stats)

        # Print speeds
        speed = self.print_stats(metric_stats, time_stats)

        # Plots
        if opt.plots:
            self._plot_confusion_matrix()

        coco_result = COCOResult()
        # Save JSON
        if opt.save_json and not empty(metric_stats.pred_json):
            try:
                coco_result = self._save_eval_result(metric_stats)
            except Exception:
                LOGGER.exception("Error when evaluating COCO mAP")

        # Return results
        if not self.training and opt.rank % 8 == 0:
            self.save_map(coco_result)
        maps = np.zeros(dataset_cfg['nc']) + coco_result.get_map()
        if opt.rank % 8 == 0:
            for i, c in enumerate(metric_stats.ap_cls):
                maps[c] = metric_stats.ap[i]

        model.set_train()
        val_result = namedtuple('ValResult', ['metric_stats', 'maps', 'speed', 'coco_result'])
        return val_result(metric_stats, maps, speed, coco_result)

    def _save_eval_result(self, metric_stats):
        opt = self.opt
        anno_json = self.get_val_anno()
        ckpt_name = Path(opt.weights).stem if opt.weights is not None else ''  # weights
        pred_json_path = os.path.join(self.save_dir, f"{ckpt_name}_predictions_{opt.rank}.json")  # predictions json
        LOGGER.info(f'Evaluating pycocotools mAP... saving {pred_json_path}...')
        self.save_json(metric_stats.pred_json, pred_json_path)
        with SynchronizeManager(opt.rank % 8, min(8, opt.rank_size), opt.distributed_eval, self.project_dir):
            result = COCOResult()
            if opt.rank % 8 == 0:
                pred_json = metric_stats.pred_json
                if opt.distributed_eval:
                    pred_json_path, pred_json = self._merge_pred_json(prefix=ckpt_name)
                if opt.result_view or opt.recommend_threshold:
                    try:
                        self.visualize_coco(anno_json, pred_json_path)
                    except Exception:
                        LOGGER.exception("Failed when visualize evaluation result.")
                try:
                    result = self.eval_coco(anno_json, pred_json)
                    LOGGER.info(f"\nCOCO mAP:\n{result.stats_str}")
                except Exception:
                    LOGGER.exception("Exception when running pycocotools")
            coco_result = result
        return coco_result

    def _plot_confusion_matrix(self):
        dataset_cfg = self.dataset_cfg
        opt = self.opt
        matrix = ms.Tensor(self.confusion_matrix.matrix)
        if opt.distributed_eval:
            matrix = AllReduce()(matrix).asnumpy()
        self.confusion_matrix.matrix = matrix
        if opt.rank % 8 == 0:
            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(dataset_cfg['names'].values()))

    def _merge_pred_json(self, prefix=''):
        LOGGER.info("Merge detection results...")
        merged_json = os.path.join(self.project_dir, f"{prefix}_predictions_merged.json")
        merged_result = []
        # Waiting
        while True:
            json_files = list(Path(self.project_dir).rglob("*.json"))
            if len(json_files) != min(8, self.opt.rank_size):
                time.sleep(1)
                LOGGER.info("Waiting for json file...")
            else:
                break
        for json_file in json_files:
            LOGGER.info(f"Merge {json_file.resolve()}")
            with open(json_file, "r") as file_handler:
                merged_result.extend(json.load(file_handler))
        with os.fdopen(os.open(merged_json, WRITE_FLAGS, FILE_MODE), "w") as file_handler:
            json.dump(merged_result, file_handler)
        LOGGER.info(f"Merged results saved in {merged_json}.")
        return merged_json, merged_result

    def _process_dataset_cfg(self):
        if self.dataset_cfg is None:
            return
        self.dataset_cfg['train'] = os.path.join(self.dataset_cfg['root'], self.dataset_cfg['train'])
        self.dataset_cfg['val'] = os.path.join(self.dataset_cfg['root'], self.dataset_cfg['val'])
        self.dataset_cfg['test'] = os.path.join(self.dataset_cfg['root'], self.dataset_cfg['test'])

    def _create_dirs(self, cur_epoch=None):
        if cur_epoch is not None:
            project_dir = os.path.join(self.opt.project, f"epoch_{cur_epoch}")
        else:
            project_dir = self.opt.project
        save_dir = os.path.join(project_dir, f"save_dir_{self.opt.rank}")
        save_dir = increment_path(save_dir, exist_ok=self.opt.exist_ok)
        os.makedirs(os.path.join(save_dir, f"labels_{self.opt.rank}"), exist_ok=self.opt.exist_ok)
        self.project_dir = project_dir
        self.save_dir = save_dir

    def _config_model(self, model=None):
        self.training = model is not None
        if model is None:  # called by train.py
            # Load model
            # Hyperparameters
            with open(self.opt.hyp) as f:
                self.hyper_params = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
            model = Model(self.opt.cfg, ch=3, nc=self.dataset_cfg['nc'], anchors=self.hyper_params.get('anchors'),
                          sync_bn=False, hyp=self.hyper_params)  # create
            ckpt_path = self.opt.weights
            load_checkpoint_to_yolo(model, ckpt_path)
            self.grid_size = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
            self.img_size = check_img_size(self.opt.img_size, s=self.grid_size)  # check img_size
        self.grid_size = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
        self.img_size = self.img_size[0] if isinstance(self.img_size, list) else self.img_size
        self.img_size = check_img_size(self.img_size, s=self.grid_size)  # check img_size

        # Half
        if self.half_precision:
            model.to_float(ms.float16)

        model.set_train(False)
        return model

    def _config_dataset(self, dataset=None, dataloader=None):
        data_cfg = self.dataset_cfg
        rank = self.opt.rank
        rank_size = self.opt.rank_size
        if dataloader is None or dataset is None:
            LOGGER.info(f"Enable rect: {self.opt.rect}")
            task = self.opt.task if self.opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test img
            dataloader, dataset, per_epoch_size = create_dataloader(data_cfg[task], self.img_size, self.opt.batch_size,
                                                                    self.grid_size, self.opt,
                                                                    epoch_size=1, pad=0.5, rect=self.opt.rect,
                                                                    rank=rank % 8,
                                                                    rank_size=min(8, rank_size),
                                                                    num_parallel_workers=4 if rank_size > 1 else 8,
                                                                    shuffle=False,
                                                                    drop_remainder=False,
                                                                    prefix=colorstr(f'{task}: '))
            assert per_epoch_size == dataloader.get_dataset_size()
            dataloader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
            LOGGER.info(f"Test create dataset success, epoch size {per_epoch_size}.")
        else:
            assert dataset is not None
            assert dataloader is not None
            per_epoch_size = dataloader.get_dataset_size()
            dataloader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
        return dataloader, dataset, per_epoch_size

    def _nms(self, pred, labels):
        nms_start_time = time.time()
        out = non_max_suppression(pred.asnumpy(),
                                  self.opt.conf_thres,
                                  self.opt.iou_thres,
                                  labels=labels,
                                  multi_label=True,
                                  agnostic=self.opt.single_cls)
        nms_duration = time.time() - nms_start_time
        return out, nms_duration

    def _write_txt(self, pred, shape, path):
        # Save result to txt
        path = Path(path)
        file_path = os.path.join(self.save_dir, 'labels', f'{path.stem}.txt')
        gn = np.array(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in pred.tolist():
            xywh = (xyxy2xywh(np.array(xyxy).reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
            with os.fdopen(os.open(file_path, WRITE_FLAGS, FILE_MODE), 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def _write_json_list(self, pred, pred_json, path):
        # Save one JSON result
        # >> example:
        # >> {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        path = Path(path)
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = xyxy2xywh(pred[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(pred.tolist(), box.tolist()):
            pred_json.append({
                'image_id': image_id,
                'category_id': self.cls_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})

    def _compute_metrics(self, img, targets, out: list[np.ndarray], paths, shapes, metric_stats: MetricStatistics):
        iou_vec = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        num_iou = np.prod(iou_vec.shape)
        metric_start_time = time.time()
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr, shape = labels.shape[0], pred.shape[0], shapes[si][0]  # number of labels, predictions
            if isinstance(paths[si], (np.bytes_, np.ndarray)):
                path = Path(str(codecs.decode(paths[si].tostring()).strip(b'\x00'.decode())))
            else:
                path = Path(paths[si])
            # array[N, 10], bool, correct under 10 different iou threshold
            correct = np.zeros((npr, num_iou)).astype(np.bool_)  # init
            metric_stats.seen += 1

            if npr == 0:
                if nl:
                    metric_stats.pred_stats.append((correct, *np.zeros((2, 0)).astype(np.bool_), labels[:, 0]))
                    if self.opt.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if self.opt.single_cls:
                pred[:, 5] = 0
            pred_copy = np.copy(pred)
            # native-space pred
            pred_copy[:, :4] = scale_coords(img[si].shape[1:], pred_copy[:, :4], shape, shapes[si][1:])

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tbox = scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1:])  # native-space labels
                labelsn = np.concatenate((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(pred_copy, labelsn, iou_vec)
                if self.opt.plots:
                    self.confusion_matrix.process_batch(pred_copy, labelsn)
            # correct, conf, pred_cls, target_cls
            metric_stats.pred_stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            # Save/log
            if self.opt.save_txt:
                self._write_txt(pred_copy, shape, path)
            if self.opt.save_json:
                self._write_json_list(pred_copy, metric_stats.pred_json, path)
        metric_duration = time.time() - metric_start_time
        return metric_duration

    def _eval(self, model, dataloader, per_epoch_size):
        opt = self.opt
        dataset_cfg = self.dataset_cfg
        loss = np.zeros(3)
        time_stats = TimeStatistics()
        metric_stats = MetricStatistics()
        step_start_time = time.time()
        for batch_idx, meta in enumerate(dataloader):
            # targets: Nx6 ndarray, img_id, label, x, y, w, h
            img, targets, paths, shapes = meta["img"], meta["label_out"], meta["img_files"], meta["shapes"]
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            img_tensor = Tensor.from_numpy(img)
            targets_tensor = Tensor.from_numpy(targets)
            if self.half_precision:
                img_tensor = ops.cast(img_tensor, ms.float16)
                targets_tensor = ops.cast(targets_tensor, ms.float16)

            targets = targets.reshape((-1, 6))
            targets = targets[targets[:, 1] >= 0]
            nb, _, height, width = img.shape  # batch size, channels, height, width
            data_duration = time.time() - step_start_time

            # Run model
            infer_start_time = time.time()
            # inference and training outputs
            if self.compute_loss or not opt.augment:
                pred_out, train_out = model(img_tensor)
            else:
                pred_out, train_out = (model(img_tensor, augment=opt.augment), None)
            infer_duration = time.time() - infer_start_time
            time_stats.infer_duration += infer_duration

            # Compute loss
            if self.compute_loss:
                loss += self.compute_loss(train_out, targets_tensor)[1][:3].asnumpy()  # box, obj, cls

            # NMS
            targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
            label = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
            out, nms_duration = self._nms(pred_out, label)
            time_stats.nms_duration += nms_duration

            # Metrics
            metric_duration = self._compute_metrics(img, targets, out, paths, shapes, metric_stats)
            time_stats.metric_duration += metric_duration
            # Plot images
            if opt.plots and batch_idx < 3:
                labels_path = os.path.join(self.save_dir, f'test_batch{batch_idx}_labels.jpg')  # labels
                plot_images(img, targets, paths, labels_path, dataset_cfg['names'])
                pred_path = os.path.join(self.save_dir, f'test_batch{batch_idx}_pred.jpg')  # predictions
                plot_images(img, output_to_target(out), paths, pred_path, dataset_cfg['names'])

            LOGGER.info(f"Step {batch_idx + 1}/{per_epoch_size} "
                        f"Time total {(time.time() - step_start_time):.2f}s  "
                        f"Data {data_duration * 1e3:.2f}ms  "
                        f"Infer {infer_duration * 1e3:.2f}ms  "
                        f"NMS {nms_duration * 1e3:.2f}ms  "
                        f"Metric {metric_duration * 1e3:.2f}ms")
            step_start_time = time.time()
        metric_stats.set_loss(loss / per_epoch_size)
        return metric_stats, time_stats

    def _compute_map_stats(self, metric_stats: MetricStatistics):
        opt = self.opt
        dataset_cfg = self.dataset_cfg
        # Compute metrics
        # pred_stats: list[np.ndarray], np.concatenate((correct, conf, pred_cls, target_cls), 0)
        metric_stats.pred_stats = [np.concatenate(x, 0) for x in zip(*metric_stats.pred_stats)]  # to numpy
        pred_stats_file = os.path.join(self.save_dir, f"pred_stats_{self.opt.rank}.npy")
        np.save(pred_stats_file, np.array(metric_stats.pred_stats, dtype=object), allow_pickle=True)
        if opt.distributed_eval:
            metric_stats.seen = self.reduce_sum(ms.Tensor(np.array(metric_stats.seen, dtype=np.int32))).asnumpy()
            self.synchronize()
        if self.opt.rank % 8 != 0:
            return

        pred_stats: list[list] = self._merge_pred_stats(metric_stats)
        pred_stats: list[np.ndarray] = [np.concatenate(item, axis=0) for item in pred_stats]
        metric_stats.pred_stats = pred_stats

        seen = metric_stats.seen
        names = dataset_cfg['names']
        nc = dataset_cfg['nc']
        if not empty(pred_stats) and pred_stats[0].any():
            metric_stats.compute_ap_per_class(plot=opt.plots, save_dir=self.save_dir, names=self.dataset_cfg['names'])
        nt = np.bincount(pred_stats[3].astype(int), minlength=nc)  # number of targets per class
        metric_stats.nt = nt

        # Print results
        title = ('{:22s}' + '{:11s}' * 6).format('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        pf = '{:<20s}' + '{:<12d}' * 2 + '{:<12.3g}' * 4  # print format
        LOGGER.info(title)
        LOGGER.info(pf.format('all', seen, nt.sum(), *metric_stats.get_mean_stats()))

        # Print results per class
        if (opt.verbose or (nc < 50 and not self.training)) and nc > 1 and not empty(pred_stats):
            for i, c in enumerate(metric_stats.ap_cls):
                # Class     Images  Instances          P          R      mAP50   mAP50-95:
                LOGGER.info(pf.format(names[c], seen, nt[c], *metric_stats.get_ap_per_class(i)))

    def _merge_pred_stats(self, metric_stats):
        # Merge prediction stats
        project_dir = Path(self.save_dir).parent
        pred_stats: list[list] = [[] for _ in range(len(metric_stats.pred_stats))]
        for file_path in project_dir.rglob("pred_stats*.npy"):
            stats = np.load(str(file_path.resolve()), allow_pickle=True)
            for i, item in enumerate(stats):
                pred_stats[i].append(item)
        return pred_stats


def main():
    parser = get_args_eval()
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    print(opt)

    ms_mode = ms.GRAPH_MODE if opt.ms_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=opt.device_target)
    context.set_context(mode=ms_mode, device_target=opt.device_target)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    # Distribute Test
    if opt.distributed_eval:
        init()
        rank, rank_size, parallel_mode = get_rank() % 8, min(8, get_group_size()), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)
    opt.rank_size = rank_size
    opt.rank = rank

    if opt.task in ('train', 'val', 'test'):  # run normally
        print("opt:", opt)
        opt.save_txt = opt.save_txt | opt.save_hybrid
        eval_manager = EvalManager(opt)
        eval_manager.eval()

    elif opt.task == 'speed':  # speed benchmarks
        opt.conf_thres = 0.25
        opt.iou_thres = 0.45
        opt.save_json = False
        opt.plots = False
        eval_manager = EvalManager(opt)
        eval_manager.eval()

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.65 --weights yolov5.ckpt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
        y = []  # y axis
        opt.save_json = False

        for i in x:  # img-size
            print(f'\nRunning {f} point {i}...')
            eval_manager = EvalManager(opt)
            metric_stats, _, speed, _ = eval_manager.eval()
            y.append(tuple(metric_stats)[:7] + speed)  # results and times
        np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == '__main__':
    main()
