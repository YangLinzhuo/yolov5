from __future__ import annotations

import codecs
import os
import functools
import glob
import json
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

import mindspore as ms
from mindspore import Tensor, ops
from pycocotools.coco import COCO

from config.val_args import ValConfig, DatasetConfig
from src.coco_visual import CocoVisualUtil
from src.metrics import non_max_suppression, ConfusionMatrix, ap_per_class, scale_coords, box_iou
from src.general import COCOEval as COCOeval
from src.general import (increment_path, check_img_size, colorstr, coco80_to_coco91_class, xyxy2xywh, xywh2xyxy,
                         AllReduce, Synchronize, SynchronizeManager)
from src.network.yolo import Model
from src.dataset import create_dataloader
from src.plots import output_to_target, plot_images, plot_study_txt
from third_party.yolo2coco.yolo2coco import YOLO2COCO


class TimeStatistics:
    def __init__(self):
        self.infer = 0.
        self.nms = 0.
        self.metric = 0.

    def total_time(self):
        return self.infer + self.nms + self.metric

    def get_tuple(self):
        duration_tuple = namedtuple('Duration', ['infer', 'nms', 'metric', 'total'])
        return duration_tuple(self.infer, self.nms, self.metric, self.total_time())


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
        self.nt = None

    def __iter__(self):
        for _, value in vars(self).items():
            yield value

    @property
    def loss(self):
        return self.loss_box, self.loss_obj, self.loss_cls

    @loss.setter
    def loss(self, loss_val):
        self.loss_box, self.loss_obj, self.loss_cls = loss_val.tolist()

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


class COCOResult:
    # TODO: Write in more cleaner way
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


def timer(time_stats: TimeStatistics, tag: str):
    def wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            begin_time = time.time()
            output = func(*args, **kwargs)
            end_time = time.time()
            setattr(time_stats, tag, end_time - begin_time)
            return output
        return inner_wrapper
    return wrapper


def load_checkpoint_to_yolo(model: Model, ckpt_path):
    param_dict = ms.load_checkpoint(ckpt_path)
    new_params = {}
    for k, v in param_dict.items():
        if k.startswith("model.") or k.startswith("updates"):
            new_params[k] = v
        if k.startswith("ema.ema."):
            k = k.lstrip("ema.ema.")
            new_params[k] = v
    ms.load_param_into_net(model, new_params)


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


def write_json_list(cls_map, pred, pred_json, path):
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
            'category_id': cls_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def catch_exception(msg=None):
    def decorator(func):
        @functools.wraps(func)
        def _wrapped_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                if msg is not None:
                    print(msg)
                import traceback
                traceback.print_exc()
        return _wrapped_func
    return decorator


def save_json(pred_json, save_path):
    with open(save_path, 'w') as file:
        json.dump(pred_json, file)


class EvalManager:
    def __init__(self, opt: ValConfig, model=None, dataset=None, dataloader=None):
        self.opt = opt
        self.dataset_cfg = self.__get_dataset_cfg()
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.__init()
        self.time_stats = TimeStatistics()
        self.metric_stats = MetricStatistics()
        self.confusion_matrix = ConfusionMatrix(nc=self.dataset_cfg.nc)
        self.all_reduce = AllReduce()

    def __init(self):
        self.configure_model()
        self.configure_dataloader()
        self.configure_dataset_cfg()

    def configure_model(self):
        opt, dataset_cfg, model = self.opt, self.dataset_cfg, self.model
        if model is None:  # called in training process
            # Load model and hyperparameters
            with open(opt.hyp) as f:
                hyper_params = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
            model = Model(opt.cfg, ch=3, nc=dataset_cfg.nc, anchors=hyper_params.get('anchors'),
                          sync_bn=False, hyp=hyper_params)  # create
            load_checkpoint_to_yolo(model, opt.weights)
            opt.grid_size = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
            opt.img_size = check_img_size(opt.img_size, s=opt.grid_size)  # check img_size
        opt.grid_size = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
        opt.img_size = opt.img_size[0] if isinstance(opt.img_size, list) else opt.img_size
        opt.img_size = check_img_size(opt.img_size, s=opt.grid_size)  # check img_size

        # Half
        if opt.half_precision:
            model.to_float(ms.float16)

        model.set_train(False)

    def configure_dataloader(self):
        opt, dataset_cfg = self.opt, self.dataset_cfg
        dataset, dataloader = self.dataset, self.dataloader
        if dataloader is None or dataset is None:
            task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test img
            dataloader, dataset, per_epoch_size = create_dataloader(getattr(dataset_cfg, task), opt.img_size,
                                                                    opt.batch_size,
                                                                    opt.grid_size, opt,
                                                                    epoch_size=1, pad=0.5, rect=opt.rect,
                                                                    rank=opt.rank % 8,
                                                                    rank_size=min(8, opt.rank_size),
                                                                    num_parallel_workers=4 if opt.rank_size > 1 else 8,
                                                                    shuffle=False,
                                                                    drop_remainder=False,
                                                                    prefix=colorstr(f'{task}: '))
            assert per_epoch_size == dataloader.get_dataset_size()
            dataloader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
        else:
            assert dataset is not None
            assert dataloader is not None
            per_epoch_size = dataloader.get_dataset_size()
            dataloader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
        opt.per_epoch_size = per_epoch_size
        self.dataset = dataset
        self.dataloader = dataloader

    def configure_dataset_cfg(self):
        dataset_cfg, model = self.dataset_cfg, self.model
        dataset_cfg.names = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))
        dataset_cfg.cls_start_idx = 1
        dataset_cfg.cls_map = coco80_to_coco91_class() if dataset_cfg.is_coco \
            else list(range(dataset_cfg.cls_start_idx, 1000 + dataset_cfg.cls_start_idx))

    def __get_dataset_cfg(self):
        opt = self.opt
        is_coco = opt.data.endswith('coco.yaml')
        with open(opt.data) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        if opt.single_cls:
            cfg['nc'] = 1
        for name in ('train', 'val', 'test'):
            cfg[name] = os.path.join(cfg['root'], cfg[name])
        dataset_cfg = DatasetConfig(**cfg, is_coco=is_coco)
        return dataset_cfg

    def create_folders(self, cur_epoch):
        opt = self.opt
        if cur_epoch is not None:
            opt.project_dir = os.path.join(opt.project, f"epoch_{cur_epoch}")
        else:
            opt.project_dir = opt.project
        opt.save_dir = os.path.join(opt.project_dir, f"save_dir_{opt.rank}")
        opt.save_dir = increment_path(opt.save_dir, exist_ok=opt.exist_ok)
        os.makedirs(os.path.join(opt.save_dir, f"labels_{opt.rank}"), exist_ok=opt.exist_ok)

    def val_step(self, img: np.ndarray, compute_loss):
        opt, model = self.opt, self.model
        img = Tensor.from_numpy(img)
        if opt.half_precision:
            img = ops.cast(img, ms.float16)
        # inference and training outputs
        if compute_loss or not opt.augment:
            pred_out, train_out = model(img)
        else:
            pred_out, train_out = (model(img, augment=opt.augment), None)
        return pred_out, train_out

    def loss_step(self, train_out, targets, compute_loss):
        opt = self.opt
        if not compute_loss:
            return 0.
        targets = Tensor.from_numpy(targets)
        if opt.half_precision:
            targets = ops.cast(targets, ms.float16)
        loss = compute_loss(train_out, targets)[1][:3].asnumpy()  # box, obj, cls
        return loss

    def eval_metric(self, out: list[np.ndarray], data=None):
        opt, confusion_matrix = self.opt, self.confusion_matrix
        metric_stats = self.metric_stats
        cls_map = self.dataset_cfg.cls_map
        img, targets, paths, shapes = data
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
                    if opt.plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if opt.single_cls:
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
                if opt.plots:
                    confusion_matrix.process_batch(pred_copy, labelsn)
            # correct, conf, pred_cls, target_cls
            metric_stats.pred_stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            # Save/log
            if opt.save_txt:
                self.write_txt(pred_copy, shape, path)
            if opt.save_json:
                write_json_list(cls_map, pred_copy, metric_stats.pred_json, path)
        metric_duration = time.time() - metric_start_time
        return metric_duration

    def run_val(self, compute_loss):
        total_time_stats = TimeStatistics()
        opt, dataset_cfg, model = self.opt, self.dataset_cfg, self.model
        loss = np.zeros(3)
        step_start_time = time.time()
        for idx, data in enumerate(self.dataloader):
            # before step
            # targets: Nx6 ndarray, img_id, label, x, y, w, h
            img, targets, paths, shapes = data["img"], data["label_out"], data["img_files"], data["shapes"]
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            data_step_time = time.time() - step_start_time
            # step
            infer_start_time = time.time()
            pred_out, train_out = self.val_step(img, compute_loss)
            infer_step_time = time.time() - infer_start_time
            total_time_stats.infer += infer_step_time

            loss += self.loss_step(train_out, targets, compute_loss)
            # after step
            targets = targets.reshape((-1, 6))
            targets = targets[targets[:, 1] >= 0]
            nb, _, height, width = img.shape  # batch size, channels, height, width
            targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
            label = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
            nms_start_time = time.time()
            out = non_max_suppression(pred_out.asnumpy(),
                                      opt.conf_thres, opt.iou_thres,
                                      labels=label, multi_label=True,
                                      agnostic=opt.single_cls)
            nms_step_time = time.time() - nms_start_time
            total_time_stats.nms += nms_step_time

            # Metrics
            metric_step_time = self.eval_metric(out, data=(img, targets, paths, shapes))
            total_time_stats.metric += metric_step_time
            # Plot images
            if opt.plots and idx < 3:
                labels_path = os.path.join(opt.save_dir, f'test_batch{idx}_labels.jpg')  # labels
                plot_images(img, targets, paths, labels_path, dataset_cfg.names)
                pred_path = os.path.join(opt.save_dir, f'test_batch{idx}_pred.jpg')  # predictions
                plot_images(img, output_to_target(out), paths, pred_path, dataset_cfg.names)

            print(f"Step {idx + 1}/{opt.per_epoch_size} "
                  f"Time total {(time.time() - step_start_time):.2f}s  "
                  f"Data {data_step_time * 1e3:.2f}ms  "
                  f"Infer {infer_step_time * 1e3:.2f}ms  "
                  f"NMS {nms_step_time * 1e3:.2f}ms  "
                  f"Metric {metric_step_time * 1e3:.2f}ms")
            step_start_time = time.time()

        self.metric_stats.loss = loss / opt.per_epoch_size

        # Plots
        self.plot_confusion_matrix()

    def val(self, cur_epoch=None, compute_loss=None):
        opt = self.opt
        metric_stats = self.metric_stats
        dataset_cfg = self.dataset_cfg
        self.create_folders(cur_epoch)
        self.run_val(compute_loss)
        self.compute_map_stats()
        # Print speeds
        speed = self.print_stats()
        # Save JSON
        coco_result = self.save_eval_result()

        # Return results
        self.save_map(coco_result)
        maps = np.zeros(dataset_cfg.nc) + coco_result.get_map()
        if opt.rank % 8 == 0:
            for i, c in enumerate(metric_stats.ap_cls):
                maps[c] = metric_stats.ap[i]

        self.model.set_train()
        val_result = namedtuple('ValResult', ['metric_stats', 'maps', 'speed', 'coco_result'])
        return val_result(metric_stats, maps, speed, coco_result)

    def print_stats(self):
        opt = self.opt
        time_stats = self.time_stats
        metric_stats = self.metric_stats
        total_time_fmt_str = 'Total time: {:.1f}/{:.1f}/{:.1f}/{:.1f} s ' \
                             'inference/NMS/Metric/total {:g}x{:g} image at batch-size {:g}'
        speed_fmt_str = 'Speed: {:.1f}/{:.1f}/{:.1f}/{:.1f} ms ' \
                        'inference/NMS/Metric/total per {:g}x{:g} image at batch-size {:g}'
        img_size, batch_size = opt.img_size, opt.batch_size
        total_time = (*time_stats.get_tuple(), img_size, img_size, batch_size)  # tuple
        speed = tuple(x / metric_stats.seen * 1E3 for x in total_time[:4]) + (img_size, img_size, batch_size)  # tuple
        print(speed_fmt_str.format(*speed))
        print(total_time_fmt_str.format(*total_time))
        return speed

    def merge_pred_stats(self):
        opt = self.opt
        metric_stats = self.metric_stats
        # Merge prediction stats
        project_dir = Path(opt.save_dir).parent
        pred_stats: list[list] = [[] for _ in range(len(metric_stats.pred_stats))]
        for file_path in project_dir.rglob("pred_stats*.npy"):
            stats = np.load(str(file_path.resolve()), allow_pickle=True)
            for i, item in enumerate(stats):
                pred_stats[i].append(item)
        return pred_stats

    def compute_map_stats(self):
        opt = self.opt
        metric_stats = self.metric_stats
        dataset_cfg = self.dataset_cfg
        # Compute metrics
        # pred_stats: list[np.ndarray], np.concatenate((correct, conf, pred_cls, target_cls), 0)
        metric_stats.pred_stats = [np.concatenate(x, 0) for x in zip(*metric_stats.pred_stats)]  # to numpy
        pred_stats_file = os.path.join(opt.save_dir, f"pred_stats_{opt.rank}.npy")
        np.save(pred_stats_file, np.array(metric_stats.pred_stats, dtype=object), allow_pickle=True)
        synchronize = Synchronize(opt.rank_size)
        if opt.distributed_eval:
            metric_stats.seen = self.all_reduce(ms.Tensor(np.array(metric_stats.seen, dtype=np.int32))).asnumpy()
            synchronize()
        if opt.rank % 8 != 0:
            return

        pred_stats: list[list] = self.merge_pred_stats()
        pred_stats: list[np.ndarray] = [np.concatenate(item, axis=0) for item in pred_stats]
        metric_stats.pred_stats = pred_stats

        seen = metric_stats.seen
        names = dataset_cfg.names
        nc = dataset_cfg.nc
        if pred_stats and pred_stats[0].any():
            metric_stats.compute_ap_per_class(plot=opt.plots, save_dir=opt.save_dir, names=dataset_cfg.names)
        nt = np.bincount(pred_stats[3].astype(int), minlength=nc)  # number of targets per class
        metric_stats.nt = nt

        # Print results
        title = ('{:22s}' + '{:11s}' * 6).format('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        pf = '{:<20s}' + '{:<12d}' * 2 + '{:<12.3g}' * 4  # print format
        print(title)
        print(pf.format('all', seen, nt.sum(), *metric_stats.get_mean_stats()))

        # Print results per class
        if (opt.verbose or (nc < 50 and not self.model.training)) and nc > 1 and pred_stats:
            for i, c in enumerate(metric_stats.ap_cls):
                # Class     Images  Instances          P          R      mAP50   mAP50-95:
                print(pf.format(names[c], seen, nt[c], *metric_stats.get_ap_per_class(i)))

    def plot_confusion_matrix(self):
        opt, dataset_cfg, confusion_matrix = self.opt, self.dataset_cfg, self.confusion_matrix
        if not opt.plots:
            return
        matrix = ms.Tensor(confusion_matrix.matrix)
        names = dataset_cfg.names if isinstance(dataset_cfg.names, list) else list(dataset_cfg.names.values())
        if opt.distributed_eval:
            matrix = self.all_reduce(matrix).asnumpy()
        confusion_matrix.matrix = matrix
        if opt.rank % 8 == 0:
            confusion_matrix.plot(save_dir=opt.save_dir, names=names)

    def write_txt(self, pred, shape, path):
        opt = self.opt
        # Save result to txt
        path = Path(path)
        file_path = os.path.join(opt.save_dir, 'labels', f'{path.stem}.txt')
        gn = np.array(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in pred.tolist():
            xywh = (xyxy2xywh(np.array(xyxy).reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
            with open(file_path, 'w') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    @catch_exception("Exception when running pycocotools")
    def eval_coco(self, anno_json, pred_json, is_coco):
        dataset = self.dataset
        print("Start evaluating mAP...")
        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval_result = COCOeval(anno, pred, 'bbox')
        if is_coco and dataset is not None:
            eval_result.params.imgIds = [int(Path(x).stem) for x in dataset.img_files]  # image IDs to evaluate
        eval_result.evaluate()
        eval_result.accumulate()
        eval_result.summarize(category_ids=-1)
        coco_result = COCOResult(eval_result)
        print("Finish evaluating mAP.")
        return coco_result

    @catch_exception("Failed when visualize evaluation result.")
    def visualize_coco(self, anno_json, pred_json_path):
        opt = self.opt
        dataset_cfg = self.dataset_cfg
        print("Start visualization result.")
        dataset_coco = COCO(anno_json)
        coco_visual = CocoVisualUtil()
        eval_types = ["bbox"]

        @dataclass
        class Config:
            dataset: str = "coco"

        config = Config(dataset="coco")
        data_dir = Path(dataset_cfg.val).parent
        img_path_name = os.path.splitext(os.path.basename(dataset_cfg.val))[0]
        im_path_dir = os.path.join(data_dir, "images", img_path_name)
        with open(pred_json_path, 'r') as f:
            result = json.load(f)
        result_files = coco_visual.results2json(dataset_coco, result, "./results.pkl")
        coco_visual.coco_eval(config, result_files, eval_types, dataset_coco, im_path_dir=im_path_dir,
                              score_threshold=None,
                              recommend_threshold=opt.recommend_threshold)

    @catch_exception("Error when evaluating COCO mAP:")
    def save_eval_result(self):
        opt = self.opt
        metric_stats = self.metric_stats
        dataset_cfg = self.dataset_cfg
        dataset = self.dataset
        if not opt.save_json or not metric_stats.pred_json:
            return COCOResult()
        anno_json = self.get_val_anno()
        ckpt_name = Path(opt.weights).stem if opt.weights is not None else ''  # weights
        pred_json_path = os.path.join(opt.save_dir, f"{ckpt_name}_predictions_{opt.rank}.json")  # predictions json
        print(f'Evaluating pycocotools mAP... saving {pred_json_path}...')
        save_json(metric_stats.pred_json, pred_json_path)
        with SynchronizeManager(opt.rank % 8, min(8, opt.rank_size), opt.distributed_eval, opt.project_dir):
            result = COCOResult()
            if opt.rank % 8 == 0:
                pred_json = metric_stats.pred_json
                if opt.distributed_eval:
                    pred_json_path, pred_json = self.merge_pred_json(prefix=ckpt_name)
                if opt.result_view or opt.recommend_threshold:
                    self.visualize_coco(anno_json, pred_json_path)
                result = self.eval_coco(anno_json, pred_json, dataset_cfg.is_coco, dataset=dataset)
                print(f"\nCOCO mAP:\n{result.stats_str}")
            coco_result = result
        return coco_result

    def save_map(self, coco_result):
        opt = self.opt
        dataset_cfg = self.dataset_cfg
        if self.model.training or opt.rank % 8 != 0:
            return
        s = f"\n{len(glob.glob(os.path.join(opt.save_dir, 'labels/*.txt')))} labels saved to " \
            f"{os.path.join(opt.save_dir, 'labels')}" if opt.save_txt else ''
        print(f"Results saved to {opt.save_dir}, {s}")
        with open("class_map.txt", "w") as file:
            file.write(f"COCO map:\n{coco_result.stats_str}\n")
            if coco_result.category_stats_strs:
                for idx, category_str in enumerate(coco_result.category_stats_strs):
                    file.write(f"\nclass {dataset_cfg.names[idx]}:\n{category_str}\n")

    def get_val_anno(self):
        opt = self.opt
        dataset_cfg = self.dataset_cfg
        data_dir = Path(dataset_cfg.val).parent
        anno_json = os.path.join(data_dir, "annotations/instances_val2017.json")
        if opt.transfer_format and not os.path.exists(anno_json):
            # data format transfer if annotations does not exists
            print("Transfer annotations from yolo to coco format.")
            transformer = YOLO2COCO(data_dir, output_dir=data_dir,
                                    class_names=dataset_cfg.names, class_map=dataset_cfg.cls_map,
                                    mode='val', annotation_only=True)
            transformer()
        return anno_json

    def merge_pred_json(self, prefix=''):
        opt = self.opt
        print("Merge detection results...")
        merged_json = os.path.join(opt.project_dir, f"{prefix}_predictions_merged.json")
        merged_result = []
        # Waiting
        while True:
            json_files = list(Path(opt.project_dir).rglob("*.json"))
            if len(json_files) != min(8, opt.rank_size):
                time.sleep(1)
                print("Waiting for json file...")
            else:
                break
        for json_file in json_files:
            print(f"Merge {json_file.resolve()}")
            with open(json_file, "r") as file_handler:
                merged_result.extend(json.load(file_handler))
        with open(merged_json, "w") as file_handler:
            json.dump(merged_result, file_handler)
        print(f"Merged results saved in {merged_json}.")
        return merged_json, merged_result
