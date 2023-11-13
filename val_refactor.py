from __future__ import annotations

import codecs
import glob
import json
import os
import time
import yaml
from collections import namedtuple
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops
from mindspore.communication.management import get_group_size, get_rank
from pycocotools.coco import COCO

from config.val_args import ValConfig, DatasetConfig, get_val_args
from src.coco_visual import CocoVisualUtil
from src.dataset import create_dataloader
from src.metrics import (ConfusionMatrix, non_max_suppression, ap_per_class, scale_coords,
                         box_iou)
from src.general import COCOEval as COCOeval
from src.general import (increment_path, check_img_size, colorstr, coco80_to_coco91_class, xyxy2xywh, xywh2xyxy,\
                        AllReduce, Synchronize, SynchronizeManager)
from src.network.yolo import Model
from src.plots import output_to_target, plot_images, plot_study_txt
from third_party.yolo2coco.yolo2coco import YOLO2COCO


class Dict(dict):
    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, item):
        return self.__getitem__(item)


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
        self.confusion_matrix = None
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


def check_file(file):
    # Search for file if not found
    if Path(file).is_file() or file == '':
        return file
    files = glob.glob('./**/' + file, recursive=True)  # find file
    assert files, f'File Not Found: {file}'  # assert file was found
    assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
    return files[0]  # return file


def configure_env(opt: ValConfig):
    ms_mode = ms.GRAPH_MODE if opt.ms_mode == "graph" else ms.PYNATIVE_MODE
    ms.context.set_context(mode=ms_mode, device_target=opt.device_target)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        ms.context.set_context(device_id=device_id)
    rank, rank_size, parallel_mode = 0, 1, ms.ParallelMode.STAND_ALONE
    # Distribute Test
    if opt.distributed_eval:
        ms.communication.management.init()
        rank, rank_size, parallel_mode = get_rank() % 8, min(8, get_group_size()), ms.ParallelMode.DATA_PARALLEL
    ms.context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)
    opt.rank_size, opt.rank = rank_size, rank


def get_dataset_cfg(opt: ValConfig):
    is_coco = False
    if isinstance(opt.data, str):
        is_coco = opt.data.endswith('coco.yaml')
        with open(opt.data) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
    elif isinstance(opt.data, dict):
        cfg = opt.data
    else:
        raise TypeError("The type of opt.data must be str or dict.")
    if opt.single_cls:
        cfg['nc'] = 1
    cfg['train'] = os.path.join(cfg['root'], cfg['train'])
    cfg['val'] = os.path.join(cfg['root'], cfg['val'])
    cfg['test'] = os.path.join(cfg['root'], cfg['test'])
    dataset_cfg = DatasetConfig(**cfg, is_coco=is_coco)
    return dataset_cfg


def create_folders(opt: ValConfig, cur_epoch=None):
    if cur_epoch is not None:
        opt.project_dir = os.path.join(opt.project, f"epoch_{cur_epoch}")
    else:
        opt.project_dir = opt.project
    opt.save_dir = os.path.join(opt.project_dir, f"save_dir_{opt.rank}")
    opt.save_dir = increment_path(opt.save_dir, exist_ok=opt.exist_ok)
    os.makedirs(os.path.join(opt.save_dir, f"labels_{opt.rank}"), exist_ok=opt.exist_ok)


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


def configure_model(opt: ValConfig, dataset_cfg, model=None):
    is_training = model is not None
    if model is None:  # called in training process
        # Load model and hyperparameters
        with open(opt.hyp) as f:
            hyper_params = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        model = Model(opt.cfg, ch=3, nc=dataset_cfg['nc'], anchors=hyper_params.get('anchors'),
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
    return is_training, model


def configure_dataset(opt: ValConfig, dataset_cfg, dataset=None, dataloader=None):
    if dataloader is None or dataset is None:
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test img
        dataloader, dataset, per_epoch_size = create_dataloader(dataset_cfg[task], opt.img_size, opt.batch_size,
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
    return dataloader, dataset


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


def write_txt(opt: ValConfig, pred, shape, path):
    # Save result to txt
    path = Path(path)
    file_path = os.path.join(opt.save_dir, 'labels', f'{path.stem}.txt')
    gn = np.array(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in pred.tolist():
        xywh = (xyxy2xywh(np.array(xyxy).reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
        with open(file_path, 'w') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


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


def eval_metric(opt: ValConfig, confusion_matrix, out: list[np.ndarray], metric_stats: MetricStatistics, cls_map,
                data=None):
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
            write_txt(opt, pred_copy, shape, path)
        if opt.save_json:
            write_json_list(cls_map, pred_copy, metric_stats.pred_json, path)
    metric_duration = time.time() - metric_start_time
    return metric_duration


def model_step(opt: ValConfig, model, img: np.ndarray, compute_loss=None):
    img = Tensor.from_numpy(img)
    if opt.half_precision:
        img = ops.cast(img, ms.float16)
    # inference and training outputs
    if compute_loss or not opt.augment:
        pred_out, train_out = model(img)
    else:
        pred_out, train_out = (model(img, augment=opt.augment), None)
    return pred_out, train_out


def loss_step(opt: ValConfig, train_out, targets: np.ndarray, compute_loss=None):
    targets = Tensor.from_numpy(targets)
    if not compute_loss:
        return 0.
    if opt.half_precision:
        targets = ops.cast(targets, ms.float16)
    loss = compute_loss(train_out, targets)[1][:3].asnumpy()  # box, obj, cls
    return loss


def run_eval(opt: ValConfig, dataset_cfg: DatasetConfig, model, dataloader, compute_loss=None):
    loss = np.zeros(3)
    step_start_time = time.time()
    time_stats = TimeStatistics()
    metric_stats = MetricStatistics()
    confusion_matrix = ConfusionMatrix(nc=dataset_cfg.nc)
    for idx, data in enumerate(dataloader):
        # targets: Nx6 ndarray, img_id, label, x, y, w, h
        img, targets, paths, shapes = data["img"], data["label_out"], data["img_files"], data["shapes"]
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        data_duration = time.time() - step_start_time

        # Run model
        infer_start_time = time.time()
        pred_out, train_out = model_step(opt, model, img, compute_loss)
        infer_duration = time.time() - infer_start_time
        time_stats.infer += infer_duration

        # Compute loss
        loss += loss_step(opt, train_out, targets, compute_loss)

        # NMS
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
        nms_duration = time.time() - nms_start_time
        time_stats.nms += nms_duration

        # Metrics
        metric_duration = eval_metric(opt, confusion_matrix, out, metric_stats, dataset_cfg.cls_map,
                                      data=(img, targets, paths, shapes))
        time_stats.metric += metric_duration

        # Plot images
        if opt.plots and idx < 3:
            labels_path = os.path.join(opt.save_dir, f'test_batch{idx}_labels.jpg')  # labels
            plot_images(img, targets, paths, labels_path, dataset_cfg.names)
            pred_path = os.path.join(opt.save_dir, f'test_batch{idx}_pred.jpg')  # predictions
            plot_images(img, output_to_target(out), paths, pred_path, dataset_cfg.names)

        print(f"Step {idx + 1}/{opt.per_epoch_size} "
              f"Time total {(time.time() - step_start_time):.2f}s  "
              f"Data {data_duration * 1e3:.2f}ms  "
              f"Infer {infer_duration * 1e3:.2f}ms  "
              f"NMS {nms_duration * 1e3:.2f}ms  "
              f"Metric {metric_duration * 1e3:.2f}ms")

        step_start_time = time.time()
    metric_stats.loss = loss / opt.per_epoch_size

    # Plots
    if opt.plots:
        plot_confusion_matrix(opt, dataset_cfg, confusion_matrix)

    return metric_stats, time_stats


def merge_pred_stats(opt: ValConfig, metric_stats: MetricStatistics):
    # Merge prediction stats
    project_dir = Path(opt.save_dir).parent
    pred_stats: list[list] = [[] for _ in range(len(metric_stats.pred_stats))]
    for file_path in project_dir.rglob("pred_stats*.npy"):
        stats = np.load(str(file_path.resolve()), allow_pickle=True)
        for i, item in enumerate(stats):
            pred_stats[i].append(item)
    return pred_stats


def compute_map_stats(opt: ValConfig, dataset_cfg: DatasetConfig, metric_stats: MetricStatistics, is_training):
    # Compute metrics
    # pred_stats: list[np.ndarray], np.concatenate((correct, conf, pred_cls, target_cls), 0)
    metric_stats.pred_stats = [np.concatenate(x, 0) for x in zip(*metric_stats.pred_stats)]  # to numpy
    pred_stats_file = os.path.join(opt.save_dir, f"pred_stats_{opt.rank}.npy")
    np.save(pred_stats_file, np.array(metric_stats.pred_stats, dtype=object), allow_pickle=True)
    reduce_sum = AllReduce()
    synchronize = Synchronize(opt.rank_size)
    if opt.distributed_eval:
        metric_stats.seen = reduce_sum(ms.Tensor(np.array(metric_stats.seen, dtype=np.int32))).asnumpy()
        synchronize()
    if opt.rank % 8 != 0:
        return

    pred_stats: list[list] = merge_pred_stats(opt, metric_stats)
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
    if (opt.verbose or (nc < 50 and not is_training)) and nc > 1 and pred_stats:
        for i, c in enumerate(metric_stats.ap_cls):
            # Class     Images  Instances          P          R      mAP50   mAP50-95:
            print(pf.format(names[c], seen, nt[c], *metric_stats.get_ap_per_class(i)))


def print_stats(opt: ValConfig, metric_stats: MetricStatistics, time_stats: TimeStatistics):
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


def plot_confusion_matrix(opt: ValConfig, dataset_cfg: DatasetConfig, confusion_matrix):
    matrix = ms.Tensor(confusion_matrix.matrix)
    names = dataset_cfg.names if isinstance(dataset_cfg.names, list) else list(dataset_cfg.names.values())
    if opt.distributed_eval:
        matrix = AllReduce()(matrix).asnumpy()
    confusion_matrix.matrix = matrix
    if opt.rank % 8 == 0:
        confusion_matrix.plot(save_dir=opt.save_dir, names=names)


def get_val_anno(opt: ValConfig, dataset_cfg):
    data_dir = Path(dataset_cfg["val"]).parent
    anno_json = os.path.join(data_dir, "annotations/instances_val2017.json")
    if opt.transfer_format and not os.path.exists(anno_json):
        # data format transfer if annotations does not exists
        print("Transfer annotations from yolo to coco format.")
        transformer = YOLO2COCO(data_dir, output_dir=data_dir,
                                class_names=dataset_cfg.names, class_map=dataset_cfg.cls_map,
                                mode='val', annotation_only=True)
        transformer()
    return anno_json


def save_json(pred_json, save_path):
    with open(save_path, 'w') as file:
        json.dump(pred_json, file)


def merge_pred_json(opt: ValConfig, prefix=''):
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


def visualize_coco(opt: ValConfig, dataset_cfg: DatasetConfig, anno_json, pred_json_path):
    print("Start visualization result.")
    dataset_coco = COCO(anno_json)
    coco_visual = CocoVisualUtil()
    eval_types = ["bbox"]
    config = {"dataset": "coco"}
    data_dir = Path(dataset_cfg.val).parent
    img_path_name = os.path.splitext(os.path.basename(dataset_cfg.val))[0]
    im_path_dir = os.path.join(data_dir, "images", img_path_name)
    with open(pred_json_path, 'r') as f:
        result = json.load(f)
    result_files = coco_visual.results2json(dataset_coco, result, "./results.pkl")
    coco_visual.coco_eval(Dict(config), result_files, eval_types, dataset_coco, im_path_dir=im_path_dir,
                          score_threshold=None,
                          recommend_threshold=opt.recommend_threshold)


def eval_coco(anno_json, pred_json, is_coco, dataset=None):
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


def save_eval_result(opt: ValConfig, metric_stats, dataset_cfg: DatasetConfig, dataset=None):
    anno_json = get_val_anno(opt, dataset_cfg)
    ckpt_name = Path(opt.weights).stem if opt.weights is not None else ''  # weights
    pred_json_path = os.path.join(opt.save_dir, f"{ckpt_name}_predictions_{opt.rank}.json")  # predictions json
    print(f'Evaluating pycocotools mAP... saving {pred_json_path}...')
    save_json(metric_stats.pred_json, pred_json_path)
    with SynchronizeManager(opt.rank % 8, min(8, opt.rank_size), opt.distributed_eval, opt.project_dir):
        result = COCOResult()
        if opt.rank % 8 == 0:
            pred_json = metric_stats.pred_json
            if opt.distributed_eval:
                pred_json_path, pred_json = merge_pred_json(opt, prefix=ckpt_name)
            if opt.result_view or opt.recommend_threshold:
                try:
                    visualize_coco(opt, dataset_cfg, anno_json, pred_json_path)
                except Exception:
                    print("Failed when visualize evaluation result.")
                    import traceback
                    traceback.print_exc()
            try:
                result = eval_coco(anno_json, pred_json, dataset_cfg.is_coco, dataset=dataset)
                print(f"\nCOCO mAP:\n{result.stats_str}")
            except Exception:
                print("Exception when running pycocotools")
                import traceback
                traceback.print_exc()
        coco_result = result
    return coco_result


def save_map(opt: ValConfig, dataset_cfg, coco_result):
    s = f"\n{len(glob.glob(os.path.join(opt.save_dir, 'labels/*.txt')))} labels saved to " \
        f"{os.path.join(opt.save_dir, 'labels')}" if opt.save_txt else ''
    print(f"Results saved to {opt.save_dir}, {s}")
    with open("class_map.txt", "w") as file:
        file.write(f"COCO map:\n{coco_result.stats_str}\n")
        if coco_result.category_stats_strs:
            for idx, category_str in enumerate(coco_result.category_stats_strs):
                file.write(f"\nclass {dataset_cfg['names'][idx]}:\n{category_str}\n")


def val(opt: ValConfig, model=None, dataset=None, dataloader=None,
        cur_epoch=None, compute_loss=None):
    dataset_cfg = get_dataset_cfg(opt)
    create_folders(opt, cur_epoch)
    is_training, model = configure_model(opt, dataset_cfg, model)
    dataloader, dataset = configure_dataset(opt, dataset_cfg, dataset, dataloader)

    dataset_cfg.names = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))
    dataset_cfg.cls_start_idx = 1
    dataset_cfg.cls_map = coco80_to_coco91_class() if dataset_cfg.is_coco \
        else list(range(dataset_cfg.cls_start_idx, 1000 + dataset_cfg.cls_start_idx))

    # Test
    metric_stats, time_stats = run_eval(opt, dataset_cfg, model, dataloader, compute_loss)
    compute_map_stats(opt, dataset_cfg, metric_stats, is_training)

    # Print speeds
    speed = print_stats(opt, metric_stats, time_stats)

    coco_result = COCOResult()
    # Save JSON
    if opt.save_json and metric_stats.pred_json:
        try:
            coco_result = save_eval_result(opt, metric_stats, dataset_cfg)
        except Exception as e:
            import traceback
            print("Error when evaluating COCO mAP:")
            traceback.print_exc()

    # Return results
    if not is_training and opt.rank % 8 == 0:
        save_map(opt, dataset_cfg, coco_result)
    maps = np.zeros(dataset_cfg.nc) + coco_result.get_map()
    if opt.rank % 8 == 0:
        for i, c in enumerate(metric_stats.ap_cls):
            maps[c] = metric_stats.ap[i]

    model.set_train()
    val_result = namedtuple('ValResult', ['metric_stats', 'maps', 'speed', 'coco_result'])
    return val_result(metric_stats, maps, speed, coco_result)


def main():
    opt: ValConfig = get_val_args()
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.data, opt.cfg, opt.hyp = map(check_file, (opt.data, opt.cfg, opt.hyp))  # check files
    configure_env(opt)

    if opt.task in ('train', 'val', 'test'):  # run normally
        opt.save_txt = opt.save_txt | opt.save_hybrid
        val(opt)


if __name__ == "__main__":
    main()
