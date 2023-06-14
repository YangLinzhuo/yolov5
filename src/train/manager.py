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

from collections import deque
from contextlib import nullcontext
import copy
import math
import os
import random
import stat
import time
import yaml

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.profiler.profiling import Profiler

from src.config.data import DatasetConfig
from src.config.args import TrainConfig
from src.config.hyp import Hyp
from src.dataset.dataset import Dataset

from src.boost import build_train_network
from src.network.common import EMA
from src.network.loss import ComputeLoss
from src.network.yolo import Model
from src.optimizer import YoloMomentum, get_group_param, get_lr
from src.general import LOGGER, check_img_size, labels_to_class_weights
from src.dataset.dataloader import create_dataloader


WRITE_FLAGS = os.O_WRONLY | os.O_CREAT    # Default write flags
READ_FLAGS = os.O_RDONLY    # Default read flags
FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP   # Default file authority mode

@ops.constexpr
def _get_new_size(img_shape, gs, imgsz):
    sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs)) // gs * gs  # size
    sf = sz / max(img_shape[2:])  # scale factor
    new_size = img_shape
    if sf != 1:
        # new size (stretched to gs-multiple)
        # Use tuple because nn.interpolate only supports tuple `sizes` parameter must be tuple
        new_size = tuple(math.ceil(x * sf / gs) * gs for x in img_shape[2:])
    return new_size


def create_train_network(model, compute_loss, ema, optimizer, loss_scaler=None,
                         sens=1.0, enable_clip_grad=True, opt=None, gs=None, imgsz=None):
    class NetworkWithLoss(nn.Cell):
        def __init__(self, model, compute_loss, opt):
            super(NetworkWithLoss, self).__init__()
            self.model = model
            self.compute_loss = compute_loss
            self.rank_size = opt.rank_size
            self.lbox_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lbox_loss")
            self.lobj_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lobj_loss")
            self.lcls_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lcls_loss")
            self.multi_scale = opt.multi_scale if hasattr(opt, 'multi_scale') else False
            self.gs = gs
            self.imgsz = imgsz

        def construct(self, x, label, sizes=None):
            x /= 255.0
            if self.multi_scale and self.training:
                x = ops.interpolate(x, sizes=_get_new_size(x.shape, self.gs, self.imgsz),
                                    coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = self.model(x)
            loss, loss_items = self.compute_loss(pred, label)
            loss_items = ops.stop_gradient(loss_items)
            loss *= self.rank_size
            loss = F.depend(loss, ops.assign(self.lbox_loss, loss_items[0]))
            loss = F.depend(loss, ops.assign(self.lobj_loss, loss_items[1]))
            loss = F.depend(loss, ops.assign(self.lcls_loss, loss_items[2]))
            return loss

    LOGGER.info(f"rank_size: {opt.rank_size}")
    net_with_loss = NetworkWithLoss(model, compute_loss, opt)
    train_step = build_train_network(network=net_with_loss, ema=ema, optimizer=optimizer,
                                     level='O0', boost_level='O1', amp_loss_scaler=loss_scaler,
                                     sens=sens, enable_clip_grad=enable_clip_grad)
    return train_step


class CheckpointQueue:
    def __init__(self, max_ckpt_num):
        self.max_ckpt_num = max_ckpt_num
        self.ckpt_queue = deque()

    def append(self, ckpt_path):
        self.ckpt_queue.append(ckpt_path)
        if len(self.ckpt_queue) > self.max_ckpt_num:
            ckpt_to_delete = self.ckpt_queue.popleft()
            os.remove(ckpt_to_delete)


def save_ema(ema, ema_ckpt_path, append_dict=None):
    params_list = []
    for p in ema.ema_weights:
        _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
        params_list.append(_param_dict)
    ms.save_checkpoint(params_list, ema_ckpt_path, append_dict=append_dict)



class TrainManager:
    def __init__(self, hyp: Hyp, opt: TrainConfig, data_cfg: DatasetConfig, dataset: Dataset):
        self.hyp = hyp
        self.opt = opt
        self.data_cfg = data_cfg
        self.dataset = dataset

        self.best_map = 0.
        self.weight_dir = os.path.join(self.opt.save_dir, "weights")

    def set_seed(self, seed: int = 2):
        import numpy as np
        import random
        import mindspore as ms
        np.random.seed(seed)
        random.seed(seed)
        ms.set_seed(seed)

    def train(self):
        opt = self.opt
        hyp = self.hyp
        self._modelarts_sync(opt.data_url, opt.data_dir)

        num_cls = self.data_cfg.num_cls
        # Directories
        os.makedirs(self.weight_dir, exist_ok=True)
        # Save run settings
        self.dump_cfg()
        self._modelarts_sync(opt.save_dir, opt.train_url)

        # Model
        sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and opt.rank_size > 1
        # Create Model
        from dataclasses import asdict
        model = Model(opt.cfg, ch=3, nc=num_cls, sync_bn=sync_bn, opt=opt, hyp=asdict(hyp))
        model.to_float(ms.float16)
        ema = EMA(model) if opt.ema else None

        # Freeze
        model = self.freeze_layer(model)

        # Image sizes
        gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, _ = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
        # train_epoch_size = 1 if opt.optimizer == "thor" else opt.epochs - resume_epoch
        train_epoch_size = 1 if opt.optimizer == "thor" else opt.epochs
        dataloader, dataset, per_epoch_size = self.get_dataset(model, train_epoch_size, mode="train")
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        assert mlc < num_cls, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' \
                              % (mlc, num_cls, opt.data, num_cls - 1)

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / opt.total_batch_size), 1)  # accumulate loss before optimizing
        hyp.weight_decay *= opt.total_batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {hyp.weight_decay}")
        optimizer = self.get_optimizer(model, per_epoch_size, 0)

        # Model parameters
        model = self._configure_model_params(dataset, imgsz, model, nl)

        # Build train process function
        # amp
        ms.amp.auto_mixed_precision(model, amp_level=opt.ms_amp_level)
        compute_loss = ComputeLoss(model)  # init loss class
        ms.amp.auto_mixed_precision(compute_loss, amp_level=opt.ms_amp_level)
        loss_scaler = self.get_loss_scaler()
        train_step = self.get_train_step(compute_loss, ema, model, optimizer,
                                         gs=gs, imgsz=imgsz, loss_scaler=loss_scaler)
        model.set_train(True)
        optimizer.set_train(True)
        run_profiler_epoch = 2
        ema_ckpt_queue = CheckpointQueue(opt.max_ckpt_num)
        ckpt_queue = CheckpointQueue(opt.max_ckpt_num)

        data_size = dataloader.get_dataset_size()
        jit = opt.ms_mode.lower() == "graph"
        sink_process = ms.data_sink(train_step, dataloader, steps=data_size * opt.epochs, sink_size=data_size, jit=jit)

        summary_dir = os.path.join(opt.save_dir, opt.summary_dir, f"rank_{opt.rank}")
        steps_per_epoch = data_size
        with ms.SummaryRecord(summary_dir) if opt.summary else nullcontext() as summary_record:
            for cur_epoch in range(0, opt.epochs):
                cur_epoch = cur_epoch + 1
                start_train_time = time.time()
                loss = sink_process()
                end_train_time = time.time()
                step_time = end_train_time - start_train_time
                LOGGER.info(f"Epoch {opt.epochs - 0}/{cur_epoch}, step {data_size}, "
                            f"epoch time {step_time * 1000:.2f} ms, "
                            f"step time {step_time * 1000 / data_size:.2f} ms, "
                            f"loss: {loss.asnumpy() / opt.batch_size:.4f}, "
                            f"lbox loss: {train_step.network.lbox_loss.asnumpy():.4f}, "
                            f"lobj loss: {train_step.network.lobj_loss.asnumpy():.4f}, "
                            f"lcls loss: {train_step.network.lcls_loss.asnumpy():.4f}.")
                if opt.profiler and (cur_epoch == run_profiler_epoch):
                    break
                self.save_ckpt(ckpt_queue, cur_epoch, ema, ema_ckpt_queue, model)
        return 0

    def save_ckpt(self, ckpt_queue, cur_epoch, ema, ema_ckpt_queue, model):
        opt = self.opt

        def is_save_epoch():
            return (cur_epoch >= opt.start_save_epoch) and (cur_epoch % opt.save_interval == 0)

        if opt.save_checkpoint and (opt.rank % 8 == 0) and is_save_epoch():
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5]  # delete ".yaml"
            ckpt_path = os.path.join(self.weight_dir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path, append_dict={"epoch": cur_epoch})
            ckpt_queue.append(ckpt_path)
            self._modelarts_sync(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
            if ema:
                ema_ckpt_path = os.path.join(self.weight_dir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                append_dict = {"updates": ema.updates, "epoch": cur_epoch}
                save_ema(ema, ema_ckpt_path, append_dict)
                ema_ckpt_queue.append(ema_ckpt_path)
                LOGGER.info(f"Save ckpt path: {ema_ckpt_path}")
                self._modelarts_sync(ema_ckpt_path, opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])


    def get_train_step(self, compute_loss, ema, model, optimizer, gs, imgsz, loss_scaler):
        if self.opt.ms_strategy == "StaticShape":
            train_step = create_train_network(model, compute_loss, ema, optimizer,
                                              loss_scaler=loss_scaler, sens=self.opt.ms_grad_sens, opt=self.opt,
                                              enable_clip_grad=self.hyp["enable_clip_grad"],
                                              gs=gs, imgsz=imgsz)
        else:
            raise NotImplementedError
        return train_step

    def get_loss_scaler(self):
        opt = self.opt
        if opt.ms_loss_scaler == "dynamic":
            from mindspore.amp import DynamicLossScaler
            loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)
        elif opt.ms_loss_scaler == "static":
            from mindspore.amp import StaticLossScaler
            loss_scaler = StaticLossScaler(opt.ms_loss_scaler_value)
        else:
            loss_scaler = None
        return loss_scaler

    def _configure_model_params(self, dataset, imgsz, model, nl):
        hyp = self.hyp
        opt = self.opt
        num_cls = self.data_cfg.num_cls
        cls_names = self.data_cfg.cls_names
        hyp.box *= 3. / nl  # scale to layers
        hyp.cls *= num_cls / 80. * 3. / nl  # scale to classes and layers
        hyp.obj *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        hyp.label_smoothing = opt.label_smoothing
        model.nc = num_cls  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.class_weights = Tensor(labels_to_class_weights(dataset.labels, num_cls) * num_cls)  # attach class weights
        model.names = cls_names
        return model

    def get_optimizer(self, model, per_epoch_size, resume_epoch):
        opt = self.opt
        hyp = self.hyp
        pg0, pg1, pg2 = get_group_param(model)
        lr_pg0, lr_pg1, lr_pg2, momentum_pg, _ = get_lr(opt, hyp, per_epoch_size, resume_epoch)
        group_params = [
            {'params': pg0, 'lr': lr_pg0, 'weight_decay': hyp.weight_decay},
            {'params': pg1, 'lr': lr_pg1, 'weight_decay': 0.0},
            {'params': pg2, 'lr': lr_pg2, 'weight_decay': 0.0}]
        LOGGER.info(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
        if opt.optimizer == "sgd":
            optimizer = nn.SGD(group_params, learning_rate=hyp.lr0, momentum=hyp.momentum, nesterov=True,
                               loss_scale=opt.ms_optim_loss_scale)
        elif opt.optimizer == "momentum":
            optimizer = YoloMomentum(group_params, learning_rate=hyp.lr0, momentum=momentum_pg, use_nesterov=True,
                                     loss_scale=opt.ms_optim_loss_scale)
        elif opt.optimizer == "adam":
            optimizer = nn.Adam(group_params, learning_rate=hyp.lr0, beta1=hyp.momentum, beta2=0.999,
                                loss_scale=opt.ms_optim_loss_scale)
        else:
            raise NotImplementedError
        return optimizer

    def get_dataset(self, model, epoch_size, mode="train"):
        opt = self.opt
        gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
        imgsz, _ = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

        dataloader, dataset, per_epoch_size = create_dataloader(self.dataset, self.hyp, self.opt,
                                                                shuffle=True, image_weights=opt.image_weights,
                                                                num_parallel_workers=12, drop_remainder=True,
                                                                is_training=True)
        return dataloader, dataset, per_epoch_size


    def freeze_layer(self, model):
        freeze = self.opt.freeze
        # parameter names to freeze (full or partial)
        freeze = freeze if len(freeze) > 1 else range(freeze[0])
        freeze = [f'model.{x}.' for x in freeze]
        for n, p in model.parameters_and_names():
            if any(x in n for x in freeze):
                LOGGER.info(f'freezing {n}')
                p.requires_grad = False
        return model

    def dump_cfg(self):
        if self.opt.rank != 0:
            return
        save_dir = self.opt.save_dir
        with os.fdopen(os.open(os.path.join(save_dir, "hyp.yaml"), WRITE_FLAGS, FILE_MODE), 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with os.fdopen(os.open(os.path.join(save_dir, "opt.yaml"), WRITE_FLAGS, FILE_MODE), 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)

    def _modelarts_sync(self, src_dir, dst_dir):
        if not self.opt.enable_modelarts:
            return
        from src.modelarts import sync_data
        os.makedirs(dst_dir, exist_ok=True)
        sync_data(src_dir, dst_dir)
