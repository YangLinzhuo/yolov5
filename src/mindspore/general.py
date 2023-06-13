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

from __future__ import annotations

import os
import time

import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import ops

from src.general import LOGGER

_true = ms.Tensor(True, ms.bool_)


def all_finite_cpu(inputs):
    return _true


class AllReduce(nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ops.AllReduce(op=ops.ReduceOp.SUM)

    def construct(self, x):
        return self.all_reduce(x)


class Synchronize:
    def __init__(self, rank_size):
        self.all_reduce = AllReduce()
        self.rank_size = rank_size

    def __call__(self):
        sync = ms.Tensor(np.array([1]).astype(np.int32))
        sync = self.all_reduce(sync)  # For synchronization
        sync = sync.asnumpy()[0]
        if sync != self.rank_size:
            raise ValueError(
                f"Sync value {sync} is not equal to number of device {self.rank_size}. "
                f"There might be wrong with devices."
            )


class SynchronizeManager:
    def __init__(self, rank, rank_size, distributed, project_dir):
        self.rank = rank
        self.rank_size = rank_size
        self.distributed = distributed  # whether distributed or not
        self.sync = Synchronize(rank_size) if (distributed and rank_size > 1) else None
        self.sync_file = os.path.join(project_dir, 'sync_file.temp')

    def __enter__(self):
        if self.distributed:
            if self.rank == 0:
                LOGGER.info(f"Create sync file {self.sync_file}")
                os.mknod(self.sync_file)
            if self.sync is not None:
                self.sync()
        return self.sync_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.distributed:
            return
        if self.rank == 0:
            if os.path.exists(self.sync_file):
                LOGGER.info(f"Delete sync file {self.sync_file}")
                os.remove(self.sync_file)
        else:
            LOGGER.info(f"Waiting for rank [0] device...")
            while os.path.exists(self.sync_file):
                time.sleep(1)
            LOGGER.info(f"Rank [{self.rank}] continue executing.")
