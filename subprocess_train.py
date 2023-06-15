from copy import deepcopy
from multiprocessing import Process
import os
from pathlib import Path

from src.config.args import get_args_train, TrainConfig
from src.config.data import DatasetConfig
from src.config.hyp import get_hyp, Hyp
from src.dataset.dataset import Dataset
from src.general import check_file, empty, LOGGER, increment_path


def subprocess_train(hyp: Hyp, opt: TrainConfig, dataset_cfg: DatasetConfig, dataset: Dataset):
    # Export environment variable
    # os.environ["DEVICE_ID"] = str(opt.device_id)
    print(os.environ["DEVICE_ID"])
    print(os.environ["RANK_ID"])

    from mindspore.profiler.profiling import Profiler
    from src.train.manager import TrainManager
    from mindspore.communication.management import get_group_size, get_rank, init
    from mindspore import context
    from mindspore.context import ParallelMode

    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode, device_target=opt.device_target, save_graphs=False)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', "0"))
        context.set_context(device_id=device_id)
    # Distribute Train
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if opt.distributed_train:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size,
                                          all_reduce_fusion_config=[10, 70, 130, 190, 250, 310])

    opt.rank, opt.rank_size = rank, rank_size
    opt.total_batch_size = opt.batch_size * opt.rank_size

    # Train
    profiler = None
    if opt.profiler:
        profiler = Profiler()

    if not opt.evolve:
        LOGGER.info(f"OPT: {opt}")
        train_manager = TrainManager(hyp, opt, dataset_cfg, dataset)
        train_manager.train()
    else:
        raise NotImplementedError("Not support evolve train")

    if opt.profiler:
        profiler.analyse()


def judge_multi_train(processes):
    exitcode = 0
    for p in processes:
        p.join()
        if p.exitcode:
            exitcode = p.exitcode
        p.close()
    return exitcode


def train():
    opt = get_args_train()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert not empty(opt.cfg) or not empty(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name

    # Hyperparameters
    hyp = get_hyp(opt.hyp)

    server_id = opt.server_id
    device_num = opt.device_num
    device_id = opt.device_id
    rank_id_start = device_num * server_id
    subprocesses = []

    # Load Cache and Images
    dataset_cfg = DatasetConfig(opt.data)
    # stride is from model config yaml file
    if isinstance(opt.img_size, list):
        img_size = opt.img_size[0]
    else:
        img_size = opt.img_size
    # Train dataset
    dataset = Dataset(dataset_cfg.train, stride=32, img_size=img_size, cache_images=opt.cache_images)
    if opt.distributed_train:
        os.environ["RANK_TABLE_FILE"] = os.path.realpath(opt.rank_table_file)
        os.environ["RANK_SIZE"] = str(opt.rank)

    from mindspore import context
    ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    # Must call set_context here, otherwise subprocess might terminate unexpectedly
    context.set_context(mode=ms_mode, device_target=opt.device_target, save_graphs=False)

    for i in range(device_num):
        opt_copy = deepcopy(opt)
        opt_copy.rank = rank_id_start + i
        opt_copy.device_id = device_id + i
        os.environ["RANK_ID"] = str(opt_copy.rank)
        os.environ["DEVICE_ID"] = str(opt_copy.device_id)
        LOGGER.info(f"start training for rank {opt_copy.rank}, device {opt_copy.device_id}")
        # print(f"start training for rank {opt_copy.rank}, device {opt_copy.device_id}")
        p = Process(target=subprocess_train, args=(hyp, opt_copy, dataset_cfg, dataset))
        p.start()
        subprocesses.append(p)
    exitcode = judge_multi_train(subprocesses)
    if exitcode:
        raise RuntimeError("Distributed train failed!")

def main():
    train()


if __name__ == "__main__":
    main()
