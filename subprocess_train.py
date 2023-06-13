from copy import deepcopy
from multiprocessing import Process
import os
import yaml

from src.config.args import get_args_train, TrainConfig
from src.config.data import DatasetConfig
from src.dataset.dataset import Dataset
from src.general import check_file, empty, LOGGER


class TrainManager:
    def __init__(self, hyp, opt):
        self.hyp = hyp
        self.opt = opt

        self.best_map = 0.
        self.data_cfg = None
        self.weight_dir = os.path.join(self.opt.save_dir, "weights")

    def train(self):
        pass


def subprocess_train(hyp, opt: TrainConfig):
    # Export environment variable
    os.environ["DEVICE_ID"] = str(opt.device_id)
    os.environ["RANK_ID"] = str(opt.rank)
    print(os.environ["DEVICE_ID"])
    print(os.environ["RANK_ID"])

    from mindspore.profiler.profiling import Profiler
    # Train
    # profiler = None
    # if opt.profiler:
    #     profiler = Profiler()
    #
    # if not opt.evolve:
    #     LOGGER.info(f"OPT: {opt}")
    #     train_manager = TrainManager(hyp, opt)
    #     train_manager.train()
    # else:
    #     raise NotImplementedError("Not support evolve train")
    #
    # if opt.profiler:
    #     profiler.analyse()


def train():
    opt = get_args_train()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert not empty(opt.cfg) or not empty(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    server_id = opt.server_id
    device_num = opt.device_num
    device_id = opt.device_id
    rank_id_start = device_num * server_id
    subprocesses = []

    # Load Cache and Images
    dataset_cfg = DatasetConfig(opt.data)
    dataset = Dataset(dataset_cfg.train)

    for i in range(device_num):
        opt_copy = deepcopy(opt)
        opt_copy.rank = rank_id_start + i
        opt_copy.device_id = device_id + i
        LOGGER.info(f"start training for rank {opt_copy.rank}, device {opt_copy.device_id}")
        # print(f"start training for rank {opt_copy.rank}, device {opt_copy.device_id}")
        p = Process(target=subprocess_train, args=(hyp, opt_copy))
        p.start()
        subprocesses.append(p)


def main():
    train()


if __name__ == "__main__":
    main()
