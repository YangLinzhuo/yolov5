from dataclasses import dataclass
from dataclasses import field as data_field
from typing import List, Optional, Any, Dict
from config.base import MsArgumentParser, Config


def meta(help_msg: Optional[str] = None, choices: Optional[List[Any]] = None,
         action: Optional[str] = None):
    if help_msg is None and choices is None and action is None:
        return None
    meta_data: Dict[str, Any] = {}
    if help_msg is not None:
        meta_data["help"] = help_msg
    if choices is not None:
        meta_data["choices"] = choices
    if action is not None:
        meta_data["action"] = action
    return meta_data


@dataclass
class BasicConfig(Config):
    ms_mode: str = data_field(default="graph", metadata=meta(choices=["graph", "pynative"]))
    device: str = data_field(
        default="ascend",
        metadata=meta(help_msg="device target, Ascend/GPU/CPU", choices=["cpu", "gpu", "ascend"])
    )
    cfg: str = data_field(default="config/network/yolov5s.yaml", metadata=meta(help_msg="model.yaml path"))
    data: str = data_field(default="config/data/coco.yaml", metadata=meta(help_msg="data.yaml path"))
    hyp: str = data_field(
        default="config/data/hyp.scratch-low.yaml",
        metadata=meta(help_msg="hyper-parameters path")
    )
    batch_size: int = data_field(default=32, metadata=meta(help_msg="batch size for each device"))


@dataclass
class InferBasicConfig(Config):
    verbose: bool = data_field(default=False, metadata=meta("report mAP by class"))
    augment: bool = data_field(default=False, metadata=meta("augmented inference"))
    single_cls: bool = data_field(default=False, metadata=meta(help_msg="treat as single-class dataset"))
    conf_thres: float = data_field(default=0.001, metadata=meta(help_msg="object confidence threshold"))
    iou_thres: float = data_field(default=0.65, metadata=meta(help_msg="IOU threshold for NMS"))
    task: str = data_field(
        default="val",
        metadata=meta(
            help_msg="train, val, test, speed or study",
            choices=["train", "val", "test", "speed", "study"]
        )
    )
    img_size: int = data_field(default=640, metadata=meta(help_msg="inference size (pixels)"))


@dataclass
class EvalConfig(BasicConfig, InferBasicConfig):
    is_distributed: bool = data_field(default=False, metadata=meta(help_msg="Distribute test or not"))
    weights: str = data_field(default="./EMA_yolov5s_300.ckpt", metadata=meta(help_msg="model.ckpt path(s)"))
    rect: bool = data_field(default=False, metadata=meta(help_msg="rectangular training"))
    save_txt: bool = data_field(default=False, metadata=meta(help_msg="save results to *.txt"))
    save_hybrid: bool = data_field(
        default=False,
        metadata=meta(help_msg="save label+prediction hybrid results to *.txt'")
    )
    save_conf: bool = data_field(
        default=False,
        metadata=meta(help_msg="save confidences in --save-txt labels")
    )
    save_json: bool = data_field(
        default=False,
        metadata=meta(help_msg="save a cocoapi-compatible JSON results file")
    )
    project: str = data_field(default="./run_test", metadata=meta(help_msg="save to project/name"))
    exist_ok: bool = data_field(
        default=False,
        metadata=meta(help_msg="existing project/name ok, do not increment")
    )
    trace: bool = data_field(default=False, metadata=meta(help_msg="trace model"))
    plots: bool = data_field(default=True, metadata=meta(help_msg="enable plot"))
    v5_metric: bool = data_field(
        default=False,
        metadata=meta(help_msg="assume maximum recall as 1.0 in AP calculation")
    )
    transfer_format: bool = data_field(
        default=True,
        metadata=meta(help_msg="whether transform data format to coco")
    )
    result_view: bool = data_field(default=False, metadata=meta(help_msg="view the eval result"))
    recommend_threshold: bool = data_field(default=False, metadata=meta(help_msg="recommend threshold in eval"))


@dataclass
class TrainConfig(EvalConfig):
    ms_strategy: str = data_field(
        default="StaticShape",
        metadata=meta(
            help_msg="train strategy, StaticCell/StaticShape/MultiShape/DynamicShape",
            choices=["StaticCell", "StaticShape", "MultiShape", "DynamicShape"]
        )
    )
    ms_amp_level: str = data_field(
        default="O0",
        metadata=meta(help_msg="amp level", choices=["O0", "O1", "O2"])
    )
    ms_loss_scaler: Optional[str] = data_field(
        default=None,
        metadata=meta(help_msg="Train loss scaler", choices=["static", "dynamic", "None"])
    )
    ms_loss_scaler_value: float = data_field(default=1.0, metadata=meta(help_msg="static loss scale value"))
    ms_optim_loss_scale: float = data_field(default=1.0, metadata=meta(help_msg="optimizer loss scale"))
    ms_grad_sens: float = data_field(default=1024, metadata=meta(help_msg="grad sens"))
    accumulate: bool = data_field(default=False, metadata=meta(help_msg="accumulate gradient"))
    overflow_still_update: bool = data_field(
        default=False,
        metadata=meta(help_msg="When overflow happens, still update weight")
    )
    clip_grad: bool = data_field(default=False, metadata=meta(help_msg="Clip gradient"))
    profiler: bool = data_field(default=False, metadata=meta(help_msg="Enable profiler"))
    ema: bool = data_field(default=True, metadata=meta(help_msg="Enable ema"))
    recompute: bool = data_field(default=False, metadata=meta(help_msg="Enable recompute"))
    recompute_layers: int = 0
    weights: str = data_field(default='', metadata=meta(help_msg="Initial weights path"))
    ema_weight: str = data_field(default='', metadata=meta(help_msg="Initial ema weights path"))
    epochs: int = 300
    img_size: List[int] = data_field(   # type: ignore
        default_factory=lambda: [640, 640],
        metadata=meta(help_msg="[Train, Eval] image sizes")
    )

    save_checkpoint: bool = data_field(default=True, metadata=meta(help_msg="Enable save checkpoint"))
    start_save_epoch: int = data_field(
        default=100,
        metadata=meta(help_msg="Epoch to start saving checkpoint")
    )
    save_interval: int = data_field(default=5, metadata=meta(help_msg="Epoch interval to save checkpoint"))
    max_ckpt_num: int = data_field(
        default=40,
        metadata=meta(help_msg="The maximum number of save checkpoint, delete previous checkpoints if "
                                    "the number of saved checkpoints are larger than this value")
    )
    resume: bool = data_field(default=False, metadata=meta(help_msg="Resume specified checkpoint training"))
    nosave: bool = data_field(default=False, metadata=meta(help_msg="Only save final checkpoint"))
    notest: bool = data_field(default=False, metadata=meta(help_msg="Only test final epoch"))
    noautoanchor: bool = data_field(default=False, metadata=meta(help_msg="Disable autoanchor check"))
    evolve: bool = data_field(default=False, metadata=meta("Evolve hyper-parameters"))
    bucket: str = data_field(default='', metadata=meta(help_msg="gsutil bucket"))
    cache_images: str = data_field(
        default='',
        metadata=meta(help_msg="Cache images for faster training", choices=['', 'ram', 'disk'])
    )
    image_weights: bool = data_field(
        default=False,
        metadata=meta(help_msg="Use weighted image selection for training")
    )
    multi_scale: bool = data_field(default=False, metadata=meta(help_msg="Vary img-size +/- 50%%"))
    optimizer: str = data_field(
        default="sgd",
        metadata=meta(help_msg="Select optimizer", choices=["sgd", "momentum", "adam"])
    )
    sync_bn: bool = data_field(
        default=False,
        metadata=meta(help_msg="Use SyncBatchNorm, only available in DDP mode")
    )
    project: str = data_field(default="runs/train", metadata=meta(help_msg="Save to project/name"))
    entity: Optional[str] = data_field(default=None, metadata=meta(help_msg="W&B entity"))
    name: str = data_field(default="exp", metadata=meta(help_msg="Save to project/name"))
    quad: bool = data_field(default=False, metadata=meta(help_msg="Quad dataloader"))
    linear_lr: bool = data_field(default=True, metadata=meta(help_msg="Linear LR"))
    result_view: bool = data_field(default=False, metadata=meta(help_msg="View the eval result"))
    label_smoothing: float = data_field(default=0.0, metadata=meta(help_msg="Label smoothing epsilon"))
    upload_dataset: bool = data_field(default=False, metadata=meta(help_msg="Upload dataset as W&B artifact table"))
    bbox_interval: int = data_field(
        default=-1,
        metadata=meta(help_msg="Set bounding-box image logging interval for W&B")
    )
    save_period: int = data_field(default=-1, metadata=meta(help_msg="Log model after every 'save_period' epoch"))
    artifact_alias: str = data_field(
        default="latest",
        metadata=meta(help_msg="version of dataset artifact to be used")
    )
    freeze: List[int] = data_field(
        default_factory=lambda: [0],
        metadata=meta(help_msg="Freeze layers: backbone of yolov5, first3=0 1 2")
    )
    summary: bool = data_field(
        default=False,
        metadata=meta(help_msg="Whether use SummaryRecord to log intermediate data")
    )
    summary_dir: str = data_field(
        default="summary",
        metadata=meta(help_msg="Folder to save summary files with project/summary_dir structure")
    )
    summary_interval: int = data_field(default=-1, metadata=meta(help_msg="Epoch interval to save summary files"))

    # Args for evaluation
    run_eval: bool = data_field(default=True, metadata=meta(help_msg="Enable evaluation after some epoch"))
    eval_start_epoch: int = data_field(default=200, metadata=meta(help_msg="Epoch to do evaluation"))
    eval_epoch_interval: int = data_field(default=10, metadata=meta(help_msg="Epoch interval to do evaluation"))

    # Args for ModelArts
    enable_modelarts: bool = data_field(default=False, metadata=meta(help_msg="Enable modelarts"))
    data_url: str = data_field(default="", metadata=meta(help_msg="ModelArts: obs path to dataset folder"))
    train_url: str = data_field(default="", metadata=meta(help_msg="ModelArts: obs path to dataset folder"))
    data_dir: str = data_field(
        default="/cache/data",
        metadata=meta(help_msg="ModelArts: obs path to dataset folder")
    )


@dataclass
class InferConfig(BasicConfig, InferBasicConfig):
    om: str = data_field(default="yolov5s.om", metadata=meta(help_msg="model.om path"))
    # Currently not support enabling rect for inference
    rect: bool = data_field(default=False, metadata=meta(help_msg="Enable rectangular image processing"))
    output_dir: str = data_field(default="./output", metadata=meta(help_msg="Folder path to save prediction json"))


@dataclass
class ExportConfig(BasicConfig):
    batch_size: int = data_field(default=1, metadata=meta(help_msg="Size of each image batch"))
    img_size: int = data_field(default=640, metadata=meta(help_msg="Inference size (pixels)"))
    file_format: str = data_field(default="MINDIR", metadata=meta(help_msg="Treat as single-class dataset"))
    output_path: str = data_field(default="./", metadata=meta(help_msg="Output preprocess data path"))
    weights: str = data_field(default="./EMA_yolov5s_300.ckpt", metadata=meta(help_msg="model.ckpt path"))


def get_args(data_class):
    parser = MsArgumentParser(data_class)
    args = parser.parse_args()
    return data_class(**vars(args))


def get_args_export() -> ExportConfig:
    return get_args(ExportConfig)


def main():
    data_class = TrainConfig
    parser = MsArgumentParser(data_class)   # type: ignore
    args = parser.parse_args()
    print(data_class(**vars(args)))


if __name__ == "__main__":
    main()
