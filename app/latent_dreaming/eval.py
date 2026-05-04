import os, sys
import resource
import math
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from .compile import compile_model, compile_transform, compile_dataloader
from .compile._visualize import visualize_batch
from utils.distributed import init_distributed
from utils.logger import Logger
from datasets.utils.metadata import NuplanFrame

logger = Logger(__name__)


def _load_epoch_ckpt(models_dict: dict, checkpoint_dir: str, epoch: int,
                     map_location=None) -> None:
    for name, model in models_dict.items():
        path = os.path.join(checkpoint_dir, f"epoch_{epoch}_{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No epoch-{epoch} checkpoint for '{name}' at {path}"
            )
        model.load_state_dict(torch.load(path, map_location=map_location))


def main(args: dict, yaml_path: str):
    OmegaConf.register_new_resolver("div", lambda x, y: int(x / y), replace=True)
    OmegaConf.resolve(args)

    common_cfg = args.get("common", {})
    fpcs       = common_cfg.get("fpcs", 8)

    model_cfg  = args.get("model", {})
    loader_cfg = args.get("loader", {})
    duration   = loader_cfg.get("dataset", {}).get("params", {}).get("duration_s", 5.0)

    augment_cfg    = args.get("data_aug", {})
    crop_size      = augment_cfg.get("crop_size", 224)

    loss_cfg       = args.get("loss", {})
    context_length = loss_cfg.get("context_length", 1)
    gen_chunksz    = loss_cfg.get("gen_chunksz", 1)
    loss_exp       = loss_cfg.get("loss_exp", 2.0)
    randomize_goal = loss_cfg.get("randomize_goal", False)

    meta_cfg           = args.get("meta", {})
    dtype_str          = meta_cfg.get("dtype", "float32")
    continue_from_path = meta_cfg.get("continue_from_path", None)

    eval_cfg        = args.get("eval", {})
    visualize_epoch = eval_cfg.get("epoch", 0)
    nfe             = eval_cfg.get("nfe", 20)
    eta             = eval_cfg.get("eta", 0.0)
    umap_neighbors  = eval_cfg.get("umap_neighbors", 30)
    up_power        = eval_cfg.get("up_power", 16)

    if dtype_str.lower() == "bfloat16":
        dtype, mixed_precision = torch.bfloat16, True
    elif dtype_str.lower() == "float16":
        dtype, mixed_precision = torch.float16, True
    else:
        dtype, mixed_precision = torch.float32, False

    world_size, rank = init_distributed()
    device = torch.device(f"cuda:{rank}")

    world_model, _ = compile_model(model_cfg, device=device)
    world_model.eval()

    if continue_from_path is None:
        raise ValueError(
            "Eval mode requires meta.continue_from_path pointing to a run directory."
        )
    run_dir  = os.path.abspath(os.path.expanduser(continue_from_path))
    if os.path.basename(run_dir) == "weights":
        run_dir = os.path.dirname(run_dir)
    ckpt_dir = os.path.join(run_dir, "weights")

    _load_epoch_ckpt({"wm": world_model}, ckpt_dir,
                     epoch=visualize_epoch, map_location=device)
    logger.INFO(f"Loaded epoch-{visualize_epoch} checkpoint from {ckpt_dir}")

    transforms = compile_transform(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    dataloader, _ = compile_dataloader(
        loader_cfg=loader_cfg,
        transform=transforms,
        world_sz=world_size,
        rank=rank,
    )

    frames: NuplanFrame
    images: torch.Tensor
    frames, images, _ = next(iter(dataloader))
    if frames is None or images is None:
        raise RuntimeError("First batch returned None – check the dataloader.")

    visualize_batch(
        world_model=world_model,
        images=images,
        frames=frames,
        device=device,
        dtype=dtype,
        mixed_precision=mixed_precision,
        context_length=context_length,
        duration=duration,
        fpcs=fpcs,
        gen_chunksz=gen_chunksz,
        loss_exp=loss_exp,
        randomize_goal=randomize_goal,
        nfe=nfe,
        eta=eta,
        n_neighbors=umap_neighbors,
        up_power=up_power,
        epoch=visualize_epoch,
        log_stats=None,   # interactive mode
    )

    plt.show(block=True)

