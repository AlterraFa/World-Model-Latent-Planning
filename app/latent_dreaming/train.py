import os, sys
import resource
import time
import gc
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed import all_gather, all_reduce

from .compile import (
    compile_model,
    compile_transform,
    compile_dataloader,
    compile_opt
)
from utils.training_logger import (
    get_next_run,
    create_self_supervised_logger,
    NoOpLogger
)
from datasets.utils.coordinate_transform import ego2local
from datasets.utils.metadata import NuplanFrame
from utils.distributed import init_distributed
from utils.logger import Logger
from utils.early_stop import EarlyStopping, MultiModuleEarlyStopping

logger = Logger(__name__)

def gpu_timer(funct, log_timming = True):
    log_timming = log_timming and torch.cuda.is_available()
    
    elapsed_time = -1.0
    if log_timming:
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record()
        
    result = funct()
    if log_timming:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    
    return result, elapsed_time

def load_ckpt(
    models_dict,
    optimizer,
    scaler,
    checkpoint_dir,
    prefer_best=True,
    map_location=None,
):
    meta_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location, weights_only=False)
    score = meta.get("score")
    best_loss = meta.get("best_loss", score)
    start_epoch = meta.get("epoch", 0)

    prefix = "best_" if prefer_best else "last_"
    missing_models = []
    for name, model in models_dict.items():
        model_path = os.path.join(checkpoint_dir, f"{prefix}{name}.pt")
        if not os.path.exists(model_path):
            missing_models.append(model_path)
            continue

        model_state = torch.load(model_path, map_location=map_location)
        model.load_state_dict(model_state)

    if missing_models:
        missing = "\n".join(missing_models)
        raise FileNotFoundError(
            f"Missing expected resume weights in {checkpoint_dir}:\n{missing}"
        )

    if optimizer is not None and meta.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])
    if scaler is not None and meta.get("scaler_state_dict") is not None:
        scaler.load_state_dict(meta["scaler_state_dict"])

    return models_dict, optimizer, scaler, start_epoch + 1, score, best_loss


def save_config_pretty(config_dict, save_path):
    # 1. Convert OmegaConf to a plain dict first
    # resolve=True handles any ${interpolation} keys
    if isinstance(config_dict, (DictConfig, ListConfig)):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    
    def dict_to_commented(d):
        if isinstance(d, dict):
            cm = CommentedMap()
            for k, v in d.items():
                cm[k] = dict_to_commented(v)
            return cm
        elif isinstance(d, list):
            return [dict_to_commented(i) for i in d]
        return d

    pretty_data = dict_to_commented(config_dict)

    # Now pretty_data is definitely a CommentedMap, so this will work
    first = True
    for key in pretty_data.keys():
        if not first:
            # This method belongs to ruamel.yaml, and now it exists
            pretty_data.yaml_set_comment_before_after_key(key, before='\n')
        first = False

    with open(save_path, 'w') as f:
        yaml.dump(pretty_data, f)

    
GLOBAL_SEED = 12
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
ACTION_LOSS = {}
def loss_registry(name):
    def decorator(fn):
        ACTION_LOSS[name] = fn
        return fn
    return decorator

def main(args: dict, yaml_path: str):
    OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))
    OmegaConf.resolve(args) 

    common_cfg   = args.get("common", {})
    patch_size   = common_cfg.get('patch_size', 16)
    tubelet_size = common_cfg.get('tubelet_size', 2)
    crop_size    = common_cfg.get('crop_size', 224)
    fpcs         = common_cfg.get('fpcs', 8)
    
    model_cfg = args.get('model', {})
    
    loader_cfg = args.get('loader', {})
    duration = loader_cfg.get("dataset_config", {}).get("duration", 5.0)
    
    augment_cfg = args.get('data_aug', {})
    auto_augment        = augment_cfg.get('auto_augment', False)
    horizontal_flip     = augment_cfg.get('horizontal_flip', False)
    motion_shift        = augment_cfg.get('motion_shift', False)
    random_aspect_ratio = augment_cfg.get('random_resize_aspect_ratio', (1.0, 1.0))
    random_resize_scale = augment_cfg.get('random_resize_scale', (1.0, 1.0))
    reprob              = augment_cfg.get('reprob', 0.0)
    crop_size           = augment_cfg.get('crop_size', 244)
    
    loss_cfg = args.get('loss', {})
    context_length = loss_cfg.get("context_length", 1) #-- Seconds
    
    optim_cfg  = args.get('optimization', {})
    num_epochs = optim_cfg.get('epochs', 100)
    ipe        = optim_cfg.get("ipe", 200)
    
    meta_cfg = args.get("meta", {})
    dtype              = meta_cfg.get('dtype', 'float32')
    save_freq          = meta_cfg.get('save_every_freq', 2)
    seed               = meta_cfg.get('seed', 0)
    sync_gc            = meta_cfg.get('sync_gc', False)
    save_root_dir      = meta_cfg.get('save_root_dir', "./")
    continue_from_path = meta_cfg.get('continue_from_path', None)
    continue_train          = bool(continue_from_path)
    resume_prefer_best      = bool(meta_cfg.get('resume_prefer_best', True))
    
    ckpt_cfg = args.get('checkpoint', {})
    patience = ckpt_cfg.get('patience', num_epochs)
    min_delta = ckpt_cfg.get('min_delta', 0.0)
    
    logging_cfg = args.get('logging', {})
    progress_type         = logging_cfg.get('progress_type', 'table')
    save_csv              = logging_cfg.get('save_csv', True)
    save_batch_csv        = logging_cfg.get('save_batch_csv', False)
    save_epoch_csv        = logging_cfg.get('save_epoch_csv', True)
    log_batch_tensorboard = logging_cfg.get('log_batch_tensorboard', False)
    log_model_graph       = logging_cfg.get('log_model_graph', False)

    torch.manual_seed(seed)
    world_size, rank = init_distributed()
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        logger.CUSTOM("SUCCESS", f"DDP enabled (world_size={world_size}, rank={rank})")
    else:
        logger.INFO("DDP disabled (single-GPU/single-process mode)")

    # =================== DTYPE SELECTION =================== #
    if dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    # =================== DTYPE SELECTION =================== #

    # =================== INIT WORLD MODEL =================== #
    device_type = f'cuda:{rank}'
    device = torch.device(device_type)
    world_model, ema_wm = compile_model(model_cfg, device = device)
    # =================== INIT WORLD MODEL =================== #
    
    # =================== INIT LOADER AND TRANSFORM =================== #
    transforms = compile_transform(
        random_horizontal_flip     = horizontal_flip,
        random_resize_aspect_ratio = random_aspect_ratio,
        random_resize_scale        = random_resize_scale,
        reprob       = reprob,
        auto_augment = auto_augment,
        motion_shift = motion_shift,
        crop_size    = crop_size,
    )

    dataloader, sampler = compile_dataloader(
        train_cfg = loader_cfg,
        transform = transforms,
        world_sz  = world_size,
        rank      = rank
    )
    # =================== INIT LOADER AND TRANSFORM =================== #
    
    
    # =================== INIT 5 OPTIMIZER =================== #
    optim, scaler, lr_scheduler, wd_scheduler, ema_scheduler = compile_opt(
        model = world_model,
        optim_config = optim_cfg,
        mixed_precision = mixed_precision
    )
    # =================== INIT 5 OPTIMIZER =================== #

    app_name = __name__.split(".")[1]
    beaut_name = f"{' '.join(app_name.split('_')).title()}"
    log_dir = os.path.join(save_root_dir, app_name)
    logger.INFO(f"[i][u]{beaut_name}[/][/] save root directory: {log_dir}")


    continue_run_dir = None
    continue_run_name = None
    if continue_train:
        continue_run_dir = os.path.abspath(os.path.expanduser(continue_from_path))
        if os.path.basename(continue_run_dir) == "weights":
            continue_run_dir = os.path.dirname(continue_run_dir)
        if not os.path.isdir(continue_run_dir):
            raise FileNotFoundError(f"continue_from_path does not exist: {continue_from_path}")
        continue_run_name = os.path.basename(continue_run_dir)
        if not continue_run_name.startswith("run"):
            raise ValueError(
                f"Expected continue_from_path to point to a run directory like '.../run1', got: {continue_run_dir}"
            )
            
    if rank == 0:
        if continue_train:
            resolved_run_idx = int(continue_run_name.removeprefix("run"))
            logger.INFO(f"Resuming requested. Selected run directory: {continue_run_dir}")
        else:
            resolved_run_idx = get_next_run(log_dir)
        run_idx_tensor = torch.tensor([resolved_run_idx], dtype=torch.long, device=device)
    else:
        run_idx_tensor = torch.tensor([0], dtype=torch.long, device=device)

    if dist.is_initialized() and world_size > 1:
        dist.broadcast(run_idx_tensor, src=0)
    run_idx = int(run_idx_tensor.item())

    start_epoch = 0
    resume_score = None
    resume_best_loss = None
    run_name = f"run{run_idx}"
    run_dir = os.path.join(log_dir, run_name)

    if continue_train and continue_run_dir is not None:
        run_dir = continue_run_dir
        run_name = os.path.basename(run_dir)
        models_to_resume = {
            "wm": world_model,
            "ema_wm": ema_wm,
        }
        (
            resumed_models,
            optim,
            scaler,
            start_epoch,
            resume_score,
            resume_best_loss,
        ) = load_ckpt(
            models_dict=models_to_resume,
            optimizer=optim,
            scaler=scaler,
            checkpoint_dir=os.path.join(run_dir, "weights"),
            prefer_best=resume_prefer_best,
            map_location=device,
        )
        world_model = resumed_models["wm"]
        ema_wm = resumed_models["ema_wm"]
        logger.INFO(
            f"Resumed {beaut_name} from {run_dir} at epoch {start_epoch} "
            f"using {'best' if resume_prefer_best else 'latest last'} checkpoints"
        )

    if rank == 0:
        log_stats = create_self_supervised_logger(
            log_dir = log_dir,
            epochs = num_epochs,
            run_name = run_name,
            progress_type = progress_type,
            save_csv = save_csv,
            save_batch_csv = save_batch_csv,
            save_epoch_csv = save_epoch_csv,
            use_wandb = True,
            wandb_config = args,
            wandb_project = app_name
        )
        saver = MultiModuleEarlyStopping(
            patience = patience,
            freq = save_freq,
            path_root = os.path.join(run_dir, "weights"),
            weights_only = True,
            min_delta = min_delta
        )
        if resume_best_loss is not None:
            saver.best_loss = resume_best_loss
        elif resume_score is not None:
            saver.best_loss = resume_score
        if not continue_train:
            yaml_name = f"{args['app']}-{world_model.__class__.__qualname__}-{crop_size}px.yaml"
            save_config_pretty(args, os.path.join(run_dir, yaml_name))
            
        if log_model_graph:
            B = 1
            inp = torch.randn((B, 3, 8, crop_size, crop_size), device = device)
            inp_ctx = inp[:, :, :-1]
            inp_target = inp[:, :, -1:]
            inp_goal = torch.randn((B, 2), device = device)

            z_target = world_model.encode_frames(inp_target)
            t = torch.rand((B,), device=inp_ctx.device)
            frame_rate = torch.full((B, ), 5)
            log_stats.log_model_graph(world_model, (inp_ctx, z_target, inp_goal, t, frame_rate))
    else:
        log_stats = NoOpLogger()

    if sync_gc:
        gc.disable()
        gc.collect()

    loader = iter(dataloader)
    
    def train_step(frames, images):
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()
        
        images = images.to(device, dtype = dtype)
        split_fpcs = math.ceil(context_length / duration * fpcs)
        context_image = images[:, :, :split_fpcs]
        target_image  = images[:, :, split_fpcs:]
        
        ego2local(frames.ego_pose)
        with torch.autocast(device.type, dtype = dtype, enabled = mixed_precision):
            latent_ctx = world_model.encode_frames(context_image)
        
    if start_epoch > 0:
        for _ in range(start_epoch * ipe):
            lr_scheduler.step()
            wd_scheduler.step()
        logger.INFO(f"Advanced LR/WD schedulers by {start_epoch * ipe} steps")

    with log_stats:
        log_stats.start_training("Diffusion World Model conditioned on Goal")
        sampler.set_epoch(start_epoch)
        
        for epoch in range(start_epoch, num_epochs):
            log_stats.start_epoch(epoch, ipe, desc = "Training")
            
            for itr in log_stats.batch_iterator([i for i in range(ipe)]):

                iter_retries = 0
                iter_success = False
                while not iter_success:
                    try:
                        sample = next(loader)
                        iter_success = True
                    except StopIteration:
                        loader = iter(dataloader)
                        sampler.set_epoch(epoch)
                    except Exception as e:
                        NUM_RETRIES = 5
                        if iter_retries < NUM_RETRIES:
                            logger.WARNING(f"Encountered an error while iterating loader: {e}")
                            iter_retries += 1
                            time.sleep(5)
                        else:
                            logger.ERROR("Exceeded maximum retries when iterating dataloade. Please check for error", exit_code = 5, full_traceback = e)
                            
                frames: NuplanFrame; images: torch.Tensor
                frames, images = sample

                gpu_timer(partial(train_step, frames, images))