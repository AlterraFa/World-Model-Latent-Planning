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
    build_backends,
    NoOpLogger,
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
    gen_chunksz    = loss_cfg.get("gen_chunksz", 1)
    loss_exp       = loss_cfg.get("loss_exp", 2.0)
    randomize_goal = loss_cfg.get('randomize_goal', False)
    
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
    log_batch_scalars         = logging_cfg.get('log_batch_scalars', False)
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

    continue_run_dir = continue_run_name = None
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
    logger.INFO("On run", run_idx)

    start_epoch = 0
    resume_score = resume_best_loss = None
    run_name = f"run{run_idx}"
    run_dir = os.path.join(log_dir, run_name)

    if continue_train:
        run_dir = continue_run_dir
        run_name = os.path.basename(run_dir)
        resumed_models, optim, scaler, start_epoch, resume_score, resume_best_loss = load_ckpt(
            models_dict={"wm": world_model, "ema_wm": ema_wm},
            optimizer=optim,
            scaler=scaler,
            checkpoint_dir=os.path.join(run_dir, "weights"),
            prefer_best=resume_prefer_best,
            map_location=device,
        )
        world_model, ema_wm = resumed_models["wm"], resumed_models["ema_wm"]
        logger.INFO(
            f"Resumed {beaut_name} from {run_dir} at epoch {start_epoch} "
            f"using {'best' if resume_prefer_best else 'latest last'} checkpoints"
        )

    if rank == 0:
        backends_cfg = logging_cfg.get("backends", [])
        if not backends_cfg:
            log_stats = NoOpLogger()
        else:
            _backends = build_backends(
                backends_cfg,
                runtime_params={
                    "log_dir":  run_dir,
                    "project":  app_name,
                    "run_name": run_name,
                    "resume":   "allow" if continue_train else None,
                },
            )
            log_stats = create_self_supervised_logger(
                log_dir=run_dir,
                epochs=num_epochs,
                backends=_backends,
                progress_type=progress_type,
                save_csv=save_csv,
                save_batch_csv=save_batch_csv,
                save_epoch_csv=save_epoch_csv,
                log_batch_tensorboard=log_batch_scalars,
                resume_epoch=start_epoch,
            )
        saver = MultiModuleEarlyStopping(
            patience=patience,
            freq=save_freq,
            path_root=os.path.join(run_dir, "weights"),
            weights_only=True,
            min_delta=min_delta,
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
            inp = torch.randn((B, 3, 8, crop_size, crop_size), device=device)
            z_target = world_model.encode_frames(inp[:, :, -1:])
            t = torch.rand((B,), device=device)
            frame_rate = torch.full((B,), 5)
            log_stats.log_model_graph(world_model, (inp[:, :, :-1], z_target, torch.randn((B, 2), device=device), t, frame_rate))
    else:
        log_stats = NoOpLogger()

    if sync_gc:
        gc.disable()
        gc.collect()

    loader = iter(dataloader)
    
    def train_step(frames: NuplanFrame, images: torch.Tensor):
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()


        def add_noise(x, t, noise=None):
            noise = torch.randn_like(x) if noise is None else noise
            s = [x.shape[0]] + [1] * (x.dim() - 1)
            x_t = alpha(t).view(*s) * x + sigma(t).view(*s) * noise
            return x_t, noise

        sigma_min = 1e-6
        def alpha(t):
            return 1.0 - t

        def sigma(t):
            return sigma_min + t * (1.0 - sigma_min)

        def A_(t):
            return 1.0

        def B_(t):
            return -(1.0 - sigma_min)
        
        def preprocessing(images, frames):
            B = images.shape[0]
            images = images.to(device, dtype = dtype)
            ego_location = ego2local(frames.ego_pose).to(device, dtype = dtype)
            frame_rate = 1 / (frames.frame_timestamps.diff(n=1, dim=1).float().mean(-1) / 1e6).to(dtype)
            diffuse_time = torch.rand((B, ), device = device, dtype = dtype)
            
            split_fpcs    = math.ceil(context_length / duration * fpcs)
            context_image = images[:, :, :split_fpcs]
            target_image  = images[:, :, split_fpcs:]

            goal_idx = torch.randint(gen_chunksz + split_fpcs + 1, fpcs, (B,), dtype=torch.long, device=device)
            goal     = ego_location[torch.arange(B, device=device), goal_idx]  # (B, 2)

            choosen_target              = target_image[:, :, :gen_chunksz]
            latent_target         = world_model.encode_frames(choosen_target)
            noisy_target, noise = add_noise(latent_target, diffuse_time)

            return (context_image, latent_target, noisy_target, noise, goal, frame_rate, diffuse_time)
        
        @torch.no_grad()
        def ema_update(decay):
            params_k = list(ema_wm.parameters())
            params_q_cpu = [p.to("cpu", non_blocking=True) for p in world_model.parameters()]
            torch._foreach_mul_(params_k, decay)
            torch._foreach_add_(params_k, params_q_cpu, alpha=1 - decay)
            
        
        
        with torch.autocast(device.type, dtype = dtype, enabled = mixed_precision):
            (
                context_image,
                latent_target,
                noisy_target,
                noise,
                goal,
                frame_rate,
                diffuse_time,
            ) = preprocessing(images, frames)


            velocity = A_(diffuse_time) * latent_target + B_(diffuse_time) * noise
            pred_vel = world_model(
                x = context_image,
                noise = noisy_target,
                goal = goal,
                t = diffuse_time,
                frame_rate = frame_rate
            )
            
            loss: torch.Tensor = (((pred_vel - velocity) ** loss_exp) / loss_exp)
            loss_pstep = loss.mean((0, 2, 3, 4))
            loss = loss.mean()

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
        else:
            loss.backward()
            
        if mixed_precision:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()

        m = next(ema_scheduler)
        ema_update(m)
        
        return (
            loss, 
            loss_pstep,
            _new_lr,
            _new_wd
        )
        
        
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

                (loss, loss_pstep, curr_lr, curr_wd), elapsed_time = gpu_timer(partial(train_step, frames, images))

                loss_val = loss.item()
                if np.isnan(loss_val) or np.isinf(loss_val):
                    logger.ERROR(
                        f"Model failed to converge. {'nan' if np.isnan(loss_val) else 'inf'} detected",
                        exit_code=-213,
                    )

                batch_metrics = {
                    "LR":   curr_lr,
                    "WD":   curr_wd,
                    "Loss": loss_val,
                }
                for step_i, step_loss in enumerate(loss_pstep.tolist()):
                    batch_metrics[f"Loss|t{step_i}"] = step_loss

                log_stats.log_batch(batch_metrics)

            log_stats.log_epoch()

            if sync_gc:
                gc.collect()

            if rank == 0:
                models_to_save = {
                    "wm":     world_model,
                    "ema_wm": ema_wm,
                }
                saver(
                    score=log_stats.get_metric("Loss", "train"),
                    models_dict=models_to_save,
                    optimizer=optim,
                    scaler=scaler,
                    epoch=epoch,
                )

                if saver.early_stop:
                    logger.INFO("Early stopping triggered")

            should_stop = False
            if rank == 0:
                should_stop = bool(saver.early_stop)

            if dist.is_initialized() and world_size > 1:
                stop_tensor = torch.tensor([int(should_stop)], device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break