import os, sys
import resource
import time
import gc
from ruamel.yaml import YAML
from functools import partial
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))

import random
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
    yaml = YAML()
    # Basic formatting
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    
    # This turns the dict into a 'ruamel' internal dict that supports comments/spacing
    from ruamel.yaml.comments import CommentedMap
    
    def dict_to_commented(d):
        if isinstance(d, dict):
            cm = CommentedMap()
            for k, v in d.items():
                cm[k] = dict_to_commented(v)
            return cm
        return d

    pretty_data = dict_to_commented(config_dict)

    # Add a blank line before every top-level key for readability
    first = True
    for key in pretty_data.keys():
        if not first:
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
    
    model_cfg = args.get('model', {})
    
    training_cfg = args.get('train', {})
    
    augment_cfg = args.get('data_aug', {})
    
    loss_cfg = args.get('loss', {})
    
    optim_cfg = args.get('optimization', {})
    anneal       = optim_cfg.get('anneal', 15)
    num_epochs   = optim_cfg.get('epochs', 100)
    final_lr     = optim_cfg.get('final_lr', 0.0)
    final_wd     = optim_cfg.get("final_weight_decay", 0.0)
    ipe          = optim_cfg.get('ipe', 100)
    lr           = optim_cfg.get('lr', 1e-3)
    start_lr     = optim_cfg.get('start_lr', 1e-3)
    warmup       = optim_cfg.get('warmup', 10)
    weight_decay = optim_cfg.get('weight_decay', 0.0)
    betas        = optim_cfg.get('betas', (0.9, 0.999))
    eps          = optim_cfg.get('eps', 1.0e-8)
    ema          = optim_cfg.get('ema', [0.9, 1.0])
    
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
    compile_model(model_cfg, device = device)
    # =================== INIT WORLD MODEL =================== #