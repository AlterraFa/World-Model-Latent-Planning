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
def main(args: dict, yaml_path: str):
    pass
