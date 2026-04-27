import os
import sys
import argparse
from pathlib import Path

FILES = {
    "eval.py": 'import os, sys\nimport resource\nimport time\nimport gc\nfrom ruamel.yaml import YAML\nfrom functools import partial\nfrom pathlib import Path\n\nproject_root = Path(__file__).resolve().parents[4]\nsys.path.insert(0, str(project_root))\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\nFOLDER_DIR = os.path.dirname(os.path.dirname(__file__))\n\nimport random\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\nimport torch.distributed as dist\nfrom torch.nn import functional as F\nfrom torch.nn.parallel import DistributedDataParallel as DDP\nfrom utils.distributed import all_gather, all_reduce\n\nfrom .compile import (\n    compile_model,\n    compile_transform,\n    compile_dataloader,\n    compile_opt\n)\nfrom utils.training_logger import (\n    get_next_run,\n    create_self_supervised_logger,\n    NoOpLogger\n)\nfrom utils.distributed import init_distributed\nfrom utils.logger import Logger\nfrom utils.early_stop import EarlyStopping, MultiModuleEarlyStopping\n\nlogger = Logger(__name__)\n\ndef gpu_timer(funct, log_timming = True):\n    log_timming = log_timming and torch.cuda.is_available()\n    \n    elapsed_time = -1.0\n    if log_timming:\n        start = torch.cuda.Event(enable_timing = True)\n        end = torch.cuda.Event(enable_timing = True)\n        start.record()\n        \n    result = funct()\n    if log_timming:\n        end.record()\n        torch.cuda.synchronize()\n        elapsed_time = start.elapsed_time(end)\n    \n    return result, elapsed_time\n\ndef load_ckpt(\n    models_dict,\n    optimizer,\n    scaler,\n    checkpoint_dir,\n    prefer_best=True,\n    map_location=None,\n):\n    meta_path = os.path.join(checkpoint_dir, "checkpoint.pt")\n    if not os.path.exists(meta_path):\n        raise FileNotFoundError(f"Missing {meta_path}")\n\n    meta = torch.load(meta_path, map_location=map_location, weights_only=False)\n    score = meta.get("score")\n    best_loss = meta.get("best_loss", score)\n    start_epoch = meta.get("epoch", 0)\n\n    prefix = "best_" if prefer_best else "last_"\n    missing_models = []\n    for name, model in models_dict.items():\n        model_path = os.path.join(checkpoint_dir, f"{prefix}{name}.pt")\n        if not os.path.exists(model_path):\n            missing_models.append(model_path)\n            continue\n\n        model_state = torch.load(model_path, map_location=map_location)\n        model.load_state_dict(model_state)\n\n    if missing_models:\n        missing = "\\n".join(missing_models)\n        raise FileNotFoundError(\n            f"Missing expected resume weights in {checkpoint_dir}:\\n{missing}"\n        )\n\n    if optimizer is not None and meta.get("optimizer_state_dict") is not None:\n        optimizer.load_state_dict(meta["optimizer_state_dict"])\n    if scaler is not None and meta.get("scaler_state_dict") is not None:\n        scaler.load_state_dict(meta["scaler_state_dict"])\n\n    return models_dict, optimizer, scaler, start_epoch + 1, score, best_loss\n\ndef save_config_pretty(config_dict, save_path):\n    yaml = YAML()\n    # Basic formatting\n    yaml.indent(mapping=2, sequence=4, offset=2)\n    yaml.preserve_quotes = True\n    yaml.default_flow_style = False\n    \n    # This turns the dict into a \'ruamel\' internal dict that supports comments/spacing\n    from ruamel.yaml.comments import CommentedMap\n    \n    def dict_to_commented(d):\n        if isinstance(d, dict):\n            cm = CommentedMap()\n            for k, v in d.items():\n                cm[k] = dict_to_commented(v)\n            return cm\n        return d\n\n    pretty_data = dict_to_commented(config_dict)\n\n    # Add a blank line before every top-level key for readability\n    first = True\n    for key in pretty_data.keys():\n        if not first:\n            pretty_data.yaml_set_comment_before_after_key(key, before=\'\\n\')\n        first = False\n\n    with open(save_path, \'w\') as f:\n        yaml.dump(pretty_data, f)\n\n    \nGLOBAL_SEED = 12\nrandom.seed(GLOBAL_SEED)\nnp.random.seed(GLOBAL_SEED)\ntorch.manual_seed(GLOBAL_SEED)\ntorch.backends.cudnn.benchmark = True\ndef main(args: dict, yaml_path: str):\n    pass\n',
    "train.py": 'import os, sys\nimport resource\nimport time\nimport gc\nfrom ruamel.yaml import YAML\nfrom functools import partial\nfrom pathlib import Path\n\nproject_root = Path(__file__).resolve().parents[4]\nsys.path.insert(0, str(project_root))\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\nFOLDER_DIR = os.path.dirname(os.path.dirname(__file__))\n\nimport random\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\nimport torch.distributed as dist\nfrom torch.nn import functional as F\nfrom torch.nn.parallel import DistributedDataParallel as DDP\nfrom utils.distributed import all_gather, all_reduce\n\nfrom .compile import (\n    compile_model,\n    compile_transform,\n    compile_dataloader,\n    compile_opt\n)\nfrom utils.training_logger import (\n    get_next_run,\n    create_self_supervised_logger,\n    NoOpLogger\n)\nfrom utils.distributed import init_distributed\nfrom utils.logger import Logger\nfrom utils.early_stop import EarlyStopping, MultiModuleEarlyStopping\n\nlogger = Logger(__name__)\n\ndef gpu_timer(funct, log_timming = True):\n    log_timming = log_timming and torch.cuda.is_available()\n    \n    elapsed_time = -1.0\n    if log_timming:\n        start = torch.cuda.Event(enable_timing = True)\n        end = torch.cuda.Event(enable_timing = True)\n        start.record()\n        \n    result = funct()\n    if log_timming:\n        end.record()\n        torch.cuda.synchronize()\n        elapsed_time = start.elapsed_time(end)\n    \n    return result, elapsed_time\n\ndef load_ckpt(\n    models_dict,\n    optimizer,\n    scaler,\n    checkpoint_dir,\n    prefer_best=True,\n    map_location=None,\n):\n    meta_path = os.path.join(checkpoint_dir, "checkpoint.pt")\n    if not os.path.exists(meta_path):\n        raise FileNotFoundError(f"Missing {meta_path}")\n\n    meta = torch.load(meta_path, map_location=map_location, weights_only=False)\n    score = meta.get("score")\n    best_loss = meta.get("best_loss", score)\n    start_epoch = meta.get("epoch", 0)\n\n    prefix = "best_" if prefer_best else "last_"\n    missing_models = []\n    for name, model in models_dict.items():\n        model_path = os.path.join(checkpoint_dir, f"{prefix}{name}.pt")\n        if not os.path.exists(model_path):\n            missing_models.append(model_path)\n            continue\n\n        model_state = torch.load(model_path, map_location=map_location)\n        model.load_state_dict(model_state)\n\n    if missing_models:\n        missing = "\\n".join(missing_models)\n        raise FileNotFoundError(\n            f"Missing expected resume weights in {checkpoint_dir}:\\n{missing}"\n        )\n\n    if optimizer is not None and meta.get("optimizer_state_dict") is not None:\n        optimizer.load_state_dict(meta["optimizer_state_dict"])\n    if scaler is not None and meta.get("scaler_state_dict") is not None:\n        scaler.load_state_dict(meta["scaler_state_dict"])\n\n    return models_dict, optimizer, scaler, start_epoch + 1, score, best_loss\n\ndef save_config_pretty(config_dict, save_path):\n    yaml = YAML()\n    # Basic formatting\n    yaml.indent(mapping=2, sequence=4, offset=2)\n    yaml.preserve_quotes = True\n    yaml.default_flow_style = False\n    \n    # This turns the dict into a \'ruamel\' internal dict that supports comments/spacing\n    from ruamel.yaml.comments import CommentedMap\n    \n    def dict_to_commented(d):\n        if isinstance(d, dict):\n            cm = CommentedMap()\n            for k, v in d.items():\n                cm[k] = dict_to_commented(v)\n            return cm\n        return d\n\n    pretty_data = dict_to_commented(config_dict)\n\n    # Add a blank line before every top-level key for readability\n    first = True\n    for key in pretty_data.keys():\n        if not first:\n            pretty_data.yaml_set_comment_before_after_key(key, before=\'\\n\')\n        first = False\n\n    with open(save_path, \'w\') as f:\n        yaml.dump(pretty_data, f)\n\n    \nGLOBAL_SEED = 12\nrandom.seed(GLOBAL_SEED)\nnp.random.seed(GLOBAL_SEED)\ntorch.manual_seed(GLOBAL_SEED)\ntorch.backends.cudnn.benchmark = True\nACTION_LOSS = {}\ndef loss_registry(name):\n    def decorator(fn):\n        ACTION_LOSS[name] = fn\n        return fn\n    return decorator\n\ndef main(args: dict, yaml_path: str):\n    pass\n',
    "compile/dataloader.py": "import torch\nfrom torch.utils.data import DataLoader\nfrom utils.logger import Logger\nlogger = Logger(__name__)\n\ndef compile_dataloader(train_cfg, transform, collate_fn, num_workers, persistance_workers, pin_memory, world_sz, rank):\n    pass\n",
    "compile/loss.py": '''import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger
logger = Logger(__name__)

def compile_loss(cfg: dict = None):
    pass
''',
    "compile/models.py": '''import torch
import torch.nn as nn
import copy
from utils.logger import Logger
logger = Logger(__name__)

def _log_module_params(module_name: str, params: dict):
    logger.DEBUG(f"{module_name} config:")
    logger.DEBUG(params)

def compile_model(cfg: dict = None):
    pass
''',
    "compile/optim.py": '''import torch
from utils.schedulers import CosineWSDSchedule, CosineWDSchedule, CosineSchedule
from utils.logger import Logger
logger = Logger(__name__)

def compile_opt(apred, lpred, iterations_per_epoch, start_lr, ref_lr, warmup, anneal, num_epochs, wd=1e-06, final_wd=1e-06, final_lr=0.0, mixed_precision=False, betas=(0.9, 0.999), eps=1e-08, zero_init_bias_wd=True, enc_lr_scale=1.0):
    pass
''',
    "compile/resume.py": '''import os
import torch
import glob
from utils.logger import Logger
logger = Logger(__name__)

def load_checkpoint(model, optimizer, checkpoint_dir, checkpoint_name='probe.pt', prefer_best=True, map_location=None):
    pass

def restore_resume_state(resume_meta: dict, scaler, criterion, lr_scheduler, wd_scheduler, start_epoch: int, ipe: int, rank: int, run_idx: int, resume_prefer_best: bool):
    pass

def _load_state_dict_compat(model, state_dict: dict):
    pass

def _normalize_state_dict_for_model(model, state_dict: dict) -> dict:
    pass
''',
    "compile/transform.py": '''from typing import Optional
from utils.logger import Logger
logger = Logger(__name__)

def compile_transform(random_horizontal_flip=True, random_resize_aspect_ratio=(3 / 4, 4 / 3), random_resize_scale=(0.3, 1.0), reprob=0.0, auto_augment=False, motion_shift=False, crop_size=224, normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), pad_frame_count: Optional[int]=None, pad_frame_method: str='circulant'):
    pass
''',
    "compile/__init__.py": '''from .models import compile_model
from .transform import compile_transform
from .dataloader import compile_dataloader
from .optim import compile_opt
from .loss import compile_loss

__all__ = ['compile_model', 'compile_transform', 'compile_dataloader', 'compile_opt', 'compile_loss']
''',

}

def main():
    parser = argparse.ArgumentParser(description="Autogen app structure")
    parser.add_argument("--app_name", required=True, help="Name of the new app to generate")
    parser.add_argument("--prevent_override", default="True", choices=["True", "False"], help="Prevent overriding existing app directory (default: True)")
    args = parser.parse_args()

    app_name = args.app_name
    app_dir = Path("app")
    
    if not app_dir.exists():
        print(f"Error: {app_dir} does not exist.")
        sys.exit(1)

    dest_app_dir = app_dir / app_name
    
    if dest_app_dir.exists() and args.prevent_override == "True":
        print(f"Error: App directory {dest_app_dir} already exists. Pass --prevent_override False to allow overriding.")
        sys.exit(1)

    for cfile, file_content in FILES.items():
        dest_filepath = dest_app_dir / cfile
        dest_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_filepath, "w") as f:
            f.write(file_content)

    print(f"Successfully generated app structure for '{app_name}' in {dest_app_dir}")

    # Generate default config
    cfgs_dir = Path("cfgs") / app_name
    cfgs_dir.mkdir(parents=True, exist_ok=True)
    cfg_filepath = cfgs_dir / "default.yaml"
    
    config_content = f"""app: {app_name}

common: &common_settings
  patch_size: 16
  tubelet_size: 2
  crop_size: 256
  fpcs: 12

train:
  <<: *common_settings
  batch_size: 20
  num_workers: 4
  persistent_workers: true
  pin_mem: true
  datasets_path: []

data_aug: {{}}

loss: {{}}

optimization: {{}}

meta: {{}}

checkpoint: {{}}

logging: {{}}
"""
    with open(cfg_filepath, "w") as f:
        f.write(config_content)
        
    print(f"Successfully generated config for '{app_name}' in {cfg_filepath}")

if __name__ == "__main__":
    main()
