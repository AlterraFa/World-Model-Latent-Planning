import torch
import torch.nn as nn
import copy
import json

from omegaconf import OmegaConf, DictConfig 
from utils.logger import Logger
from utils.autoload_modules import instantiate_from_config
logger = Logger(__name__)

def _log_module_params(module_name: str, params: dict):
    """
    Prints a dynamically sized, perfectly aligned box containing the model config.
    """
    if isinstance(params, DictConfig):
        printable_params = OmegaConf.to_container(params, resolve=True)
    else:
        printable_params = params

    json_str = json.dumps(printable_params, indent=4, default=str)
    lines = json_str.split('\n')

    header_text = f" CONFIGURATION FOR: {module_name} "
    content_width = max(max(len(line) for line in lines), len(header_text))
    box_width = content_width + 2  # Adding padding spaces

    logger.INFO(f"╔{'═' * box_width}╗")
    
    logger.INFO(f"║{header_text.center(box_width)}║")
    
    logger.INFO(f"╠{'═' * box_width}╣")
    
    for line in lines:
        logger.INFO(f"║ {line.ljust(content_width)} ║")
        
    logger.INFO(f"╚{'═' * box_width}╝")

def compile_model(cfg: dict = None, device = torch.device('cpu')) -> tuple[nn.Module, nn.Module]:
    model  = instantiate_from_config(cfg)
    model  = model.to(device)
    ema_wm = copy.deepcopy(model)

    _log_module_params(model.__class__.__name__, cfg)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    logger.DEBUG(f"Encoder number of parameters: {count_params(model)} with {count_params(model.encoder)} params from [u][b]{model.encoder.__class__.__name__}[/][/] and {count_params(model.diffuser)} params from [u][b]{model.diffuser.__class__.__name__}[/][/]")
    return model, ema_wm