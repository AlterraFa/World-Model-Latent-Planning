import torch
import torch.nn as nn
import copy
import json

from omegaconf import OmegaConf, DictConfig 
from utils.logger import Logger, log_parameters
from utils.autoload_modules import instantiate_from_config
logger = Logger(__name__)


def compile_model(cfg: dict = None, device = torch.device('cpu')) -> tuple[nn.Module, nn.Module]:
    model  = instantiate_from_config(cfg)
    model  = model.to(device)
    ema_wm = copy.deepcopy(model)

    log_parameters(logger, model.__class__.__name__, cfg)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    logger.DEBUG(f"Encoder number of parameters: {count_params(model)} with {count_params(model.encoder)} params from [u][b]{model.encoder.__class__.__name__}[/][/] and {count_params(model.diffuser)} params from [u][b]{model.diffuser.__class__.__name__}[/][/]")
    return model, ema_wm