import torch
import torch.nn as nn
import copy
from utils.logger import Logger
logger = Logger(__name__)

def _log_module_params(module_name: str, params: dict):
    logger.DEBUG(f"{module_name} config:")
    logger.DEBUG(params)

def compile_model(cfg: dict = None):
    pass
