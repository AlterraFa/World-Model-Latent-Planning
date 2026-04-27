import torch
from torch.utils.data import DataLoader
from utils.logger import Logger
logger = Logger(__name__)

def compile_dataloader(train_cfg, transform, collate_fn, num_workers, persistance_workers, pin_memory, world_sz, rank):
    pass
