import torch
import os
from glob import glob
from torch.utils.data import DataLoader, DistributedSampler
from utils.logger import Logger
from datasets.dataset import NuplanDataset
from datasets.collator import CollateNuplan
from utils.logger import log_parameters
logger = Logger(__name__)

def compile_dataloader(train_cfg, transform, world_sz, rank):
    dataset_config = train_cfg.get("dataset_config", {})
    databases_dir = glob(os.path.join(dataset_config.get('datasets_root', "./"), "**/*.db"), recursive = True)
    dataset = NuplanDataset(
        database_paths     = databases_dir,
        image_root         = dataset_config.get('datasets_root', "./"),
        fpcs               = dataset_config.get("fpcs", 8),
        duration_s         = dataset_config.get("duration", 8),
        n_clips            = dataset_config.get("n_clips", 1),
        allow_clip_overlap = dataset_config.get("allow_overlap", False),
        random_jiggle_part = dataset_config.get("random_jiggle", True),
        shared_transform   = transform,
    )
    
    dataloader = DataLoader(
        dataset = dataset,
        batch_size         = train_cfg.get("batch_size", 2),
        num_workers        = train_cfg.get("num_workers", 0),
        persistent_workers = train_cfg.get('persistent_workers', True),
        pin_memory         = train_cfg.get('pin_memory', False),
        collate_fn         = CollateNuplan(),
        shuffle            = True
    )

    sampler = DistributedSampler(
        dataset, num_replicas = world_sz, rank = rank, shuffle = True 
    )
    
    log_parameters(logger, "Dataloader", train_cfg)
    return dataloader, sampler