import torch
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, DistributedSampler
from utils.logger import Logger
from datasets.dataset import NuplanDataset
from datasets.collator import CollateNuplan
from utils.logger import log_parameters

IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

logger = Logger("compile_loaders")


_SKIP_DIRS = frozenset({"sensor_blobs", "sensor_blob", "blobs"})
_SKIP_PREFIXES = ("CAM_", "LIDAR_", "RADAR_", "MIC_")


def _scan_subdir(subdir: str, max_depth: int, root: str) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(subdir):
        depth = dirpath[len(root):].count(os.sep)
        found.extend(os.path.join(dirpath, f) for f in filenames if f.endswith(".db"))
        if depth >= max_depth:
            dirnames.clear()
        else:
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and not any(d.startswith(p) for p in _SKIP_PREFIXES)
            ]
    return found


def _find_db_files(root: str, max_depth: int = 3) -> list[str]:
    """Parallel walk up to max_depth levels to find .db files."""
    root = os.path.abspath(root)
    try:
        top_entries = [e.path for e in os.scandir(root) if e.is_dir()]
    except PermissionError:
        return []

    results = []
    lock = threading.Lock()

    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_scan_subdir, d, max_depth, root): d for d in top_entries}
        with tqdm(as_completed(futures), total=len(futures), desc="Scanning for .db files", unit="dir") as pbar:
            for future in pbar:
                found = future.result()
                with lock:
                    results.extend(found)
                pbar.set_postfix(dbs=len(results))

    # also check root-level .db files
    try:
        results.extend(
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".db")
        )
    except PermissionError:
        pass

    return results


def compile_dataloader(train_cfg, transform, world_sz, rank):
    logger.DEBUG("Scanning for database file")
    dataset_config = train_cfg.get("dataset_config", {})
    datasets_root = dataset_config.get('datasets_root', "./")
    max_depth = dataset_config.get('db_scan_depth', 2)
    databases_dir = _find_db_files(datasets_root, max_depth=max_depth)
    logger.DEBUG(f"Found {len(databases_dir)} database file")
    dataset = NuplanDataset(
        database_paths     = databases_dir,
        image_root         = dataset_config.get('datasets_root', "./"),
        fpcs               = dataset_config.get("fpcs", 8),
        duration_s         = dataset_config.get("duration", 8),
        n_clips            = dataset_config.get("n_clips", 1),
        allow_clip_overlap = dataset_config.get("allow_overlap", False),
        random_jiggle_part = dataset_config.get("random_jiggle", True),
        meta_keys          = "ego_pose",
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