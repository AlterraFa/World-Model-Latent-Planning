import torch
import os
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, DistributedSampler
from utils.logger import Logger
from utils.autoload_modules import get_obj_from_str
from utils.logger import log_parameters

IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

logger = Logger("compile_loaders")


_SKIP_DIRS = frozenset({"sensor_blobs", "sensor_blob", "blobs"})
_SKIP_PREFIXES = ("CAM_", "LIDAR_", "RADAR_", "MIC_")


def _find_files(root: str, ext: str, max_depth: int = 3) -> list[str]:
    """Parallel walk up to max_depth levels to find files with a given extension."""
    root = os.path.abspath(root)
    try:
        top_entries = [e.path for e in os.scandir(root) if e.is_dir()]
    except PermissionError:
        return []

    results = []
    lock    = threading.Lock()

    def _scan(subdir):
        found = []
        for dirpath, dirnames, filenames in os.walk(subdir):
            depth = dirpath[len(root):].count(os.sep)
            found.extend(os.path.join(dirpath, f) for f in filenames if f.endswith(ext))
            if depth >= max_depth:
                dirnames.clear()
            else:
                dirnames[:] = [
                    d for d in dirnames
                    if d not in _SKIP_DIRS and not any(d.startswith(p) for p in _SKIP_PREFIXES)
                ]
        return found

    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_scan, d): d for d in top_entries}
        with tqdm(as_completed(futures), total=len(futures), desc=f"Scanning for {ext}", unit="dir") as pbar:
            for future in pbar:
                with lock:
                    results.extend(future.result())
                pbar.set_postfix(found=len(results))

    try:
        results.extend(os.path.join(root, f) for f in os.listdir(root) if f.endswith(ext))
    except PermissionError:
        pass

    return results


def compile_dataloader(loader_cfg, transform, world_sz, rank):
    dataset_cfg  = loader_cfg.get("dataset",  {})
    collator_cfg = loader_cfg.get("collator", {})

    if not dataset_cfg.get("target"):
        raise KeyError("loader.dataset.target is required")
    if not collator_cfg.get("target"):
        raise KeyError("loader.collator.target is required")

    # ── Collator ──────────────────────────────────────────────────────────
    collate_fn = get_obj_from_str(collator_cfg["target"])(**collator_cfg.get("params", {}))

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_cls = get_obj_from_str(dataset_cfg["target"])

    # Copy static params; pop scanning hints not passed to constructor
    params    = dict(dataset_cfg.get("params", {}))
    scan_root = params.pop("data_root",    "./")
    max_depth = params.pop("db_scan_depth", 3)
    params.pop("shared_transform",     None)  # runtime-only, never from yaml
    params.pop("individual_transform", None)

    # Detect the first required positional arg to decide which files to scan
    sig       = inspect.signature(dataset_cls.__init__)
    first_arg = next(
        (n for n, p in sig.parameters.items()
         if n != "self" and p.default is inspect.Parameter.empty),
        None,
    )
    if first_arg == "npz_paths":
        paths      = _find_files(scan_root, ".npz", max_depth=max_depth)
        path_kwarg = {"npz_paths": paths}
        logger.DEBUG(f"Found {len(paths)} .npz file(s) under {scan_root}")
    else:
        paths      = _find_files(scan_root, ".db", max_depth=max_depth)
        path_kwarg = {"database_paths": paths}
        logger.DEBUG(f"Found {len(paths)} .db file(s) under {scan_root}")

    # Filter to only kwargs the constructor actually accepts (handles
    # extra keys from yaml anchors like <<: *common_settings)
    valid    = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in params.items() if k in valid}

    dataset = dataset_cls(**path_kwarg, **filtered, shared_transform=transform)

    dataloader = DataLoader(
        dataset            = dataset,
        batch_size         = loader_cfg.get("batch_size",         2),
        num_workers        = loader_cfg.get("num_workers",         0),
        persistent_workers = loader_cfg.get("persistent_workers", True),
        pin_memory         = loader_cfg.get("pin_memory",         False),
        collate_fn         = collate_fn,
        shuffle            = True,
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_sz, rank=rank, shuffle=True
    )

    log_parameters(logger, "Dataloader", loader_cfg)
    return dataloader, sampler