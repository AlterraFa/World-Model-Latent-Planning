import torch
import numpy as np
from .utils.metadata import NuplanFrame, EgoPose

class CollateNuplan:
    """Collate NuPlan samples (with optional multi-clip expansion).

    Input can contain ``None``, dict samples, dicts with ``{"clips": [...]}``,
    or legacy ``(NuplanFrame, images)`` tuples. ``clips`` are flattened, so the
    effective batch size is ``B_eff`` (typically ``B * nclips``).

    Returns
    -------
    (batched_frame, stacked_images)
        ``batched_frame`` is a ``NuplanFrame`` with:
        - ``timestamp``: ``(B_eff,)`` int64, anchor lidar timestamp (us)
        - ``frame_timestamps``: ``(B_eff, N)`` int64, per-frame lidar timestamps
        - ``ego_pose``: ``(B_eff, N, 7)`` float32 as
          ``[x, y, z, qw, qx, qy, qz]`` (NaN for missing)
        - remaining fields kept as batched Python lists

        ``stacked_images`` is ``None`` or ``(B_eff, N, H, W, C)`` uint8 (RGB).
    """

    def __call__(self, batch):
        # Filter out failed samples and flatten clip-packed items.
        flat_batch = []
        for item in batch:
            if item is None:
                continue
            if isinstance(item, dict) and "clips" in item:
                clips = item.get("clips") or []
                flat_batch.extend([clip for clip in clips if clip is not None])
            else:
                flat_batch.append(item)

        if not flat_batch:
            return None, None

        # Support dict-based samples from load_sequences.
        if isinstance(flat_batch[0], dict):
            images_list = [item.get("images") for item in flat_batch]
            timing_list = [item["timing_ms"] for item in flat_batch if item.get("timing_ms") is not None]
            frames = []
            for item in flat_batch:
                if "frame" in item and item["frame"] is not None:
                    frames.append(item["frame"])
                    continue
                frames.append(sample_dict_to_nuplan_frame(item))
        else:
            # Backward compatibility for tuple/list sample format.
            timing_list = []
            frames, images_list = zip(*flat_batch)

        valid_pairs = [(f, i) for f, i in zip(frames, images_list) if f is not None]
        if not valid_pairs:
            return None, None
        frames, images_list = zip(*valid_pairs)

        # Stack images → (B, ...) tensor; pad missing with zeros.
        # VideoTransform returns torch Tensors; use torch.stack directly to
        # avoid the numpy round-trip of np.stack + torch.from_numpy.
        stacked_images = None
        valid_images = [img for img in images_list if img is not None]
        if valid_images:
            ref_shape = valid_images[0].shape
            if isinstance(valid_images[0], torch.Tensor):
                ref_dtype = valid_images[0].dtype
                arrays = [
                    img if img is not None else torch.zeros(ref_shape, dtype=ref_dtype)
                    for img in images_list
                ]
                stacked_images = torch.stack(arrays, dim=0)
            else:
                arrays = [
                    img if img is not None else np.zeros(ref_shape, dtype=np.uint8)
                    for img in images_list
                ]
                stacked_images = torch.from_numpy(np.stack(arrays, axis=0))

        # timestamp → (B,) int64
        timestamp_tensor = torch.tensor([f.timestamp for f in frames], dtype=torch.int64)

        # frame_timestamps → (B, N) int64
        ft_lists = [f.frame_timestamps for f in frames]
        max_n = max((len(t) for t in ft_lists), default=0)
        if max_n > 0:
            ft_arr = np.zeros((len(ft_lists), max_n), dtype=np.int64)
            for i, ts in enumerate(ft_lists):
                ft_arr[i, : len(ts)] = ts
            frame_timestamps_tensor = torch.from_numpy(ft_arr)
        else:
            frame_timestamps_tensor = torch.zeros((len(frames), 0), dtype=torch.int64)

        # ego_pose → (B, N, 7) float32  [x, y, z, qw, qx, qy, qz]; NaN for missing
        ego_lists = [f.ego_pose for f in frames]
        if ego_lists and max_n > 0:
            ego_arr = np.full((len(ego_lists), max_n, 7), np.nan, dtype=np.float32)
            for i, ep_list in enumerate(ego_lists):
                for j, ep in enumerate(ep_list):
                    if ep is not None and j < max_n:
                        ego_arr[i, j] = [ep.x, ep.y, ep.z, ep.qw, ep.qx, ep.qy, ep.qz]
            ego_tensor = torch.from_numpy(ego_arr)
        else:
            ego_tensor = torch.full((len(frames), max_n, 7), float("nan"))

        batched_frame = NuplanFrame(
            token=[f.token for f in frames],
            timestamp=timestamp_tensor,
            image_paths=[f.image_paths for f in frames],
            frame_tokens=[f.frame_tokens for f in frames],
            frame_timestamps=frame_timestamps_tensor,
            ego_pose=ego_tensor,
            agents=[f.agents for f in frames],
            traffic_lights=[f.traffic_lights for f in frames],
            scenario_tags=[f.scenario_tags for f in frames],
            map_version=[f.map_version for f in frames],
            vehicle_name=[f.vehicle_name for f in frames],
            scene_name=[f.scene_name for f in frames],
            roadblock_ids=[f.roadblock_ids for f in frames],
        )

        # Average per-phase timing across the batch (only present when NUPLAN_TIMING=1).
        avg_timing: dict | None = None
        if timing_list:
            keys = timing_list[0].keys()
            avg_timing = {k: round(sum(d[k] for d in timing_list) / len(timing_list), 2) for k in keys}

        return batched_frame, stacked_images, avg_timing



def ego_pose_array_to_list(ego_pose_arr, frame_ts):
    if ego_pose_arr is None:
        return []
    ego_pose_arr = np.asarray(ego_pose_arr)
    if ego_pose_arr.size == 0:
        return []

    ts_list = list(frame_ts) if frame_ts is not None else []
    poses = []
    for idx, row in enumerate(ego_pose_arr):
        if np.isnan(row).all():
            poses.append(None)
            continue
        ts = ts_list[idx] if idx < len(ts_list) else 0
        poses.append(EgoPose(
            timestamp=int(ts),
            x=float(row[0]), y=float(row[1]), z=float(row[2]),
            qw=float(row[3]), qx=float(row[4]), qy=float(row[5]), qz=float(row[6]),
            vx=0.0, vy=0.0, vz=0.0,
            acceleration_x=0.0, acceleration_y=0.0, acceleration_z=0.0,
            angular_rate_x=0.0, angular_rate_y=0.0, angular_rate_z=0.0,
        ))
    return poses


def sample_dict_to_nuplan_frame(sample):
    meta = sample.get("meta") or {}
    frame_ts = meta.get("frame_timestamps", [])
    return NuplanFrame(
        token=meta.get("token", ""),
        timestamp=meta.get("timestamp", 0),
        image_paths=meta.get("image_paths", []),
        frame_tokens=meta.get("frame_tokens", []),
        frame_timestamps=frame_ts,
        ego_pose=ego_pose_array_to_list(sample.get("ego_pose"), frame_ts),
        agents=sample.get("agents", []),
        traffic_lights=sample.get("traffic_lights", []),
        scenario_tags=sample.get("scenario_tags", []),
        map_version=meta.get("map_version", ""),
        vehicle_name=meta.get("vehicle_name", ""),
        scene_name=meta.get("scene_name", ""),
        roadblock_ids=meta.get("roadblock_ids", []),
    )