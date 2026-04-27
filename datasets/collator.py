import torch
import numpy as np
from .utils.metadata import NuplanFrame, EgoPose

class CollateNuplan:
    """Collate function for :class:`NuplanDataset` / :class:`VideoDataset` batches.

    Accepts a list of samples produced by ``__getitem__`` (dicts that may contain
    ``{"clips": [...]}`` from ``load_sequences`` or ``(NuplanFrame, ndarray)``
    tuples from the legacy path)
    and assembles them into a single ``(NuplanFrame, Tensor)`` pair ready for a
    training step.

    Samples that are ``None`` (failed loads) are silently dropped. Clip lists are
    flattened so the effective batch size is ``B_eff = sum_i num_clips_i``
    (typically ``B * nclips``). If every sample/clip is invalid the method
    returns ``(None, None)``.

    Returns
    -------
    batched_frame : NuplanFrame
        A single :class:`NuplanFrame` whose fields are either batched tensors or
        lists-of-lists indexed by ``[batch_index]`` or ``[batch_index, frame_index]``.

        Tensor fields
        ~~~~~~~~~~~~~
        ``timestamp`` – shape ``(B_eff,)``, dtype int64
            Anchor lidar_pc timestamp for each sample, in **microseconds since
            epoch**.  This is the timestamp of the first lidar_pc in the log
            and serves as the scene identity timestamp.

        ``frame_timestamps`` – shape ``(B_eff, N)``, dtype int64
            Timestamp of the nearest matched lidar_pc for every sampled camera
            frame, in **microseconds since epoch**.  Rows with fewer than ``N``
            sampled frames are zero-padded on the right.  Index as
            ``frame_timestamps[b, i]`` to get the timestamp of frame ``i`` in
            batch element ``b``.

        ``ego_pose`` – shape ``(B_eff, N, 7)``, dtype float32
            Ego-vehicle state at each sampled frame, sourced from the NuPlan
            ``ego_pose`` table via the nearest lidar_pc.  The 7 columns are::

                col 0  x    – easting   (m), UTM projection stored in the DB
                col 1  y    – northing  (m), UTM projection stored in the DB
                col 2  z    – altitude  (m), above WGS-84 ellipsoid
                col 3  qw   – quaternion w (real part)  } unit quaternion
                col 4  qx   – quaternion x              } encoding vehicle
                col 5  qy   – quaternion y              } heading in the
                col 6  qz   – quaternion z              } world/UTM frame

            Frames where no lidar_pc could be matched are filled with ``NaN``
            across all 7 columns.  Yaw can be recovered with::

                yaw = arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))

        List fields  (variable length / non-tensorisable)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``token`` – ``List[str]``, length B
            Hex-encoded lidar_pc token for the log anchor frame of each sample.

        ``image_paths`` – ``List[List[str]]``, shape (B, N_i)
            Absolute filesystem path to each decoded JPEG.
            ``image_paths[b][i]`` corresponds to ``stacked_images[b, i]``.

        ``frame_tokens`` – ``List[List[str]]``, shape (B, N_i)
            Hex-encoded lidar_pc token for the nearest lidar_pc matched to each
            camera frame.  Empty string when no match was found.

        ``agents`` – ``List[List[List[AgentBox]]]``, shape (B, N_i, *)
            Per-frame detected agents (vehicles, pedestrians, cyclists, …).
            ``agents[b][i]`` is a list of :class:`AgentBox` objects at frame
            ``i`` of batch element ``b``.  Each ``AgentBox`` carries position
            ``(x, y, z)``, ``yaw``, dimensions ``(width, length, height)``,
            velocity ``(vx, vy, vz)``, ``confidence``, ``category``, and
            ``track_token``, all in the world/UTM frame.
            Empty list when the ``agents`` meta key was not requested.

        ``traffic_lights`` – ``List[List[List[TrafficLight]]]``, shape (B, N_i, *)
            Per-frame traffic-light states.  ``traffic_lights[b][i]`` is a list
            of :class:`TrafficLight` objects, each with ``lane_connector_id``
            (integer map ID) and ``status`` (``'green'`` | ``'red'``).
            Empty list when the ``traffic_lights`` meta key was not requested.

        ``scenario_tags`` – ``List[List[List[str]]]``, shape (B, N_i, *)
            Per-frame NuPlan scenario-type tags (e.g. ``'starting_straight_traffic_light_intersection_traversal'``).
            ``scenario_tags[b][i]`` is a list of tag strings for frame ``i``.
            Empty list when the ``scenario_tags`` meta key was not requested.

        ``map_version`` – ``List[str]``, length B
            NuPlan map version string for the log (e.g. ``'nuplan-maps-v1.0'``).
            Empty string when the ``log_meta`` meta key was not requested.

        ``vehicle_name`` – ``List[str]``, length B
            Identifier of the data-collection vehicle (e.g. ``'tutorial_vehicle'``).
            Empty string when the ``log_meta`` meta key was not requested.

        ``scene_name`` – ``List[str]``, length B
            Human-readable scene name from the NuPlan ``scene`` table.
            Empty string when the ``log_meta`` meta key was not requested.

        ``roadblock_ids`` – ``List[List[str]]``, length B
            Ordered list of roadblock IDs the ego vehicle traverses in the scene,
            encoded as strings.  Empty list when the ``log_meta`` meta key was
            not requested.

    stacked_images : torch.Tensor or None
        Camera frames stacked into a uint8 tensor of shape ``(B, N, H, W, C)``
        where ``B_eff`` is the flattened batch size after clip expansion and
        ``C = 3`` (RGB channel order). Samples with no decoded images are
        replaced with a zero-filled frame of the same spatial resolution as the
        first valid sample.  ``None`` when every sample in the batch lacks images.

    Usage
    -----
    .. code-block:: python

        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=CollateNuplan(),
            num_workers=4,
            pin_memory=True,   # works because NuplanFrame implements pin_memory()
        )
        frames, images = next(iter(loader))
        # If dataset.nclips == 3 and DataLoader batch_size == 4:
        # images                   : (12, 16, H, W, 3)  uint8,  RGB
        # frames.timestamp         : (12,)              int64,  µs since epoch
        # frames.frame_timestamps  : (12, 16)           int64,  µs since epoch
        # frames.ego_pose          : (12, 16, 7)        float32 [x,y,z,qw,qx,qy,qz]

        # Recover yaw from quaternion:
        qw, qx, qy, qz = frames.ego_pose[...,3], frames.ego_pose[...,4], \\
                          frames.ego_pose[...,5], frames.ego_pose[...,6]
        yaw = torch.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))  # (4, 16)
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
            frames = []
            for item in flat_batch:
                if "frame" in item and item["frame"] is not None:
                    frames.append(item["frame"])
                    continue
                frames.append(sample_dict_to_nuplan_frame(item))
        else:
            # Backward compatibility for tuple/list sample format.
            frames, images_list = zip(*flat_batch)

        valid_pairs = [(f, i) for f, i in zip(frames, images_list) if f is not None]
        if not valid_pairs:
            return None, None
        frames, images_list = zip(*valid_pairs)

        # Stack images → (B, N, H, W, C) tensor; pad missing with zeros
        stacked_images = None
        valid_images = [img for img in images_list if img is not None]
        if valid_images:
            ref_shape = valid_images[0].shape  # (N, H, W, C)
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

        return batched_frame, stacked_images


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