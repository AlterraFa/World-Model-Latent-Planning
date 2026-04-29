import bisect
import os, sys
import pickle

script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
import pandas as pd
import numpy as np
import glob
import torch
import re
import random
import threading
import sqlite3
from torch.utils.data import Dataset
from .utils.metadata import (
    AgentBox,
    EgoPose,
    MetaKey,
    NuplanFrame,
    TrafficLight     
)
from .utils.decode import decode_batch
from rich import print


IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

class NuplanDataset(Dataset):
    """Action-conditioned video dataset backed by NuPlan SQLite databases.

    Each sample is one lidar_pc anchor frame. The loader reads all lidar_pc
    frames from the anchor scene, samples valid frame indices from that full
    sequence, and then queries image/metadata for each sampled index.
    Returned NuplanFrame fields are list-aligned by frame index
    (image_paths[i] with frame_tokens[i], frame_timestamps[i], ego_pose[i],
    agents[i], traffic_lights[i], and scenario_tags[i]).

    Args:
        database_paths: List of .db file paths to index.
        image_root:     Root directory prepended to image.filename_jpg paths.
        fpcs:           Max camera frames to return per sample.
        fps:            Reserved for compatibility with prior API.
        nclips:         Number of temporal clips to sample from the queried
                frame set from the scene lidar_pc sequence.
        frame_step:     Temporal stride between sampled lidar indices.
        allow_clip_overlap: Reserved for compatibility with prior API.
        random_jiggle_part: Randomly shift sampled start index.
        channel:        Camera channel to load (default 'CAM_F0').
        meta_keys:      Set of metadata fields to query and populate.  Pass
                        ``None`` (default) to fetch everything.  Supported
                        keys: ``'ego_pose'``, ``'agents'``,
                        ``'traffic_lights'``, ``'scenario_tags'``,
                        ``'log_meta'`` (map_version / vehicle_name /
                        scene_name / roadblock_ids).
        shared_transform:     Applied to the decoded image array (N, H, W, C).
        individual_transform: Applied per-frame before stacking.
    """

    # All recognised metadata keys
    ALL_META_KEYS = frozenset(MetaKey)

    def _normalize_meta_keys(self, meta_keys):
        if meta_keys is None:
            return self.ALL_META_KEYS
        if isinstance(meta_keys, (str, MetaKey)):
            return frozenset([str(meta_keys)])
        return frozenset(str(k) for k in meta_keys)

    def __init__(
        self,
        database_paths,
        image_root="./",
        fpcs=16,
        duration_s=10.0,
        random_jiggle_part=True,
        channel="CAM_F0",
        meta_keys=None,
        shared_transform=None,
        individual_transform=None,
    ):
        super().__init__()
        self._db_paths = database_paths
        self.image_root = image_root
        self.fpcs = fpcs
        self.duration_s = duration_s
        self.random_jiggle_part = random_jiggle_part
        self.channel = channel
        self.meta_keys = self._normalize_meta_keys(meta_keys)
        self.individual_transform = individual_transform
        self.shared_transform = shared_transform
        self._local = threading.local()
        self._log_time_cache = {}

        # Build log_name -> directory map so images spread across
        # nuplan-v1.1_train_camera_N/ splits can all be resolved.
        self._log_dirs: dict = {}  # log_name -> parent dir containing log_name/
        image_root_abs = os.path.abspath(image_root)
        if os.path.isdir(image_root_abs):
            for entry in os.scandir(image_root_abs):
                if entry.is_dir():
                    for sub in os.scandir(entry.path):
                        if sub.is_dir() and sub.name not in self._log_dirs:
                            self._log_dirs[sub.name] = entry.path

        # Build flat index: one entry per log (one sample = one log + random time window).
        _cache_path = ".cache/dataset_cache.pkl"
        _db_set = set(database_paths)
        _cached_db_set: set = set()
        if os.path.exists(_cache_path):
            with open(_cache_path, "rb") as _f:
                _cached_db_set, self.samples, self._log_time_cache = pickle.load(_f)
        if _cached_db_set != _db_set:
            # New or removed DBs detected — rebuild index from scratch.
            self.samples = []
            self._log_time_cache = {}
            for db_path in tqdm(database_paths, desc="Indexing databases"):
                con = sqlite3.connect(db_path)
                con.row_factory = sqlite3.Row
                rows = con.execute(
                    "SELECT hex(token) FROM log ORDER BY rowid ASC"
                ).fetchall()
                for r in rows:
                    log_token_hex = r["hex(token)"]
                    self.samples.append((db_path, log_token_hex))
                    self._log_time_cache[(db_path, log_token_hex)] = self._build_log_time_cache(
                        con, bytearray.fromhex(log_token_hex)
                    )
                con.close()
            os.makedirs(os.path.dirname(_cache_path), exist_ok=True)
            with open(_cache_path, "wb") as _f:
                pickle.dump((_db_set, self.samples, self._log_time_cache), _f)

    # ------------------------------------------------------------------
    # Image path resolution
    # ------------------------------------------------------------------

    def _resolve_image_path(self, filename_jpg: str) -> str:
        """Return absolute path for a filename_jpg from the DB.

        filename_jpg is relative (e.g. '<log_name>/CAM_F0/<hash>.jpg').
        Images may be split across multiple nuplan-v1.1_train_camera_N/
        subdirectories; use the pre-built log→dir map to find the right one.
        Falls back to os.path.join(image_root, filename_jpg) if unknown.
        """
        log_name = filename_jpg.split("/")[0]
        base = self._log_dirs.get(log_name, self.image_root)
        return os.path.join(base, filename_jpg)

    # ------------------------------------------------------------------
    # Connection management (one connection per worker thread, reused)
    # ------------------------------------------------------------------

    def _get_conn(self, db_path):
        if not hasattr(self._local, "conns"):
            self._local.conns = {}
        if db_path not in self._local.conns:
            con = sqlite3.connect(db_path, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.conns[db_path] = con
        return self._local.conns[db_path]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def _build_log_time_cache(self, con, log_token_bytes):
        """Precompute lidar/camera timestamp arrays for one log."""
        log_lp_rows = self._get_log_lidar_rows(con, log_token_bytes)
        if not log_lp_rows:
            return {
                "anchor_ts": 0,
                "anchor_token": "",
                "lp_timestamps": [],
                "lp_tokens_raw": [],
                "cam_rows": [],
                "cam_timestamps": [],
            }

        anchor_ts = log_lp_rows[0]["timestamp"]
        anchor_token = log_lp_rows[0]["token"].hex()
        lp_timestamps = [row["timestamp"] for row in log_lp_rows]
        lp_tokens_raw = [row["token"] for row in log_lp_rows]

        all_cam_rows = self._get_log_camera_rows(con, log_lp_rows)
        cam_rows = [(row["filename_jpg"], row["timestamp"]) for row in all_cam_rows]
        cam_timestamps = [row[1] for row in cam_rows]

        return {
            "anchor_ts": anchor_ts,
            "anchor_token": anchor_token,
            "lp_timestamps": lp_timestamps,
            "lp_tokens_raw": lp_tokens_raw,
            "cam_rows": cam_rows,
            "cam_timestamps": cam_timestamps,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        for retry in range(5):
            try:
                sample = self.load_sequences(index)
                if sample is not None:
                    return sample
            except Exception as e:
                if retry < 4:
                    print(f"Error loading sample at {index=}, retrying ({retry+1}/5): {e}")
                else:
                    print(f"Failed to load sample at {index=} after 5 retries: {e}")
        return None

    # ------------------------------------------------------------------
    # Core loader – returns (images, NuplanFrame)
    # ------------------------------------------------------------------

    def _get_anchor_info(self, con, token_bytes):
        return con.execute(
            "SELECT timestamp, scene_token FROM lidar_pc WHERE token = ?", (token_bytes,)
        ).fetchone()

    def _get_scene_rows(self, con, scene_token):
        return con.execute(
            """
            SELECT token, timestamp
            FROM lidar_pc
            WHERE scene_token = ?
            ORDER BY timestamp ASC
            """,
            (scene_token,),
        ).fetchall()

    def _get_scene_camera_rows(self, con, scene_start_ts, scene_end_ts):
        return con.execute(
            """
            SELECT img.filename_jpg, img.timestamp
            FROM image AS img
            INNER JOIN camera AS cam ON cam.token = img.camera_token
            WHERE cam.channel = ?
              AND img.timestamp BETWEEN ? AND ?
            ORDER BY img.timestamp ASC
            """,
            (self.channel, scene_start_ts, scene_end_ts),
        ).fetchall()

    def _sample_sequence_indices(self, seq_len):
        nclips = max(1, int(self.nclips))
        frame_step = max(1, int(self.frame_step))
        fpcs = max(1, int(self.fpcs)) if self.fpcs is not None else seq_len
        target_span = max(1, fpcs * frame_step)

        sampled_indices = []
        for clip_idx in range(nclips):
            if seq_len <= target_span:
                start_idx = 0
                end_idx = seq_len
            else:
                max_start = seq_len - target_span
                if self.random_jiggle_part:
                    start_idx = np.random.randint(0, max_start + 1)
                elif nclips > 1:
                    start_idx = int(round((clip_idx / (nclips - 1)) * max_start))
                else:
                    start_idx = 0
                end_idx = start_idx + target_span

            clip_indices = list(range(start_idx, end_idx, frame_step))
            if len(clip_indices) >= fpcs:
                clip_indices = clip_indices[:fpcs]
            elif len(clip_indices) > 0:
                clip_indices = clip_indices + [clip_indices[-1]] * (fpcs - len(clip_indices))
            sampled_indices.extend(clip_indices)

        return sampled_indices

    # ------------------------------------------------------------------
    # Log-scoped helpers
    # ------------------------------------------------------------------

    def _get_log_token(self, con, token_bytes):
        """Return log_token for the anchor lidar_pc."""
        row = con.execute(
            """
            SELECT ld.log_token
            FROM lidar_pc AS lp
            INNER JOIN lidar AS ld ON ld.token = lp.lidar_token
            WHERE lp.token = ?
            """,
            (token_bytes,),
        ).fetchone()
        return row["log_token"] if row else None

    def _get_log_lidar_rows(self, con, log_token):
        """All lidar_pc rows for the log, sorted by timestamp."""
        return con.execute(
            """
            SELECT lp.token, lp.timestamp
            FROM lidar_pc AS lp
            INNER JOIN lidar AS ld ON ld.token = lp.lidar_token
            WHERE ld.log_token = ?
            ORDER BY lp.timestamp ASC
            """,
            (log_token,),
        ).fetchall()

    def _get_log_camera_rows(self, con, log_lp_rows):
        """All camera frames for the log, using lidar_pc timestamp range."""
        if not log_lp_rows:
            return []
        t_min = log_lp_rows[0]["timestamp"]
        t_max = log_lp_rows[-1]["timestamp"]
        return con.execute(
            """
            SELECT img.filename_jpg, img.timestamp
            FROM image AS img
            INNER JOIN camera AS cam ON cam.token = img.camera_token
            WHERE cam.channel = ?
              AND img.timestamp BETWEEN ? AND ?
            ORDER BY img.timestamp ASC
            """,
            (self.channel, t_min, t_max),
        ).fetchall()

    def _apply_fps_filter(self, cam_rows):
        """Subsample camera rows to self.fps by enforcing a minimum time gap."""
        if not cam_rows or self.fps is None:
            return list(cam_rows)
        min_gap_us = int(1e6 / self.fps)
        filtered = [cam_rows[0]]
        for row in cam_rows[1:]:
            if row["timestamp"] - filtered[-1]["timestamp"] >= min_gap_us:
                filtered.append(row)
        return filtered

    def _bisect_nearest_lidar(self, lp_timestamps, lp_tokens, img_ts):
        """O(log n) nearest lidar_pc lookup using a pre-sorted timestamp list."""
        idx = bisect.bisect_left(lp_timestamps, img_ts)
        if idx == 0:
            return lp_tokens[0], lp_timestamps[0]
        if idx >= len(lp_timestamps):
            return lp_tokens[-1], lp_timestamps[-1]
        if lp_timestamps[idx] - img_ts <= img_ts - lp_timestamps[idx - 1]:
            return lp_tokens[idx], lp_timestamps[idx]
        return lp_tokens[idx - 1], lp_timestamps[idx - 1]

    # ------------------------------------------------------------------
    # Batch metadata queries  (one SQL call per meta key for all frames)
    # ------------------------------------------------------------------

    def _batch_query_ego_pose(self, con, lp_tokens_bytes):
        ph = ",".join(["?"] * len(lp_tokens_bytes))
        rows = con.execute(
            f"""
            SELECT lp.token AS lp_token, ep.*
            FROM ego_pose AS ep
            INNER JOIN lidar_pc AS lp ON lp.ego_pose_token = ep.token
            WHERE lp.token IN ({ph})
            """,
            lp_tokens_bytes,
        ).fetchall()
        result = {}
        for r in rows:
            result[bytes(r["lp_token"])] = EgoPose(
                timestamp=r["timestamp"],
                x=r["x"], y=r["y"], z=r["z"],
                qw=r["qw"], qx=r["qx"], qy=r["qy"], qz=r["qz"],
                vx=r["vx"], vy=r["vy"], vz=r["vz"],
                acceleration_x=r["acceleration_x"],
                acceleration_y=r["acceleration_y"],
                acceleration_z=r["acceleration_z"],
                angular_rate_x=r["angular_rate_x"],
                angular_rate_y=r["angular_rate_y"],
                angular_rate_z=r["angular_rate_z"],
            )
        return result

    def _batch_query_agents(self, con, lp_tokens_bytes):
        ph = ",".join(["?"] * len(lp_tokens_bytes))
        rows = con.execute(
            f"""
            SELECT lb.lidar_pc_token, lb.token, lb.track_token,
                   lb.x, lb.y, lb.z, lb.yaw,
                   lb.width, lb.length, lb.height,
                   lb.vx, lb.vy, lb.vz, lb.confidence,
                   c.name AS category
            FROM lidar_box AS lb
            INNER JOIN track    AS t ON t.token = lb.track_token
            INNER JOIN category AS c ON c.token = t.category_token
            WHERE lb.lidar_pc_token IN ({ph})
            """,
            lp_tokens_bytes,
        ).fetchall()
        result = {}
        for r in rows:
            key = bytes(r["lidar_pc_token"])
            result.setdefault(key, []).append(AgentBox(
                token=r["token"].hex(),
                track_token=r["track_token"].hex(),
                category=r["category"],
                x=r["x"], y=r["y"], z=r["z"], yaw=r["yaw"],
                width=r["width"], length=r["length"], height=r["height"],
                vx=r["vx"], vy=r["vy"], vz=r["vz"],
                confidence=r["confidence"],
            ))
        return result

    def _batch_query_traffic_lights(self, con, lp_tokens_bytes):
        ph = ",".join(["?"] * len(lp_tokens_bytes))
        rows = con.execute(
            f"""
            SELECT lidar_pc_token, lane_connector_id, status
            FROM traffic_light_status
            WHERE lidar_pc_token IN ({ph})
            """,
            lp_tokens_bytes,
        ).fetchall()
        result = {}
        for r in rows:
            key = bytes(r["lidar_pc_token"])
            result.setdefault(key, []).append(
                TrafficLight(lane_connector_id=r["lane_connector_id"], status=r["status"])
            )
        return result

    def _batch_query_scenario_tags(self, con, lp_tokens_bytes):
        ph = ",".join(["?"] * len(lp_tokens_bytes))
        rows = con.execute(
            f"""
            SELECT lidar_pc_token, type
            FROM scenario_tag
            WHERE lidar_pc_token IN ({ph})
            """,
            lp_tokens_bytes,
        ).fetchall()
        result = {}
        for r in rows:
            key = bytes(r["lidar_pc_token"])
            result.setdefault(key, []).append(r["type"])
        return result

    def _query_traffic_lights(self, con, lp_frame_token):
        tl_rows = con.execute(
            """
            SELECT lane_connector_id, status
            FROM traffic_light_status
            WHERE lidar_pc_token = ?
            """,
            (lp_frame_token,),
        ).fetchall()
        return [
            TrafficLight(lane_connector_id=r["lane_connector_id"], status=r["status"])
            for r in tl_rows
        ]

    def _query_scenario_tags(self, con, lp_frame_token):
        tag_rows = con.execute(
            """
            SELECT type FROM scenario_tag WHERE lidar_pc_token = ?
            """,
            (lp_frame_token,),
        ).fetchall()
        return [r["type"] for r in tag_rows]

    def _query_log_meta(self, con, token_bytes):
        meta_row = con.execute(
            """
            SELECT l.map_version, l.vehicle_name,
                   s.name AS scene_name, s.roadblock_ids
            FROM lidar_pc  AS lp
            INNER JOIN lidar AS ld ON ld.token = lp.lidar_token
            INNER JOIN log   AS l  ON l.token  = ld.log_token
            INNER JOIN scene AS s  ON s.token  = lp.scene_token
            WHERE lp.token = ?
            """,
            (token_bytes,),
        ).fetchone()

        if meta_row is None:
            return "", "", "", []

        map_version = meta_row["map_version"] or ""
        vehicle_name = meta_row["vehicle_name"] or ""
        scene_name = meta_row["scene_name"] or ""
        raw_rb = meta_row["roadblock_ids"]
        roadblock_ids = raw_rb.split(" ") if raw_rb else []
        return map_version, vehicle_name, scene_name, roadblock_ids

    def _query_log_meta_by_log_token(self, con, log_token_bytes):
        meta_row = con.execute(
            """
            SELECT map_version, vehicle_name FROM log WHERE token = ?
            """,
            (log_token_bytes,),
        ).fetchone()
        if meta_row is None:
            return "", ""
        return meta_row["map_version"] or "", meta_row["vehicle_name"] or ""

    def load_sequences(self, index):
        db_path, log_token_hex = self.samples[index]
        con = self._get_conn(db_path)
        log_token_bytes = bytearray.fromhex(log_token_hex)
        cache = self._log_time_cache.get((db_path, log_token_hex))
        if cache is None:
            cache = self._build_log_time_cache(con, log_token_bytes)
            self._log_time_cache[(db_path, log_token_hex)] = cache

        # ── 1. Get all lidar_pc rows for the log ───────────────────────
        lp_timestamps = cache["lp_timestamps"]
        lp_tokens_raw = cache["lp_tokens_raw"]
        if not lp_timestamps:
            return None

        anchor_ts = cache["anchor_ts"]
        anchor_token = cache["anchor_token"]

        # ── 2. Get all camera frames for the log ───────────────────────
        all_cam_rows = cache["cam_rows"]
        if not all_cam_rows:
            return {
                "images": None,
                "ego_pose": np.zeros((0, 7), dtype=np.float32),
                "agents": [],
                "traffic_lights": [],
                "scenario_tags": [],
                "meta": {
                    "token": anchor_token,
                    "timestamp": anchor_ts,
                    "image_paths": [],
                    "frame_tokens": [],
                    "frame_timestamps": [],
                    "map_version": "",
                    "vehicle_name": "",
                    "scene_name": "",
                    "roadblock_ids": [],
                },
            }

        # ── 3. Pick a time window of duration_s, then sample fpcs frames ─
        cam_timestamps = cache["cam_timestamps"]
        duration_us = int(self.duration_s * 1_000_000)
        t_first = cam_timestamps[0]
        t_last = cam_timestamps[-1]
        max_start = t_last - duration_us

        if max_start <= t_first:
            window_rows = all_cam_rows
        else:
            start_ts = random.randint(t_first, max_start) if self.random_jiggle_part else t_first
            end_ts = start_ts + duration_us
            i_start = bisect.bisect_left(cam_timestamps, start_ts)
            i_end = bisect.bisect_right(cam_timestamps, end_ts)
            window_rows = all_cam_rows[i_start:i_end]

        fpcs = max(1, self.fpcs)
        if len(window_rows) <= fpcs:
            sampled_cam_rows = window_rows
        else:
            indices = np.linspace(0, len(window_rows) - 1, fpcs, dtype=int)
            sampled_cam_rows = [window_rows[i] for i in indices]

        # ── 4. Bisect-lookup nearest lidar_pc for each sampled frame ───
        image_paths = []
        frame_tokens_hex = []
        frame_timestamps = []
        lp_token_bytes_per_frame = []

        for cam_row in sampled_cam_rows:
            filename_jpg, img_ts = cam_row
            image_paths.append(self._resolve_image_path(filename_jpg))
            lp_tok, lp_ts = self._bisect_nearest_lidar(lp_timestamps, lp_tokens_raw, img_ts)
            frame_tokens_hex.append(lp_tok.hex() if lp_tok is not None else "")
            frame_timestamps.append(lp_ts if lp_ts is not None else img_ts)
            lp_token_bytes_per_frame.append(lp_tok)

        # ── 5. Batch metadata queries (one SQL call per enabled key) ───
        valid_lp_tokens = [t for t in lp_token_bytes_per_frame if t is not None]

        ego_pose_map: dict = {}
        agents_map: dict = {}
        tl_map: dict = {}
        tags_map: dict = {}

        if valid_lp_tokens:
            if "ego_pose" in self.meta_keys:
                ego_pose_map = self._batch_query_ego_pose(con, valid_lp_tokens)
            if "agents" in self.meta_keys:
                agents_map = self._batch_query_agents(con, valid_lp_tokens)
            if "traffic_lights" in self.meta_keys:
                tl_map = self._batch_query_traffic_lights(con, valid_lp_tokens)
            if "scenario_tags" in self.meta_keys:
                tags_map = self._batch_query_scenario_tags(con, valid_lp_tokens)

        # ── 6. Assemble per-frame lists from batch results ─────────────
        ego_pose = []
        agents = []
        traffic_lights = []
        scenario_tags = []

        for lp_tok in lp_token_bytes_per_frame:
            key = bytes(lp_tok) if lp_tok is not None else None
            if "ego_pose" in self.meta_keys:
                ego_pose.append(ego_pose_map.get(key))
            if "agents" in self.meta_keys:
                agents.append(agents_map.get(key, []))
            if "traffic_lights" in self.meta_keys:
                traffic_lights.append(tl_map.get(key, []))
            if "scenario_tags" in self.meta_keys:
                scenario_tags.append(tags_map.get(key, []))

        # ── 7. Log-level metadata (single query) ───────────────────────
        map_version, vehicle_name = "", ""
        scene_name, roadblock_ids = "", []
        if "log_meta" in self.meta_keys:
            map_version, vehicle_name = self._query_log_meta_by_log_token(con, log_token_bytes)

        # ── 8. Build decomposed dictionary payload ─────────────────────
        ego_pose_np = np.array(
            [
                [p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz] if p is not None else [np.nan] * 7
                for p in ego_pose
            ],
            dtype=np.float32,
        ) if ego_pose else np.zeros((0, 7), dtype=np.float32)

        # ── 9. Decode and transform images ─────────────────────────────
        if not image_paths:
            return {
                "images": None,
                "ego_pose": ego_pose_np,
                "agents": agents,
                "traffic_lights": traffic_lights,
                "scenario_tags": scenario_tags,
                "meta": {
                    "token": anchor_token,
                    "timestamp": anchor_ts,
                    "image_paths": image_paths,
                    "frame_tokens": frame_tokens_hex,
                    "frame_timestamps": frame_timestamps,
                    "map_version": map_version,
                    "vehicle_name": vehicle_name,
                    "scene_name": scene_name,
                    "roadblock_ids": roadblock_ids,
                },
            }

        images = decode_batch(image_paths)

        if self.individual_transform is not None:
            images = np.array([self.individual_transform(img) for img in images])
        if self.shared_transform is not None:
            images = self.shared_transform(images)

        return {
            "images": images,
            "ego_pose": ego_pose_np,
            "agents": agents,
            "traffic_lights": traffic_lights,
            "scenario_tags": scenario_tags,
            "meta": {
                "token": anchor_token,
                "timestamp": anchor_ts,
                "image_paths": image_paths,
                "frame_tokens": frame_tokens_hex,
                "frame_timestamps": frame_timestamps,
                "map_version": map_version,
                "vehicle_name": vehicle_name,
                "scene_name": scene_name,
                "roadblock_ids": roadblock_ids,
            },
        }

def collate_nuplan(batch):
    # Filter out failed samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None

    # Support dict-based samples from load_sequences.
    if isinstance(batch[0], dict):
        images_list = [item.get("images") for item in batch]
        frames = []
        for item in batch:
            if "frame" in item and item["frame"] is not None:
                frames.append(item["frame"])
                continue
            frames.append(sample_dict_to_nuplan_frame(item))
    else:
        # Backward compatibility for tuple/list sample format.
        frames, images_list = zip(*batch)

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

    batched_frame = NuplanFrame(
        token=[f.token for f in frames],
        timestamp=[f.timestamp for f in frames],
        image_paths=[f.image_paths for f in frames],
        frame_tokens=[f.frame_tokens for f in frames],
        frame_timestamps=[f.frame_timestamps for f in frames],
        ego_pose=[f.ego_pose for f in frames],
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


def quaternion_yaw(qw, qx, qy, qz):
    """Compute yaw (heading) from quaternion components."""
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy ** 2 + qz ** 2),
    )


def transform_ego_trajectory_local(ego_pose_seq):
    """Project global ego poses into a local frame.

    Local frame convention:
      - origin at first valid pose
      - +y aligned with first valid pose heading

    Returns a list with the same length as ego_pose_seq where missing poses
    are kept as None and valid poses are (x_local, y_local) tuples.
    """
    if not ego_pose_seq:
        return []

    transformed = [None] * len(ego_pose_seq)
    first_idx = next((i for i, ep in enumerate(ego_pose_seq) if ep is not None), None)
    if first_idx is None:
        return transformed

    ep0 = ego_pose_seq[first_idx]
    ref_x, ref_y = ep0.x, ep0.y
    yaw0 = quaternion_yaw(ep0.qw, ep0.qx, ep0.qy, ep0.qz)
    cos_a, sin_a = np.cos(-yaw0 + np.pi / 2), np.sin(-yaw0 + np.pi / 2)

    for idx, ep in enumerate(ego_pose_seq):
        if ep is None:
            continue
        dx, dy = ep.x - ref_x, ep.y - ref_y
        transformed[idx] = (
            cos_a * dx - sin_a * dy,
            sin_a * dx + cos_a * dy,
        )
    return transformed
        
if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from torch.utils.data import DataLoader

    db_paths = glob.glob("./nuplan_dataset/data/**/*.db", recursive=True)

    dataset = NuplanDataset(
        db_paths,
        meta_keys=(MetaKey.EGO_POSE,),
        image_root="./nuplan_dataset",
        fpcs=32,
        duration_s=10.0,
        random_jiggle_part=True,
    )
    print(f"Total samples: {len(dataset)}")

    loader = DataLoader(dataset, 4, shuffle = False, collate_fn = collate_nuplan, num_workers = 4, persistent_workers = True, pin_memory = False)

    frames: NuplanFrame
    images: torch.Tensor
    for frames, images in loader:
        break

    state = {"sample_idx": 0, "frame_idx": 0, "frame": None, "images": None}

    def load_sample(idx):
        result = dataset[idx]
        if result is None:
            return None, None
        if isinstance(result, dict):
            return sample_dict_to_nuplan_frame(result), result.get("images")
        return result

    def render():
        fig.clf()
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
        ax_img = fig.add_subplot(gs[0])
        ax_traj = fig.add_subplot(gs[1])

        frame = state["frame"]
        images = state["images"]
        fidx = state["frame_idx"]
        sidx = state["sample_idx"]

        if images is not None and len(images) > 0:
            fidx = max(0, min(fidx, len(images) - 1))
            state["frame_idx"] = fidx
            ax_img.imshow(images[fidx])
            n_frames = len(images)
        else:
            ax_img.text(0.5, 0.5, "No frames decoded", ha="center", va="center",
                        transform=ax_img.transAxes, color="red", fontsize=14)
            n_frames = 0

        ax_img.set_title(
            f"sample {sidx + 1}/{len(dataset)}  frame {fidx + 1}/{n_frames}  "
            f"fpcs={dataset.fpcs}  duration={dataset.duration_s}s\n"
            "Left/Right: frame  Up/Down: sample  R: resample  Q: quit",
            fontsize=8,
        )
        ax_img.axis("off")

        if frame is not None and frame.ego_pose:
            pts_by_frame = transform_ego_trajectory_local(frame.ego_pose)
            pts = [p for p in pts_by_frame if p is not None]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                n = len(xs)

                colors = plt.cm.plasma(np.linspace(0, 1, n))
                for i in range(n - 1):
                    ax_traj.plot(xs[i:i+2], ys[i:i+2], color=colors[i], linewidth=2)
                ax_traj.scatter(xs, ys, c=np.linspace(0, 1, n), cmap="plasma", s=40, zorder=5)

                # First frame marker at origin (bottom center)
                ax_traj.scatter([0], [0], color="lime", s=180, zorder=10, marker="^", label="first")

                # Current frame marker (static highlight, plot doesn't move)
                if fidx < len(pts_by_frame) and pts_by_frame[fidx] is not None:
                    cx, cy = pts_by_frame[fidx]
                    ax_traj.scatter([cx], [cy], color="cyan",
                                    s=180, zorder=11, marker="*", label=f"frame {fidx+1}")

                ax_traj.legend(fontsize=8)

                # First frame at bottom center: keep x static and extend y upward.
                y_max = max(max(ys), 1)
                y_min = min(min(ys), 0)
                y_pad = max(0.5, (y_max - y_min) * 0.1)
                ax_traj.set_xlim(-10.0, 10.0)
                ax_traj.set_ylim(y_min - y_pad, y_max + y_pad)

        ax_traj.set_title("Ego trajectory (first frame = origin, heading = up)", fontsize=9)
        ax_traj.set_xlabel("Δx (m)")
        ax_traj.set_ylabel("Δy (m)")
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw_idle()

    state["frame"], state["images"] = load_sample(0)

    fig = plt.figure(figsize=(14, 6))

    def on_key(event):
        fidx = state["frame_idx"]
        sidx = state["sample_idx"]
        images = state["images"]
        n = len(images) if images is not None else 0

        if event.key in ("right", "d"):
            state["frame_idx"] = (fidx + 1) % max(n, 1)
        elif event.key in ("left", "a"):
            state["frame_idx"] = (fidx - 1) % max(n, 1)
        elif event.key == "up":
            state["sample_idx"] = (sidx - 1) % len(dataset)
            state["frame_idx"] = 0
            state["frame"], state["images"] = load_sample(state["sample_idx"])
        elif event.key == "down":
            state["sample_idx"] = (sidx + 1) % len(dataset)
            state["frame_idx"] = 0
            state["frame"], state["images"] = load_sample(state["sample_idx"])
        elif event.key == "r":
            state["frame_idx"] = 0
            state["frame"], state["images"] = load_sample(state["sample_idx"])
        elif event.key in ("q", "escape"):
            plt.close("all")
            return
        render()

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()