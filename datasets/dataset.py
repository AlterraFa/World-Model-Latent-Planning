import bisect
import os, sys

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
        channel="CAM_F0",
        fpcs=16,
        duration_s=10.0,
        nclips = 1, 
        random_jiggle_part=True,
        allow_clip_overlap=False,
        meta_keys=None,
        shared_transform=None,
        individual_transform=None,
    ):
        super().__init__()
        self._db_paths = database_paths
        self.image_root = image_root

        self.channel = channel
        self.fpcs = fpcs
        self.duration_s = duration_s
        self.nclips = nclips

        self.random_jiggle_part = random_jiggle_part
        self.allow_clip_overlap = allow_clip_overlap
        
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
        self.samples = []  # [(db_path, log_token_hex), ...]
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

    def _sample_clip_start_times(self, cam_timestamps):
        """Return start timestamps for each clip.

        If ``allow_clip_overlap`` is False, starts are spaced by one full
        clip duration whenever possible (non-overlap). If True, starts target
        a small overlap (~10%) between consecutive clips.
        """
        nclips = max(1, int(self.nclips))
        if nclips == 1:
            if not cam_timestamps:
                return []
            duration_us = int(self.duration_s * 1_000_000)
            t_first = cam_timestamps[0]
            t_last = cam_timestamps[-1]
            max_start = t_last - duration_us
            if max_start <= t_first:
                return [int(t_first)]
            start_ts = random.randint(t_first, max_start) if self.random_jiggle_part else t_first
            return [int(start_ts)]

        if not cam_timestamps:
            return []

        duration_us = int(self.duration_s * 1_000_000)
        t_first = cam_timestamps[0]
        t_last = cam_timestamps[-1]
        max_start = t_last - duration_us

        if max_start <= t_first:
            return [int(t_first)] * nclips

        # Allow a tiny overlap when requested (~10%).
        desired_stride = duration_us if not self.allow_clip_overlap else max(1, int(duration_us * 0.9))
        available_span = max_start - t_first
        required_span = (nclips - 1) * desired_stride

        if required_span <= available_span:
            max_base_start = max_start - required_span
            base_start = random.randint(t_first, max_base_start) if self.random_jiggle_part else t_first
            return [int(base_start + i * desired_stride) for i in range(nclips)]

        # Not enough time span for desired spacing: distribute starts evenly.
        starts = np.linspace(t_first, max_start, num=nclips, dtype=np.int64)
        return [int(s) for s in starts]

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
            empty_clip = {
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
            return {"clips": [empty_clip for _ in range(max(1, int(self.nclips)))]}

        # ── 3. Pick nclips time windows and sample fpcs frames per clip ─
        cam_timestamps = cache["cam_timestamps"]
        duration_us = int(self.duration_s * 1_000_000)
        clip_starts = self._sample_clip_start_times(cam_timestamps)

        # Log-level metadata (shared across all clips from this log).
        map_version, vehicle_name = "", ""
        scene_name, roadblock_ids = "", []
        if "log_meta" in self.meta_keys:
            map_version, vehicle_name = self._query_log_meta_by_log_token(con, log_token_bytes)

        fpcs = max(1, self.fpcs)
        clips = []
        for start_ts in clip_starts:
            end_ts = start_ts + duration_us
            i_start = bisect.bisect_left(cam_timestamps, start_ts)
            i_end = bisect.bisect_right(cam_timestamps, end_ts)
            window_rows = all_cam_rows[i_start:i_end]

            # Ensure every clip has at least one frame if the log has images.
            if not window_rows:
                nearest_idx = max(0, min(len(all_cam_rows) - 1, bisect.bisect_left(cam_timestamps, start_ts)))
                window_rows = [all_cam_rows[nearest_idx]]

            if len(window_rows) <= fpcs:
                sampled_cam_rows = window_rows
            else:
                indices = np.linspace(0, len(window_rows) - 1, fpcs, dtype=int)
                sampled_cam_rows = [window_rows[i] for i in indices]

            # Bisect-lookup nearest lidar_pc for each sampled frame.
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

            # Batch metadata queries (one SQL call per enabled key).
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

            ego_pose_np = np.array(
                [
                    [p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz] if p is not None else [np.nan] * 7
                    for p in ego_pose
                ],
                dtype=np.float32,
            ) if ego_pose else np.zeros((0, 7), dtype=np.float32)

            if not image_paths:
                images = None
            else:
                images = decode_batch(image_paths)
                if self.individual_transform is not None:
                    images = np.array([self.individual_transform(img) for img in images])
                if self.shared_transform is not None:
                    images = self.shared_transform(images)

            clips.append(
                {
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
                        "clip_index": len(clips),
                        "num_clips": len(clip_starts),
                    },
                }
            )

        return {"clips": clips}
        
if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from torch.utils.data import DataLoader
    from .collator import CollateNuplan, sample_dict_to_nuplan_frame
    from .utils.coordinate_transform import transform_ego_trajectory_local

    db_paths = glob.glob("./nuplan_dataset/data/**/*.db", recursive=True)


    dataset = NuplanDataset(
        db_paths,
        meta_keys=(MetaKey.EGO_POSE,),
        image_root="./nuplan_dataset",
        nclips=2,
        fpcs=16,
        duration_s=10.0,
        random_jiggle_part=True,
    )
    print(f"Total samples: {len(dataset)}")

    def collate_nuplan(batch):
        """Convenience wrapper around :class:`CollateNuplan` for use as a plain function."""
        return CollateNuplan()(batch)
    loader = DataLoader(dataset, 4, shuffle = False, collate_fn = collate_nuplan, num_workers = 4, persistent_workers = True)

    frames: NuplanFrame
    images: torch.Tensor
    for frames, images in loader:
        print(images.shape)
        print((frames.frame_timestamps[:, -1] - frames.frame_timestamps[:, 0]) * 1e-6)
    
    state = {
        "sample_idx": 0,
        "clip_idx": 0,
        "num_clips": 1,
        "frame_idx": 0,
        "frame": None,
        "images": None,
    }

    def load_sample(idx, clip_idx=0):
        result = dataset[idx]
        if result is None:
            return None, None, 0
        if isinstance(result, dict):
            if "clips" in result:
                clips = result.get("clips") or []
                if not clips:
                    return None, None, 0
                cidx = max(0, min(int(clip_idx), len(clips) - 1))
                clip = clips[cidx]
                return sample_dict_to_nuplan_frame(clip), clip.get("images"), len(clips)
            return sample_dict_to_nuplan_frame(result), result.get("images"), 1
        frame, images = result
        return frame, images, 1

    def render():
        fig.clf()
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
        ax_img = fig.add_subplot(gs[0])
        ax_traj = fig.add_subplot(gs[1])

        frame = state["frame"]
        images = state["images"]
        fidx = state["frame_idx"]
        sidx = state["sample_idx"]
        cidx = state["clip_idx"]
        nclips = state["num_clips"]

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
            f"sample {sidx + 1}/{len(dataset)}  clip {cidx + 1}/{max(1, nclips)}  frame {fidx + 1}/{n_frames}  "
            f"fpcs={dataset.fpcs}  duration={dataset.duration_s}s\n"
            "Left/Right: frame  [/]: clip  Up/Down: sample  R: resample  Q: quit",
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

    state["frame"], state["images"], state["num_clips"] = load_sample(0, 0)

    fig = plt.figure(figsize=(14, 6))

    def on_key(event):
        fidx = state["frame_idx"]
        sidx = state["sample_idx"]
        cidx = state["clip_idx"]
        nclips = state["num_clips"]
        images = state["images"]
        n = len(images) if images is not None else 0

        if event.key in ("right", "d"):
            state["frame_idx"] = (fidx + 1) % max(n, 1)
        elif event.key in ("left", "a"):
            state["frame_idx"] = (fidx - 1) % max(n, 1)
        elif event.key == "up":
            state["sample_idx"] = (sidx - 1) % len(dataset)
            state["clip_idx"] = 0
            state["frame_idx"] = 0
            state["frame"], state["images"], state["num_clips"] = load_sample(state["sample_idx"], state["clip_idx"])
        elif event.key == "down":
            state["sample_idx"] = (sidx + 1) % len(dataset)
            state["clip_idx"] = 0
            state["frame_idx"] = 0
            state["frame"], state["images"], state["num_clips"] = load_sample(state["sample_idx"], state["clip_idx"])
        elif event.key in ("]", "n"):
            state["clip_idx"] = (cidx + 1) % max(1, nclips)
            state["frame_idx"] = 0
            state["frame"], state["images"], state["num_clips"] = load_sample(state["sample_idx"], state["clip_idx"])
        elif event.key in ("[", "p"):
            state["clip_idx"] = (cidx - 1) % max(1, nclips)
            state["frame_idx"] = 0
            state["frame"], state["images"], state["num_clips"] = load_sample(state["sample_idx"], state["clip_idx"])
        elif event.key == "r":
            state["frame_idx"] = 0
            state["frame"], state["images"], state["num_clips"] = load_sample(state["sample_idx"], state["clip_idx"])
        elif event.key in ("q", "escape"):
            plt.close("all")
            return
        render()

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()
    