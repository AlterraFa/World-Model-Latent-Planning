import bisect
import os, sys
import pickle
import time

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
from .utils.decode import decode_batch, start_decode_batch, collect_decode_batch
from rich import print


IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

# Set NUPLAN_TIMING=1 to print per-phase timing for every sample.
_TIMING = os.environ.get('NUPLAN_TIMING', '0') == '1'

class NuplanSQLiteDataset(Dataset):
    """Action-conditioned video dataset backed by NuPlan SQLite databases.

    Each sample is one log entry. The loader picks ``n_clips`` time windows
    of ``duration_s`` seconds from that log and returns them as a
    ``{"clips": [...]}`` dict.  ``CollateNuplan`` flattens the clips so the
    effective batch size is ``B * n_clips``.

    Args:
        database_paths:      List of .db file paths to index.
        image_root:          Root directory prepended to image.filename_jpg.
        fpcs:                Camera frames to return per clip.
        duration_s:          Length of each time window in seconds.
        random_jiggle_part:  Randomly shift sampled window starts.
        channel:             Camera channel (default ``'CAM_F0'``).
        meta_keys:           Metadata fields to populate (``None`` = all).
        shared_transform:    Applied to the decoded image array (N, H, W, C).
        individual_transform:Applied per-frame before stacking.
        n_clips:             Number of temporal clips to draw from each log.
                             Effective batch size becomes ``B * n_clips``.
        allow_clip_overlap:  If ``False``, clip windows are spaced so that
                             adjacent windows share at most ~10 % of their
                             length (``0.1 * duration_s``).  Ignored when
                             ``n_clips == 1``.
    """

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
        n_clips=1,
        allow_clip_overlap=True,
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
        self.n_clips = max(1, int(n_clips))
        self.allow_clip_overlap = allow_clip_overlap
        self._local_storage = None 
        self._log_time_cache = {}

        self._log_dirs: dict = {}
        image_root_abs = os.path.abspath(image_root)
        if os.path.isdir(image_root_abs):
            print(f"[Dataset] Scanning log dirs under {image_root_abs} ...")
            for entry in os.scandir(image_root_abs):
                if entry.is_dir():
                    for sub in os.scandir(entry.path):
                        if sub.is_dir() and sub.name not in self._log_dirs:
                            self._log_dirs[sub.name] = entry.path
            print(f"[Dataset] Found {len(self._log_dirs)} log dirs.")

        _cache_path = ".cache/dataset_cache_v2.pkl"
        _db_set = set(database_paths)
        _cached_db_set: set = set()
        if os.path.exists(_cache_path):
            _cache_size_mb = os.path.getsize(_cache_path) / 1024 / 1024
            print(f"[Dataset] Loading cache ({_cache_size_mb:.1f} MB) ...")
            with open(_cache_path, "rb") as _f:
                _cached_db_set, self.samples, self._log_time_cache = pickle.load(_f)
            print(f"[Dataset] Cache loaded ({len(self.samples)} samples).")
        if _cached_db_set != _db_set:
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
            try:
                os.makedirs(os.path.dirname(_cache_path), exist_ok=True)
                with open(_cache_path, "wb") as _f:
                    pickle.dump((_db_set, self.samples, self._log_time_cache), _f)
            except Exception as e:
                print(f"Error caching: {e}")
    @property
    def _local(self):
        if self._local_storage is None:
            self._local_storage = threading.local()
        return self._local_storage

    # ------------------------------------------------------------------
    # Image path resolution
    # ------------------------------------------------------------------

    def _resolve_image_path(self, filename_jpg: str) -> str:
        log_name = filename_jpg.split("/")[0]
        base = self._log_dirs.get(log_name, self.image_root)
        return os.path.join(base, filename_jpg)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self, db_path):
        # Accessing self._local here triggers the @property above
        if not hasattr(self._local, "conns"):
            self._local.conns = {}
        if db_path not in self._local.conns:
            # check_same_thread=False is important for SQLite in multi-threaded envs
            con = sqlite3.connect(db_path, check_same_thread=False)
            con.row_factory = sqlite3.Row
            # Limit per-connection page cache to prevent RAM growth in workers.
            # On Kaggle the DB lives on a slow NFS mount — a larger cache cuts
            # repeated page re-reads significantly.
            _sql_cache_kb = 16 * 1024 if IS_KAGGLE_COMMIT else 512
            con.execute(f"PRAGMA cache_size = -{_sql_cache_kb}")
            con.execute("PRAGMA temp_store = MEMORY")
            # Memory-map the DB file so the OS page cache handles NFS reads.
            # 256 MB on Kaggle; disabled locally (0 = OS default).
            _mmap = 256 * 1024 * 1024 if IS_KAGGLE_COMMIT else 0
            con.execute(f"PRAGMA mmap_size = {_mmap}")
            self._local.conns[db_path] = con
        return self._local.conns[db_path]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def _build_log_time_cache(self, con, log_token_bytes):
        all_cam_rows = self._get_log_camera_rows(con, log_token_bytes)
        cam_rows = [(row["filename_jpg"], row["timestamp"]) for row in all_cam_rows]
        cam_timestamps = [row[1] for row in cam_rows]

        if not cam_rows:
            return {
                "anchor_ts": 0,
                "anchor_token": "",
                "cam_rows": [],
                "cam_timestamps": [],
            }

        return {
            "anchor_ts": cam_rows[0][1],
            "anchor_token": log_token_bytes.hex(),
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
    # Log-scoped helpers
    # ------------------------------------------------------------------

    def _get_log_camera_rows(self, con, log_token_bytes):
        return con.execute(
            """
            SELECT img.filename_jpg, img.timestamp
            FROM image AS img
            INNER JOIN camera AS cam ON cam.token = img.camera_token
            WHERE cam.log_token = ?
              AND cam.channel = ?
            ORDER BY img.timestamp ASC
            """,
            (log_token_bytes, self.channel),
        ).fetchall()

    def _get_lp_tokens_in_range(self, con, log_token_bytes, t_min, t_max):
        """One query: all lidar_pc (token, timestamp) covering [t_min-200ms, t_max+200ms]."""
        rows = con.execute(
            """
            SELECT lp.token, lp.timestamp
            FROM lidar_pc AS lp
            INNER JOIN lidar AS ld ON ld.token = lp.lidar_token
            WHERE ld.log_token = ?
              AND lp.timestamp BETWEEN ? AND ?
            ORDER BY lp.timestamp ASC
            """,
            (log_token_bytes, t_min - 200_000, t_max + 200_000),
        ).fetchall()
        return rows

    def _bisect_nearest_token(self, ts_list, tok_list, target_ts):
        """Binary search for the nearest lidar_pc token in sorted ts_list."""
        if not ts_list:
            return None
        idx = bisect.bisect_left(ts_list, target_ts)
        if idx == 0:
            return tok_list[0]
        if idx >= len(ts_list):
            return tok_list[-1]
        if ts_list[idx] - target_ts <= target_ts - ts_list[idx - 1]:
            return tok_list[idx]
        return tok_list[idx - 1]

    # ------------------------------------------------------------------
    # Batch metadata queries
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

    # ------------------------------------------------------------------
    # Clip window sampling
    # ------------------------------------------------------------------

    def _sample_clip_start_positions(self, t_first, t_last):
        """Return a list of ``n_clips`` start timestamps (microseconds).

        When ``allow_clip_overlap=False``, starts are spaced so that
        consecutive windows share at most 10 % of ``duration_s``.
        When the log is too short to honour that constraint the windows
        are spread evenly across the available footage instead.
        """
        duration_us = int(self.duration_s * 1_000_000)
        max_start   = t_last - duration_us

        # Log shorter than one window — pin everything to t_first.
        if max_start <= t_first:
            return [t_first] * self.n_clips

        if self.n_clips == 1:
            if self.random_jiggle_part:
                return [int(np.random.randint(t_first, max_start + 1))]
            return [t_first]

        if not self.allow_clip_overlap:
            # Adjacent windows may overlap by at most 10 % of duration.
            max_overlap_us = int(duration_us * 0.10)
            min_step_us    = duration_us - max_overlap_us   # e.g. 90 % of duration

            # Total timeline needed to place n_clips windows with min_step_us steps.
            required_span = min_step_us * (self.n_clips - 1) + duration_us

            available_span = t_last - t_first
            if required_span <= available_span:
                # Distribute the slack evenly before the first and after the last window.
                slack = available_span - required_span
                first_start = t_first + slack // 2
                base_starts = [first_start + i * min_step_us for i in range(self.n_clips)]
            else:
                # Not enough footage — fall back to uniform spacing across the log.
                step = available_span // self.n_clips
                base_starts = [t_first + i * step for i in range(self.n_clips)]

            if self.random_jiggle_part:
                # Jiggle each start by up to ±20 % of the inter-clip step,
                # keeping starts within [t_first, max_start].
                step_us = base_starts[1] - base_starts[0] if len(base_starts) > 1 else min_step_us
                jiggle  = max(1, int(step_us * 0.20))
                return [
                    int(np.clip(
                        s + np.random.randint(-jiggle, jiggle + 1),
                        t_first, max_start,
                    ))
                    for s in base_starts
                ]
            return [int(np.clip(s, t_first, max_start)) for s in base_starts]

        # allow_clip_overlap=True — draw starts independently.
        if self.random_jiggle_part:
            return [int(np.random.randint(t_first, max_start + 1)) for _ in range(self.n_clips)]

        # Deterministic: evenly space across [t_first, max_start].
        return [
            int(t_first + i * (max_start - t_first) / (self.n_clips - 1))
            for i in range(self.n_clips)
        ]

    # ------------------------------------------------------------------
    # Per-clip builder  (steps 4–9 from the original load_sequences)
    # ------------------------------------------------------------------

    def _build_clip(
        self,
        con,
        sampled_cam_rows,
        anchor_token,
        anchor_ts,
        log_token_bytes,
    ):
        """Build one clip dict from a list of sampled camera rows."""
        empty_meta = {
            "token": anchor_token,
            "timestamp": anchor_ts,
            "image_paths": [],
            "frame_tokens": [],
            "frame_timestamps": [],
            "map_version": "",
            "vehicle_name": "",
            "scene_name": "",
            "roadblock_ids": [],
        }

        if not sampled_cam_rows:
            return {
                "images": None,
                "ego_pose": np.zeros((0, 7), dtype=np.float32),
                "agents": [],
                "traffic_lights": [],
                "scenario_tags": [],
                "meta": empty_meta,
            }

        # ── 4. One range query for all lp tokens, then bisect per frame ─────
        img_timestamps_list = [row[1] for row in sampled_cam_rows]
        _t0 = time.perf_counter() if _TIMING else 0.0
        if img_timestamps_list:
            lp_range = self._get_lp_tokens_in_range(
                con, log_token_bytes,
                min(img_timestamps_list), max(img_timestamps_list),
            )
            lp_ts_list  = [r["timestamp"] for r in lp_range]
            lp_tok_list = [r["token"]     for r in lp_range]
        else:
            lp_ts_list  = []
            lp_tok_list = []
        _t_lp = (time.perf_counter() - _t0) * 1000 if _TIMING else 0.0

        image_paths             = []
        frame_tokens_hex        = []
        frame_timestamps        = []
        lp_token_bytes_per_frame = []

        for cam_row in sampled_cam_rows:
            filename_jpg, img_ts = cam_row
            image_paths.append(self._resolve_image_path(filename_jpg))
            lp_tok = self._bisect_nearest_token(lp_ts_list, lp_tok_list, img_ts)
            frame_tokens_hex.append(lp_tok.hex() if lp_tok is not None else "")
            frame_timestamps.append(img_ts)
            lp_token_bytes_per_frame.append(lp_tok)

        # ── 4b. Submit image reads to background threads NOW (before meta SQL)
        #        so I/O overlaps with the ego_pose / agents queries below.
        _img_executor, _img_futures = start_decode_batch(image_paths)

        # ── 5. Batch metadata queries ───────────────────────────────────
        valid_lp_tokens = [t for t in lp_token_bytes_per_frame if t is not None]

        ego_pose_map: dict = {}
        agents_map:   dict = {}
        tl_map:       dict = {}
        tags_map:     dict = {}

        _t1 = time.perf_counter() if _TIMING else 0.0
        if valid_lp_tokens:
            if "ego_pose" in self.meta_keys:
                ego_pose_map = self._batch_query_ego_pose(con, valid_lp_tokens)
            if "agents" in self.meta_keys:
                agents_map = self._batch_query_agents(con, valid_lp_tokens)
            if "traffic_lights" in self.meta_keys:
                tl_map = self._batch_query_traffic_lights(con, valid_lp_tokens)
            if "scenario_tags" in self.meta_keys:
                tags_map = self._batch_query_scenario_tags(con, valid_lp_tokens)
        _t_meta = (time.perf_counter() - _t1) * 1000 if _TIMING else 0.0

        # ── 6. Assemble per-frame lists ─────────────────────────────────
        ego_pose       = []
        agents         = []
        traffic_lights = []
        scenario_tags  = []

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

        # ── 7. Log-level metadata ───────────────────────────────────────
        map_version, vehicle_name = "", ""
        scene_name,  roadblock_ids = "", []
        if "log_meta" in self.meta_keys:
            map_version, vehicle_name = self._query_log_meta_by_log_token(con, log_token_bytes)

        # ── 8. Build ego_pose numpy array ──────────────────────────────
        ego_pose_np = (
            np.array(
                [
                    [p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz] if p is not None else [np.nan] * 7
                    for p in ego_pose
                ],
                dtype=np.float32,
            )
            if ego_pose
            else np.zeros((0, 7), dtype=np.float32)
        )

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

        _t2 = time.perf_counter() if _TIMING else 0.0
        # Collect image futures submitted before the meta queries.
        # If start_decode_batch returned no executor (e.g. single image or
        # threading disabled), fall back to synchronous decode_batch.
        if _img_executor is not None:
            images = collect_decode_batch(_img_executor, _img_futures, len(image_paths))
        else:
            images = decode_batch(image_paths)
        _t_img = (time.perf_counter() - _t2) * 1000 if _TIMING else 0.0

        if self.individual_transform is not None:
            images = np.array([self.individual_transform(img) for img in images])
        _t3 = time.perf_counter() if _TIMING else 0.0
        if self.shared_transform is not None:
            images = self.shared_transform(images)
        _t_xfrm = (time.perf_counter() - _t3) * 1000 if _TIMING else 0.0

        _timing_ms: dict | None = None
        if _TIMING:
            _t_total = _t_lp + _t_meta + _t_img + _t_xfrm
            _timing_ms = {
                "lp_query_ms": round(_t_lp, 2),
                "meta_ms":     round(_t_meta, 2),
                "img_ms":      round(_t_img, 2),
                "xfrm_ms":     round(_t_xfrm, 2),
                "total_ms":    round(_t_total, 2),
            }
            sys.stderr.write(
                f"[nuplan timing] "
                f"lp_query={_t_lp:6.1f}ms  "
                f"meta={_t_meta:6.1f}ms  "
                f"img_decode={_t_img:6.1f}ms  "
                f"transform={_t_xfrm:6.1f}ms  "
                f"total={_t_total:6.1f}ms  "
                f"n_frames={len(image_paths)}\n"
            )
            sys.stderr.flush()

        return {
            "images":   images,
            "ego_pose": ego_pose_np,
            "agents":         agents,
            "traffic_lights": traffic_lights,
            "scenario_tags":  scenario_tags,
            "timing_ms":      _timing_ms,
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load_sequences(self, index):
        db_path, log_token_hex = self.samples[index]
        con = self._get_conn(db_path)
        log_token_bytes = bytearray.fromhex(log_token_hex)
        cache = self._log_time_cache.get((db_path, log_token_hex))
        if cache is None:
            cache = self._build_log_time_cache(con, log_token_bytes)
            self._log_time_cache[(db_path, log_token_hex)] = cache

        anchor_ts    = cache["anchor_ts"]
        anchor_token = cache["anchor_token"]

        all_cam_rows    = cache["cam_rows"]
        cam_timestamps  = cache["cam_timestamps"]

        if not all_cam_rows:
            return None

        if not cam_timestamps:
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
            return {"clips": [empty_clip] * self.n_clips}

        # ── 3. Sample n_clips window start positions via camera timestamps ──
        duration_us     = int(self.duration_s * 1_000_000)
        t_first         = cam_timestamps[0]
        t_last          = cam_timestamps[-1]
        start_positions = self._sample_clip_start_positions(t_first, t_last)

        # ── Build one sampled_cam_rows list per clip, then call _build_clip ──
        fpcs = max(1, self.fpcs)
        clips = []
        for start_ts in start_positions:
            end_ts  = start_ts + duration_us
            i_start = bisect.bisect_left(cam_timestamps, start_ts)
            i_end   = bisect.bisect_right(cam_timestamps, end_ts)
            window  = all_cam_rows[i_start:i_end]

            if len(window) <= fpcs:
                sampled_cam_rows = window
            else:
                indices          = np.linspace(0, len(window) - 1, fpcs, dtype=int)
                sampled_cam_rows = [window[i] for i in indices]

            clips.append(
                self._build_clip(
                    con, sampled_cam_rows, anchor_token, anchor_ts,
                    log_token_bytes,
                )
            )

        return {"clips": clips}

class NuplanNumpyDataset(Dataset):
    """Action-conditioned video dataset backed by pre-distilled ``.npz`` files.

    Each sample is one log entry.  The loader picks ``n_clips`` time windows
    of ``duration_s`` seconds from that log and returns them as a
    ``{"clips": [...]}`` dict, exactly like ``NuplanDataset``.
    ``CollateNpz`` (or the original ``CollateNuplan``) flattens the clips so
    the effective batch size is ``B * n_clips``.

    Only ``"ego_pose"`` metadata is available (agents, traffic lights, and
    scenario tags are returned as empty lists).  All other parameters match
    ``NuplanDataset`` 1-to-1.

    Args:
        npz_paths:           Iterable of ``.npz`` file paths produced by
                             ``distill_to_npz.py``.
        image_root:          Root directory prepended to image.filename_jpg.
        fpcs:                Camera frames to return per clip.
        duration_s:          Length of each time window in seconds.
        random_jiggle_part:  Randomly shift sampled window starts.
        shared_transform:    Applied to the decoded image array (N, H, W, C).
        individual_transform:Applied per-frame before stacking.
        n_clips:             Number of temporal clips per log.
        allow_clip_overlap:  See ``NuplanDataset`` for semantics.
    """

    def __init__(
        self,
        npz_paths,
        image_root: str = "./",
        fpcs: int = 16,
        duration_s: float = 10.0,
        random_jiggle_part: bool = True,
        shared_transform=None,
        individual_transform=None,
        n_clips: int = 1,
        allow_clip_overlap: bool = True,
    ):
        super().__init__()
        self.image_root = image_root
        self.fpcs = fpcs
        self.duration_s = duration_s
        self.random_jiggle_part = random_jiggle_part
        self.individual_transform = individual_transform
        self.shared_transform = shared_transform
        self.n_clips = max(1, int(n_clips))
        self.allow_clip_overlap = allow_clip_overlap

        # ── Build log-dir lookup (same as NuplanDataset) ──────────────────
        self._log_dirs: dict[str, str] = {}
        image_root_abs = os.path.abspath(image_root)
        if os.path.isdir(image_root_abs):
            print(f"[NpzDataset] Scanning log dirs under {image_root_abs} …")
            for entry in os.scandir(image_root_abs):
                if entry.is_dir():
                    for sub in os.scandir(entry.path):
                        if sub.is_dir() and sub.name not in self._log_dirs:
                            self._log_dirs[sub.name] = entry.path
            print(f"[NpzDataset] Found {len(self._log_dirs)} log dirs.")

        # ── Load index from .npz files ────────────────────────────────────
        self.samples: list[tuple[str, str]] = []
        self._log_time_cache: dict[tuple[str, str], dict] = {}

        npz_list = list(npz_paths)
        print(f"[NpzDataset] Indexing {len(npz_list)} npz file(s) …")
        for npz_path in tqdm(npz_list, desc="Loading NPZ index"):
            npz_path = str(npz_path)
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as exc:
                print(f"[NpzDataset] WARNING – could not load {npz_path}: {exc}")
                continue

            log_token_hex   = str(data["log_token"])
            timestamps      = data["timestamps"].astype(np.int64)
            image_paths_arr = data["image_paths"]
            ego_pose_arr    = data["ego_pose"].astype(np.float32)

            if len(timestamps) == 0:
                continue

            cam_rows       = list(zip(image_paths_arr.tolist(), timestamps.tolist()))
            cam_timestamps = timestamps.tolist()

            key = (npz_path, log_token_hex)
            self.samples.append(key)
            self._log_time_cache[key] = {
                "anchor_ts":      int(timestamps[0]),
                "anchor_token":   log_token_hex,
                "cam_rows":       cam_rows,
                "cam_timestamps": cam_timestamps,
                "ego_pose_arr":   ego_pose_arr,   # (N, 7) float32
            }

        print(f"[NpzDataset] Indexed {len(self.samples)} log(s).")

    # ------------------------------------------------------------------
    # Image path resolution
    # ------------------------------------------------------------------

    def _resolve_image_path(self, filename_jpg: str) -> str:
        log_name = filename_jpg.split("/")[0]
        base = self._log_dirs.get(log_name, self.image_root)
        return os.path.join(base, filename_jpg)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        for retry in range(5):
            try:
                sample = self.load_sequences(index)
                if sample is not None:
                    return sample
            except Exception as exc:
                if retry < 4:
                    print(f"[NpzDataset] Error at {index=}, retrying ({retry+1}/5): {exc}")
                else:
                    print(f"[NpzDataset] Failed at {index=} after 5 retries: {exc}")
        return None

    # ------------------------------------------------------------------
    # Clip window sampling (mirrors NuplanDataset exactly)
    # ------------------------------------------------------------------

    def _sample_clip_start_positions(self, t_first: int, t_last: int) -> list[int]:
        duration_us = int(self.duration_s * 1_000_000)
        max_start   = t_last - duration_us

        if max_start <= t_first:
            return [t_first] * self.n_clips

        if self.n_clips == 1:
            if self.random_jiggle_part:
                return [int(np.random.randint(t_first, max_start + 1))]
            return [t_first]

        if not self.allow_clip_overlap:
            max_overlap_us = int(duration_us * 0.10)
            min_step_us    = duration_us - max_overlap_us
            required_span  = min_step_us * (self.n_clips - 1) + duration_us
            available_span = t_last - t_first

            if required_span <= available_span:
                slack       = available_span - required_span
                first_start = t_first + slack // 2
                base_starts = [first_start + i * min_step_us for i in range(self.n_clips)]
            else:
                step        = available_span // self.n_clips
                base_starts = [t_first + i * step for i in range(self.n_clips)]

            if self.random_jiggle_part:
                step_us = (base_starts[1] - base_starts[0]) if len(base_starts) > 1 else min_step_us
                jiggle  = max(1, int(step_us * 0.20))
                return [
                    int(np.clip(s + np.random.randint(-jiggle, jiggle + 1), t_first, max_start))
                    for s in base_starts
                ]
            return [int(np.clip(s, t_first, max_start)) for s in base_starts]

        # allow_clip_overlap=True
        if self.random_jiggle_part:
            return [int(np.random.randint(t_first, max_start + 1)) for _ in range(self.n_clips)]
        return [
            int(t_first + i * (max_start - t_first) / (self.n_clips - 1))
            for i in range(self.n_clips)
        ]

    # ------------------------------------------------------------------
    # Clip builder
    # ------------------------------------------------------------------

    def _build_clip(
        self,
        sampled_cam_rows: list[tuple[str, int]],
        sampled_ego_pose: np.ndarray,
        anchor_token: str,
        anchor_ts: int,
    ) -> dict:
        empty_meta = {
            "token": anchor_token, "timestamp": anchor_ts,
            "image_paths": [], "frame_tokens": [], "frame_timestamps": [],
            "map_version": "", "vehicle_name": "", "scene_name": "", "roadblock_ids": [],
        }
        if not sampled_cam_rows:
            return {
                "images": None, "ego_pose": np.zeros((0, 7), dtype=np.float32),
                "agents": [], "traffic_lights": [], "scenario_tags": [],
                "timing_ms": None, "meta": empty_meta,
            }

        image_paths      = [self._resolve_image_path(row[0]) for row in sampled_cam_rows]
        frame_timestamps = [row[1] for row in sampled_cam_rows]
        frame_tokens_hex = [""] * len(sampled_cam_rows)

        # Submit image reads before any CPU work (no SQL to overlap, but
        # keeps the same decode path — with DCT-domain scaling — as NuplanDataset).
        _img_executor, _img_futures = start_decode_batch(image_paths)

        # ego_pose array: (N, 7) float32
        if sampled_ego_pose.size > 0:
            ego_pose_np = np.asarray(sampled_ego_pose, dtype=np.float32)
        else:
            ego_pose_np = np.full((len(sampled_cam_rows), 7), np.nan, dtype=np.float32)

        # Collect images
        if _img_executor is not None:
            images = collect_decode_batch(_img_executor, _img_futures, len(image_paths))
        else:
            images = decode_batch(image_paths)

        if images is not None and self.individual_transform is not None:
            images = np.array([self.individual_transform(img) for img in images])
        if images is not None and self.shared_transform is not None:
            images = self.shared_transform(images)

        return {
            "images":         images,
            "ego_pose":       ego_pose_np,
            "agents":         [],
            "traffic_lights": [],
            "scenario_tags":  [],
            "timing_ms":      None,
            "meta": {
                "token":            anchor_token,
                "timestamp":        anchor_ts,
                "image_paths":      image_paths,
                "frame_tokens":     frame_tokens_hex,
                "frame_timestamps": frame_timestamps,
                "map_version":      "",
                "vehicle_name":     "",
                "scene_name":       "",
                "roadblock_ids":    [],
            },
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load_sequences(self, index: int) -> dict | None:
        key   = self.samples[index]
        cache = self._log_time_cache[key]

        anchor_ts       = cache["anchor_ts"]
        anchor_token    = cache["anchor_token"]
        all_cam_rows    = cache["cam_rows"]
        cam_timestamps  = cache["cam_timestamps"]
        ego_pose_arr    = cache["ego_pose_arr"]

        if not all_cam_rows:
            return None

        duration_us     = int(self.duration_s * 1_000_000)
        t_first         = cam_timestamps[0]
        t_last          = cam_timestamps[-1]
        start_positions = self._sample_clip_start_positions(t_first, t_last)

        fpcs  = max(1, self.fpcs)
        clips = []
        for start_ts in start_positions:
            end_ts  = start_ts + duration_us
            i_start = bisect.bisect_left(cam_timestamps,  start_ts)
            i_end   = bisect.bisect_right(cam_timestamps, end_ts)
            window  = all_cam_rows[i_start:i_end]
            ep_win  = ego_pose_arr[i_start:i_end]

            if len(window) <= fpcs:
                sampled_rows     = window
                sampled_ego_pose = ep_win
            else:
                indices          = np.linspace(0, len(window) - 1, fpcs, dtype=int)
                sampled_rows     = [window[i] for i in indices]
                sampled_ego_pose = ep_win[indices]

            clips.append(
                self._build_clip(sampled_rows, sampled_ego_pose, anchor_token, anchor_ts)
            )

        return {"clips": clips}
