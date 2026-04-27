
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    import torch


class MetaKey(StrEnum):
    """Keys controlling which NuPlan metadata queries run in NuplanDataset.

    Pass a set of these to ``NuplanDataset(meta_keys=...)``.
    Omit a key to skip its SQL query and leave the field at its default.
    """
    EGO_POSE        = "ego_pose"
    AGENTS          = "agents"
    TRAFFIC_LIGHTS  = "traffic_lights"
    SCENARIO_TAGS   = "scenario_tags"
    LOG_META        = "log_meta"


# ---------------------------------------------------------------------------
# NuPlan database schema dataclasses
# One NuplanFrame = one lidar_pc anchor timestamp with all associated data
# ---------------------------------------------------------------------------

@dataclass
class EgoPose:
    timestamp: int          # microseconds
    # Position – UTM coordinates (EPSG projection stored in ego_pose.epsg)
    x: float
    y: float
    z: float
    # Orientation – unit quaternion (w, x, y, z)
    qw: float
    qx: float
    qy: float
    qz: float
    # Velocity  m/s
    vx: float
    vy: float
    vz: float
    # Linear acceleration  m/s²
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float
    # Angular rate  rad/s
    angular_rate_x: float
    angular_rate_y: float
    angular_rate_z: float


@dataclass
class AgentBox:
    token: str
    track_token: str
    category: str           # vehicle | bicycle | pedestrian | traffic_cone | barrier | czone_sign | generic_object
    # Position  m
    x: float
    y: float
    z: float
    # Orientation  rad
    yaw: float
    # Dimensions  m
    width: float
    length: float
    height: float
    # Velocity  m/s
    vx: float
    vy: float
    vz: float
    confidence: float


@dataclass
class TrafficLight:
    lane_connector_id: int
    status: str             # 'green' | 'red'


@dataclass
class NuplanFrame:
    # Anchor identity
    token: Union[str, List[str]]        # hex token (str) or batched list
    timestamp: Union[int, "torch.Tensor"]  # microseconds; (B,) int64 when batched

    # Camera images – absolute paths, ordered by timestamp.
    image_paths: Union[List[str], List[List[str]]] = field(default_factory=list)
    frame_tokens: Union[List[str], List[List[str]]] = field(default_factory=list)
    frame_timestamps: Union[List[int], "torch.Tensor"] = field(default_factory=list)  # (B, N) int64 when batched

    # Per-frame metadata aligned by index with image_paths.
    ego_pose: Union[List[Optional[EgoPose]], "torch.Tensor"] = field(default_factory=list)  # (B, N, 7) float32 when batched
    agents: Union[List[List[AgentBox]], List[List[List[AgentBox]]]] = field(default_factory=list)
    traffic_lights: Union[List[List[TrafficLight]], List[List[List[TrafficLight]]]] = field(default_factory=list)
    scenario_tags: Union[List[List[str]], List[List[List[str]]]] = field(default_factory=list)

    # Log / scene metadata (shared for the anchor scene/log)
    map_version: str = ""
    vehicle_name: str = ""
    scene_name: str = ""
    roadblock_ids: List[str] = field(default_factory=list)

    def pin_memory(self):
        """Enable use with DataLoader pin_memory=True."""
        import torch

        def _pin(v):
            return v.pin_memory() if isinstance(v, torch.Tensor) else v

        return NuplanFrame(
            token=self.token,
            timestamp=_pin(self.timestamp),
            image_paths=self.image_paths,
            frame_tokens=self.frame_tokens,
            frame_timestamps=_pin(self.frame_timestamps),
            ego_pose=_pin(self.ego_pose),
            agents=self.agents,
            traffic_lights=self.traffic_lights,
            scenario_tags=self.scenario_tags,
            map_version=self.map_version,
            vehicle_name=self.vehicle_name,
            scene_name=self.scene_name,
            roadblock_ids=self.roadblock_ids,
        )