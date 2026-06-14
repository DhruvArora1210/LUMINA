"""
Lumina v4 — Core Data Models
Single source of truth for all data structures.

v4 upgrades over v3:
  • 3D translation vector (X, Y, Z) replaces pinhole distance scalar
  • Trigonometric azimuth (arctan2) replaces pixel-proportion heading
  • Re-ID embedding field on SpatialMemory for visual deduplication
  • Probabilistic memory permanence — confidence curve, no hard stale-cliff
  • AvoidanceWaypoint model for dynamic obstacle detours
  • EdgeModelConfig for offline SLM descriptor
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Literal, Tuple

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Known Object Heights — fallback when depth model unavailable
# ─────────────────────────────────────────────────────────────
KNOWN_HEIGHTS_M: dict[str, float] = {
    "person": 1.70, "dog": 0.50, "cat": 0.30, "horse": 1.60,
    "chair": 0.90, "couch": 0.90, "bed": 0.60, "dining table": 0.75,
    "toilet": 0.45, "tv": 0.55, "refrigerator": 1.80,
    "car": 1.50, "bicycle": 1.10, "motorcycle": 1.20,
    "bottle": 0.25, "cup": 0.12, "wine glass": 0.22, "bowl": 0.12,
    "cell phone": 0.15, "laptop": 0.25, "keyboard": 0.04,
    "book": 0.28, "backpack": 0.50, "handbag": 0.30, "suitcase": 0.70,
    "umbrella": 0.90, "remote": 0.18, "sports ball": 0.22,
    "apple": 0.09, "banana": 0.20,
}
DEFAULT_OBJECT_HEIGHT_M = 0.30

# Camera intrinsics defaults (updated at runtime from actual frame)
DEFAULT_FX = 554.0
DEFAULT_FY = 554.0
DEFAULT_CX = 320.0
DEFAULT_CY = 240.0


# ─────────────────────────────────────────────────────────────
# Spatial Math Utilities
# ─────────────────────────────────────────────────────────────

def auto_focal_length(frame_width: int, fov_h_deg: float = 62.0) -> float:
    """Compute focal length in pixels from frame geometry."""
    return (frame_width / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))


def estimate_distance_geometric(label: str, bbox_height_px: float,
                                 focal_length_px: float = 554.0) -> float:
    """
    Geometric pinhole distance estimate — fallback when depth map unavailable.
    Returns metric distance in metres, clamped to [0.1, 10.0].
    """
    if bbox_height_px < 1:
        return 3.0
    known_h = KNOWN_HEIGHTS_M.get(label.lower(), DEFAULT_OBJECT_HEIGHT_M)
    dist = (known_h * focal_length_px) / bbox_height_px
    return round(max(0.1, min(10.0, dist)), 2)


def backproject_to_3d(
    cx_px: float, cy_px: float,
    depth_m: float,
    fx: float = DEFAULT_FX, fy: float = DEFAULT_FY,
    cx_intr: float = DEFAULT_CX, cy_intr: float = DEFAULT_CY,
) -> Tuple[float, float, float]:
    """
    Back-project a pixel (cx_px, cy_px) at known depth depth_m into
    camera-coordinate 3D space (X right, Y down, Z forward).

        X = (cx_px - cx_intr) * depth_m / fx
        Y = (cy_px - cy_intr) * depth_m / fy
        Z = depth_m
    """
    X = (cx_px - cx_intr) * depth_m / fx
    Y = (cy_px - cy_intr) * depth_m / fy
    Z = depth_m
    return round(X, 3), round(Y, 3), round(Z, 3)


def azimuth_from_3d(X: float, Z: float) -> float:
    """
    Compute horizontal azimuth relative to camera forward axis.
        θ = arctan2(X, Z)   in degrees
    Positive = right of camera, negative = left of camera.
    """
    return round(math.degrees(math.atan2(X, max(Z, 0.01))), 2)


def pixel_to_heading(x_norm: float, compass_heading: float, fov_h: float = 62.0) -> float:
    """
    Convert normalised x position [0,1] to absolute compass heading.
    Kept for legacy compatibility; prefer azimuth_from_3d for new code.
    """
    offset = (x_norm - 0.5) * fov_h
    return (compass_heading + offset) % 360


def relative_angle(angle_abs: float, current_heading: float) -> float:
    """Angle relative to current facing. Positive = right, negative = left."""
    rel = (angle_abs - current_heading + 180) % 360 - 180
    return round(rel, 1)


def to_clock_direction(rel_angle: float) -> Tuple[str, str]:
    """Convert relative angle to clock direction + turn instruction."""
    directions = [
        (0,    "12 o'clock", "Go straight ahead"),
        (30,   "1 o'clock",  "Turn slightly right"),
        (60,   "2 o'clock",  "Turn right"),
        (90,   "3 o'clock",  "Turn sharply right"),
        (120,  "4 o'clock",  "Turn far right"),
        (150,  "5 o'clock",  "Turn almost behind right"),
        (180,  "6 o'clock",  "Turn around"),
        (-30,  "11 o'clock", "Turn slightly left"),
        (-60,  "10 o'clock", "Turn left"),
        (-90,  "9 o'clock",  "Turn sharply left"),
        (-120, "8 o'clock",  "Turn far left"),
        (-150, "7 o'clock",  "Turn almost behind left"),
    ]
    # Normalise to (-180, 180] before matching, so inputs from any
    # convention (e.g. raw 0-360 compass headings) wrap correctly.
    norm_angle = ((rel_angle + 180) % 360) - 180
    if norm_angle == -180:
        norm_angle = 180

    def _angular_dist(a: float, b: float) -> float:
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    best = min(directions, key=lambda d: _angular_dist(norm_angle, d[0]))
    return best[1], best[2]

def format_distance(d: float) -> str:
    if d < 1.0:
        return f"{int(d * 100)} centimetres"
    return f"{d:.1f} metres"


def format_time_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    return f"{int(seconds // 3600)} hours ago"


def probabilistic_confidence(
    original_confidence: float,
    age_seconds: float,
    half_life_hours: float = 2.0,
) -> float:
    """
    Probabilistic Object Permanence: smooth exponential decay over time.
    Replaces the 30-min hard cliff. Never reaches zero.
    Returns confidence in [0.05, original_confidence].

    At age=0          → original_confidence
    At age=half_life  → original_confidence * 0.5
    At age=8h         → ~original_confidence * 0.06  (still non-zero)
    """
    half_life_s = half_life_hours * 3600.0
    decayed = original_confidence * math.exp(-0.693 * age_seconds / half_life_s)
    return max(0.05, min(original_confidence, round(decayed, 4)))


# ─────────────────────────────────────────────────────────────
# Camera Intrinsics (runtime-updated)
# ─────────────────────────────────────────────────────────────

class CameraIntrinsics(BaseModel):
    fx: float = DEFAULT_FX
    fy: float = DEFAULT_FY
    cx: float = DEFAULT_CX
    cy: float = DEFAULT_CY
    frame_width: int = 640
    frame_height: int = 480

    def backproject(self, px: float, py: float, depth_m: float) -> Tuple[float, float, float]:
        return backproject_to_3d(px, py, depth_m, self.fx, self.fy, self.cx, self.cy)

    @classmethod
    def from_frame(cls, frame_width: int, frame_height: int, fov_h_deg: float = 62.0):
        fx = auto_focal_length(frame_width, fov_h_deg)
        fy = fx  # square pixels assumption
        return cls(
            fx=fx, fy=fy,
            cx=frame_width / 2.0, cy=frame_height / 2.0,
            frame_width=frame_width, frame_height=frame_height,
        )


# ─────────────────────────────────────────────────────────────
# Vision / Detection Models
# ─────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float

    @property
    def center_x(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def center_y(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def width(self) -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def area(self) -> float: return self.width * self.height


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox
    frame_width: int
    frame_height: int


class TrackedDetection(BaseModel):
    """Detection enriched with tracker + depth output."""
    label: str
    confidence: float
    bbox: BoundingBox
    frame_width: int
    frame_height: int
    track_id: int
    track_state: Literal["new", "stable", "moving", "lost"] = "new"
    # v3: scalar
    smoothed_distance_m: float = 0.0
    approach_velocity: float = 0.0
    # v4: full 3D translation vector
    translation_x: float = 0.0   # metres right of camera axis
    translation_y: float = 0.0   # metres below camera axis
    translation_z: float = 0.0   # metres forward (depth)
    azimuth_deg: float = 0.0     # θ = arctan2(X, Z) in degrees
    # v4: Re-ID embedding (lightweight 64-d colour histogram)
    reid_embedding: Optional[List[float]] = None


# ─────────────────────────────────────────────────────────────
# Spatial Memory (stored in Qdrant)
# ─────────────────────────────────────────────────────────────

class SpatialMemory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    confidence: float
    angle_abs: float
    distance_m: float
    frame_x_norm: float
    frame_y_norm: float
    timestamp: float
    session_id: str
    user_id: str = "default_user"
    track_id: Optional[int] = None
    approach_velocity: float = 0.0
    # v4: 3D position
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    azimuth_deg: float = 0.0
    # v4: Re-ID embedding for visual deduplication (stored as list for JSON compat)
    reid_embedding: Optional[List[float]] = None
    # v4: Decay parameters
    original_confidence: float = 0.0   # confidence at time of observation
    memory_half_life_hours: float = 2.0  # per-object decay rate


class MemorySearchResult(BaseModel):
    memory: SpatialMemory
    score: float
    match_type: Literal["exact", "semantic"]
    # v4: effective (decayed) confidence for display
    effective_confidence: float = 0.0
    age_seconds: float = 0.0

    def model_post_init(self, __context) -> None:
        if self.effective_confidence == 0.0:
            age = time.time() - self.memory.timestamp
            orig = self.memory.original_confidence or self.memory.confidence
            self.effective_confidence = probabilistic_confidence(
                orig, age, self.memory.memory_half_life_hours
            )
            self.age_seconds = age


# ─────────────────────────────────────────────────────────────
# Navigation Output
# ─────────────────────────────────────────────────────────────

@dataclass
class SpatialResult:
    label: str
    distance_m: float
    angle_abs: float
    angle_relative: float
    clock_direction: str
    turn_instruction: str
    distance_str: str
    time_ago_str: str
    confidence: float
    memory_id: str
    # v4: 3D vector and azimuth
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    azimuth_deg: float = 0.0
    # v4: memory age context for probabilistic verbosity
    age_seconds: float = 0.0
    is_stale: bool = False        # guidance-quality flag (not a hard block)
    stale_message: str = ""       # e.g. "I last saw this 2 hours ago"

    def to_speech(self) -> str:
        stale_prefix = f"{self.stale_message} " if self.is_stale and self.stale_message else ""
        return (
            f"{stale_prefix}{self.turn_instruction} to {self.clock_direction} — "
            f"your {self.label} is {self.distance_str} away. "
            f"Last seen {self.time_ago_str}, confidence {self.confidence:.0%}."
        )


# ─────────────────────────────────────────────────────────────
# v4: Dynamic Obstacle Avoidance
# ─────────────────────────────────────────────────────────────

@dataclass
class AvoidanceWaypoint:
    """
    A computed detour vector around a dynamic obstacle.
    The Coordinator emits this instead of halting navigation.
    """
    obstacle_label: str
    obstacle_distance_m: float
    obstacle_track_id: int
    # Lateral step vector: strafe direction and distance
    strafe_direction: Literal["left", "right"]  # which side to dodge
    strafe_distance_m: float                      # how far to step sideways
    forward_clearance_m: float                    # safe gap after strafe
    clock_instruction: str                        # e.g. "Step left 0.8m"
    computed_at: float = field(default_factory=time.time)

    def to_speech(self) -> str:
        return (
            f"Dynamic obstacle: {self.obstacle_label} at "
            f"{self.obstacle_distance_m:.1f}m. "
            f"Step {self.strafe_direction} {self.strafe_distance_m:.1f}m, "
            f"then continue forward."
        )


# ─────────────────────────────────────────────────────────────
# Safety
# ─────────────────────────────────────────────────────────────

class SafetyAlert(BaseModel):
    level: Literal["critical", "warning", "caution"]
    label: str
    distance_m: float
    clock_direction: str
    message: str
    track_id: int
    timestamp: float
    # v4: avoidance vector attached to critical alerts
    avoidance: Optional[dict] = None


# ─────────────────────────────────────────────────────────────
# World Model
# ─────────────────────────────────────────────────────────────

class WorldObjectState(BaseModel):
    track_id: int
    label: str
    state: str
    distance_m: float
    is_approaching: bool
    velocity_m_s: float
    age_s: float
    # v4
    azimuth_deg: float = 0.0
    translation_x: float = 0.0
    translation_z: float = 0.0


# ─────────────────────────────────────────────────────────────
# Routine Engine
# ─────────────────────────────────────────────────────────────

class RoutineEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    label: str
    angle_abs: float
    distance_m: float
    time_bucket: str   # "morning" | "afternoon" | "evening" | "night"
    observation_count: int = 1
    last_updated: float = Field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────
# Edge Model Configuration (v4 — offline SLM)
# ─────────────────────────────────────────────────────────────

class EdgeModelConfig(BaseModel):
    """
    Descriptor for the local on-device quantized SLM.
    Supported backends: llama_cpp, ollama, ctransformers, transformers (INT4).
    """
    backend: Literal["llama_cpp", "ollama", "ctransformers", "transformers", "none"] = "none"
    model_path: str = ""           # path to GGUF / safetensors model file
    model_name: str = ""           # for ollama: model name tag (e.g. "phi3:mini")
    n_ctx: int = 2048
    n_threads: int = 4
    temperature: float = 0.1
    max_tokens: int = 256
    # Quantization hint for transformers backend
    load_in_4bit: bool = True


# ─────────────────────────────────────────────────────────────
# Agent System
# ─────────────────────────────────────────────────────────────

AgentName = Literal[
    "ARCHIVIST", "JANITOR", "LIBRARIAN", "COORDINATOR",
    "CRITIC", "SAFETY", "NARRATOR", "SYSTEM", "AVOIDER"
]
LogLevel = Literal["info", "success", "warning", "error", "debug"]


class AgentLog(BaseModel):
    agent: AgentName
    level: LogLevel
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[dict] = None


class CriticVerdict(BaseModel):
    approved: bool
    reason: str
    confidence_override: Optional[float] = None
    # v4: instead of rejecting, attach avoidance waypoint
    avoidance_waypoint: Optional[AvoidanceWaypoint] = None
    # v4: soft staleness warning (doesn't block navigation)
    stale_warning: str = ""


# ─────────────────────────────────────────────────────────────
# WebSocket Message Protocol
# ─────────────────────────────────────────────────────────────

class WSFrame(BaseModel):
    type: Literal["frame"] = "frame"
    jpeg_b64: str
    detections: List[dict]
    compass_heading: float


class WSAgentLog(BaseModel):
    type: Literal["agent_log"] = "agent_log"
    agent: AgentName
    level: LogLevel
    message: str
    timestamp: str
    metadata: Optional[dict] = None


class WSMemoryUpdate(BaseModel):
    type: Literal["memory_update"] = "memory_update"
    objects: List[dict]


class WSResponse(BaseModel):
    type: Literal["response"] = "response"
    text: str
    target: str
    confidence: float
    navigation: Optional[dict] = None
    critic_approved: bool
    safety_override: bool = False
    avoidance: Optional[dict] = None    # v4: detour vector if path blocked


class WSSafetyAlert(BaseModel):
    type: Literal["safety_alert"] = "safety_alert"
    level: str
    message: str
    distance_m: float
    label: str
    clock_direction: str = ""
    avoidance: Optional[dict] = None    # v4: detour instructions


class WSWorldUpdate(BaseModel):
    type: Literal["world_update"] = "world_update"
    active_objects: List[dict]
    recent_events: List[dict]


class WSSystemStatus(BaseModel):
    type: Literal["system_status"] = "system_status"
    qdrant: bool
    groq: bool
    openai: bool
    edge_model: bool         # v4: local SLM status
    camera: bool
    depth_engine: bool       # v4: monocular depth active
    model_active: str
    ambient_mode: bool = False
    cross_session: bool = False


class WSAmbientSpeech(BaseModel):
    type: Literal["ambient_speech"] = "ambient_speech"
    text: str
    trigger: str  # "new_object" | "approaching" | "object_lost" | "routine_anomaly"


class WSAvoidance(BaseModel):
    type: Literal["avoidance"] = "avoidance"
    instruction: str
    strafe_direction: str
    strafe_distance_m: float
    obstacle_label: str
    obstacle_distance_m: float


class WSError(BaseModel):
    type: Literal["error"] = "error"
    message: str
    agent: Optional[AgentName] = None
