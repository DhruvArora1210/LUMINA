"""
Lumina v5 — All Agents  (EventBus-native)

Architecture generation: v5
Compatible with: orchestrator_fixed.py (v5 bootstrap)
Replaces: agents-5-fixed__1_.py  (which is the v4 procedural design)

WHAT CHANGED FROM v4 → v5
──────────────────────────
v4 agents:
  - __init__(self, ..., emit: LogEmitter, ...)
  - expose direct-call methods: process(), filter(), search(), validate(), …
  - called sequentially by a procedural orchestrator (the "conductor" pattern)

v5 agents:
  - __init__(self, bus: EventBus, ...)
  - expose register() which self-subscribes to bus topics
  - react autonomously to published events  ← the actual "no orchestrator at runtime" claim
  - publish their outputs back onto the bus for downstream agents to consume
  - emit agent logs via bus.publish("system/agent_log", ...) instead of a callback

SUBSCRIPTION GRAPH (enforced by register()):
  vision/new_frame            → ArchivistAgent.on_new_frame
  memory/candidates_ready     → JanitorAgent.on_candidates_ready
  system/query_received       → LibrarianAgent.on_query_received
  memory/search_result        → CoordinatorAgent.on_search_result
  navigation/route_proposed   → CriticAgent.on_route_proposed
  navigation/route_rejected   → CoordinatorAgent.on_route_rejected
  hardware/safety_warning     → AvoiderAgent.on_safety_warning

The v4 cognitive logic (parse_query, compute_navigation, compose_response,
validate, build_detour_speech, etc.) is preserved exactly — only the
*interface layer* changed.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from models import (
    AgentLog, CriticVerdict, MemorySearchResult,
    SpatialMemory, SpatialResult,
    AvoidanceWaypoint,
    format_distance, format_time_ago,
    pixel_to_heading, relative_angle, to_clock_direction,
    probabilistic_confidence,
    KNOWN_HEIGHTS_M,
)
from event_bus import (
    EventBus, Event,
    FramePayload, SafetyWarningPayload,
    RouteProposalPayload, RouteRejectionPayload, RouteApprovalPayload, RouteFinalPayload,
    MemoryCandidatesPayload, MemoryWriteApprovedPayload, MemorySearchResultPayload,
    MemoryConfidenceLowPayload, CameraPanRequestPayload,
    QueryPayload, AgentLogPayload,
)

log = logging.getLogger("lumina.agents")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — emit an AgentLog onto the bus
# ─────────────────────────────────────────────────────────────────────────────

async def _bus_emit(bus: EventBus, agent_log: AgentLog) -> None:
    """Publish an AgentLog as a system/agent_log event. Replaces the v4 emit callback."""
    await bus.publish(
        "system/agent_log",
        AgentLogPayload(
            agent=agent_log.agent,
            level=agent_log.level,
            message=agent_log.message,
            metadata=agent_log.metadata,
        ),
        publisher=agent_log.agent,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WORLD MODEL  (v4 — unchanged; carried forward verbatim)
# ─────────────────────────────────────────────────────────────────────────────

_APPROACHING_THRESHOLD = -0.25
_MOVING_THRESHOLD = 0.20


@dataclass
class WorldObject:
    track_id: int
    label: str
    state: str
    distance_m: float
    angle_abs: float
    approach_velocity: float
    first_seen: float
    last_seen: float
    state_changed_at: float
    memory_id: Optional[str] = None
    translation_x: float = 0.0
    translation_z: float = 0.0
    azimuth_deg: float = 0.0

    @property
    def age_seconds(self) -> float: return time.time() - self.first_seen
    @property
    def time_since_seen(self) -> float: return time.time() - self.last_seen
    @property
    def is_approaching(self) -> bool: return self.approach_velocity < _APPROACHING_THRESHOLD
    @property
    def is_obstacle_risk(self) -> bool: return self.distance_m < 1.5 and self.is_approaching


class WorldModel:
    """
    Live scene graph with temporal event log. (v4 — unchanged)
    Carried forward as-is; WorldModel is not an agent and has no bus interface.
    """
    LOST_TIMEOUT = 3.0
    DELETE_TIMEOUT = 60.0

    def __init__(self):
        self._objects: Dict[int, WorldObject] = {}
        self._events: List[dict] = []

    def update(self, tracks: list) -> List[dict]:
        now = time.time()
        events: List[dict] = []
        seen_ids = {t.id for t in tracks}

        for track in tracks:
            tid = track.id
            new_state = self._compute_state(track)
            prev = self._objects.get(tid)

            if prev is None:
                obj = WorldObject(
                    track_id=tid, label=track.label, state=new_state,
                    distance_m=track.smoothed_distance,
                    angle_abs=getattr(track, "azimuth_deg", 0.0),
                    approach_velocity=track.approach_velocity,
                    first_seen=now, last_seen=now, state_changed_at=now,
                    translation_x=getattr(track, "translation_x", 0.0),
                    translation_z=getattr(track, "translation_z", 0.0),
                    azimuth_deg=getattr(track, "azimuth_deg", 0.0),
                )
                self._objects[tid] = obj
                events.append({"type": "appeared", "track_id": tid,
                               "label": track.label, "distance_m": track.smoothed_distance})
            else:
                old_state = prev.state
                prev.distance_m = track.smoothed_distance
                prev.approach_velocity = track.approach_velocity
                prev.last_seen = now
                prev.translation_x = getattr(track, "translation_x", prev.translation_x)
                prev.translation_z = getattr(track, "translation_z", prev.translation_z)
                prev.azimuth_deg = getattr(track, "azimuth_deg", prev.azimuth_deg)
                if new_state != old_state:
                    prev.state = new_state
                    prev.state_changed_at = now
                    events.append({"type": "state_change", "track_id": tid,
                                   "label": track.label, "from": old_state, "to": new_state,
                                   "distance_m": track.smoothed_distance})

        for tid, obj in list(self._objects.items()):
            if tid not in seen_ids:
                if obj.time_since_seen > self.DELETE_TIMEOUT:
                    del self._objects[tid]
                    events.append({"type": "deleted", "track_id": tid, "label": obj.label})
                elif obj.time_since_seen > self.LOST_TIMEOUT and obj.state != "lost":
                    obj.state = "lost"
                    events.append({"type": "lost", "track_id": tid, "label": obj.label})

        if events:
            self._events.extend(events)
            self._events = self._events[-200:]
        return events

    def set_angle(self, track_id: int, angle_abs: float):
        if track_id in self._objects:
            self._objects[track_id].angle_abs = angle_abs

    def get_scene_summary(self) -> dict:
        now = time.time()
        return {
            "active_objects": [
                {
                    "track_id": o.track_id, "label": o.label, "state": o.state,
                    "distance_m": round(o.distance_m, 2),
                    "azimuth_deg": round(o.azimuth_deg, 1),
                    "translation_x": round(o.translation_x, 3),
                    "translation_z": round(o.translation_z, 3),
                    "is_approaching": o.is_approaching,
                    "velocity_m_s": round(o.approach_velocity, 3),
                    "age_s": round(o.age_seconds, 1),
                }
                for o in self._objects.values() if o.state != "deleted"
            ],
            "recent_events": self._events[-10:],
            "total_tracked": len(self._objects),
        }

    def find_object(self, label: str) -> Optional[WorldObject]:
        for o in self._objects.values():
            if o.label == label and o.state not in ("lost", "deleted"):
                return o
        return None

    @staticmethod
    def _compute_state(track) -> str:
        v = track.approach_velocity
        if v < _APPROACHING_THRESHOLD: return "approaching"
        if v > _MOVING_THRESHOLD: return "receding"
        return "stationary"


# ─────────────────────────────────────────────────────────────────────────────
# v4 QUERY PARSER  (deterministic; used by CoordinatorAgent)
# ─────────────────────────────────────────────────────────────────────────────

_FIND_PATTERNS = [
    r"where (is|are|did i put|can i find)\b",
    r"(find|locate|show me|help me find|i need|i want|looking for|search for)\b",
    r"(can'?t see|lost|misplaced|can you find)\b",
]
_INVENTORY_PATTERNS = [
    r"what (is|are|do you see|can you see|objects)\b",
    r"(describe|scan|look around|what's (around|nearby|here|in front))\b",
]
_OBJECT_SYNONYMS = {
    "drink": "bottle", "water": "bottle", "coffee": "cup", "mug": "cup",
    "glass": "wine glass", "beer": "bottle", "phone": "cell phone",
    "mobile": "cell phone", "bag": "backpack",
    "purse": "handbag", "wallet": "handbag", "sofa": "couch",
    "table": "dining table", "fridge": "refrigerator", "tv": "tv",
    "television": "tv", "monitor": "tv", "remote": "remote",
    "clicker": "remote", "laptop": "laptop", "computer": "laptop",
    "keys": "cell phone", "fruit": "apple", "snack": "banana",
}
_STOP_WORDS = {
    "where", "is", "are", "my", "the", "a", "an", "can", "you",
    "find", "see", "i", "me", "please", "help", "need", "want",
    "put", "left", "have", "did", "do", "look", "for", "at",
}


def _deterministic_parse(text: str) -> dict:
    t = text.lower().strip()
    intent = "find"
    for p in _INVENTORY_PATTERNS:
        if re.search(p, t): intent = "inventory"; break
    for p in _FIND_PATTERNS:
        if re.search(p, t): intent = "find"; break

    for syn, label in _OBJECT_SYNONYMS.items():
        if syn in t:
            return {"target": label, "intent": intent,
                    "raw_query": text, "parser": "deterministic", "confidence": 0.9}

    words = [w for w in re.findall(r"[a-z]+", t) if w not in _STOP_WORDS]
    for w in reversed(words):
        if w in KNOWN_HEIGHTS_M:
            return {"target": w, "intent": intent,
                    "raw_query": text, "parser": "deterministic", "confidence": 0.9}
    target = words[-1] if words else text.split()[-1] if text else "object"
    return {"target": target, "intent": intent,
            "raw_query": text, "parser": "deterministic", "confidence": 0.5}


# ═════════════════════════════════════════════════════════════════════════════
# ARCHIVIST AGENT  (v5: EventBus-native)
# ═════════════════════════════════════════════════════════════════════════════

class ArchivistAgent:
    """
    v5 EventBus-native Archivist.

    Subscribes to:  vision/new_frame
    Publishes to:   memory/candidates_ready
                    system/agent_log

    Cognitive logic: v4 (3D vector + azimuth + Re-ID embedding on every
    SpatialMemory candidate).
    """
    NAME = "ARCHIVIST"

    def __init__(
        self,
        bus: EventBus,
        session_id: str,
        user_id: str,
        world_model: WorldModel,
        reid_extractor=None,
    ):
        self._bus = bus
        self._session_id = session_id
        self._user_id = user_id
        self._world = world_model
        self._reid = reid_extractor

    def register(self) -> None:
        """Attach subscriptions. Called once after bus.start()."""
        self._bus.subscribe("vision/new_frame", self.on_new_frame)
        log.debug("ArchivistAgent registered on vision/new_frame")

    # ── event handler ────────────────────────────────────────────────────────

    async def on_new_frame(self, event: Event) -> None:
        payload: FramePayload = event.payload
        tracks = getattr(payload, "tracks", [])   # SafetyCortex attaches tracks
        if not tracks:
            return

        candidates = await self._package_candidates(
            tracks=tracks,
            compass_heading=payload.heading,
            frame=payload.frame,
        )
        if candidates:
            await self._bus.publish(
                "memory/candidates_ready",
                MemoryCandidatesPayload(candidates=candidates, frame=payload.frame),
                publisher=self.NAME,
            )

    # ── cognitive core (v4 logic — unchanged) ────────────────────────────────

    async def _package_candidates(
        self, tracks: list, compass_heading: float, frame=None
    ) -> List[SpatialMemory]:
        candidates = []
        for track in tracks:
            x_norm = track.bbox.center_x / track.det.frame_width
            y_norm = track.bbox.center_y / track.det.frame_height

            if hasattr(track, "azimuth_deg") and track.azimuth_deg != 0.0:
                angle = (compass_heading + track.azimuth_deg) % 360
            else:
                angle = pixel_to_heading(x_norm, compass_heading)

            self._world.set_angle(track.id, angle)

            reid_emb = None
            if self._reid is not None and frame is not None:
                reid_emb = self._reid.extract(frame, track.bbox)

            conf = track.det.confidence
            mem = SpatialMemory(
                id=str(uuid.uuid4()),
                label=track.label,
                confidence=conf,
                original_confidence=conf,
                angle_abs=angle,
                distance_m=track.smoothed_distance,
                frame_x_norm=x_norm,
                frame_y_norm=y_norm,
                timestamp=time.time(),
                session_id=self._session_id,
                user_id=self._user_id,
                track_id=track.id,
                approach_velocity=track.approach_velocity,
                translation_x=getattr(track, "translation_x", 0.0),
                translation_y=getattr(track, "translation_y", 0.0),
                translation_z=getattr(track, "translation_z", 0.0),
                azimuth_deg=getattr(track, "azimuth_deg", 0.0),
                reid_embedding=reid_emb,
                memory_half_life_hours=2.0,
            )
            candidates.append(mem)

        if candidates:
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="info",
                message=(f"Packaged {len(candidates)} candidate(s): "
                         + ", ".join(f"{m.label}#{m.track_id} ({m.confidence:.0%})"
                                     for m in candidates)),
                metadata={"count": len(candidates)},
            ))
        return candidates


# ═════════════════════════════════════════════════════════════════════════════
# JANITOR AGENT  (v5: EventBus-native)
# ═════════════════════════════════════════════════════════════════════════════

class JanitorAgent:
    """
    v5 EventBus-native Janitor.

    Subscribes to:  memory/candidates_ready
    Publishes to:   memory/write_approved
                    system/agent_log

    Cognitive logic: v4 (visual Re-ID dedup — cosine distance on HSV
    histogram embeddings — replaces rigid spatial window).
    """
    NAME = "JANITOR"
    REID_DISTANCE_THRESHOLD = 0.15

    def __init__(
        self,
        bus: EventBus,
        dedup_distance: float,
        dedup_angle: float,
        dedup_window: float,
    ):
        self._bus = bus
        self._dist = dedup_distance
        self._angle = dedup_angle
        self._window = dedup_window
        self._track_last_write: Dict[int, float] = {}
        self._recent: List[tuple] = []
        self._reid_store: List[tuple] = []

    def register(self) -> None:
        self._bus.subscribe("memory/candidates_ready", self.on_candidates_ready)
        log.debug("JanitorAgent registered on memory/candidates_ready")

    # ── event handler ────────────────────────────────────────────────────────

    async def on_candidates_ready(self, event: Event) -> None:
        payload: MemoryCandidatesPayload = event.payload
        approved = await self._filter(payload.candidates)
        if approved:
            await self._bus.publish(
                "memory/write_approved",
                MemoryWriteApprovedPayload(approved=approved),
                publisher=self.NAME,
            )

    # ── cognitive core (v4 logic — unchanged) ────────────────────────────────

    def _purge_old(self):
        cutoff = time.time() - self._window
        self._recent = [r for r in self._recent if r[3] > cutoff]
        self._reid_store = [r for r in self._reid_store if r[2] > cutoff]
        stale_ids = [tid for tid, ts in self._track_last_write.items() if ts < cutoff]
        for tid in stale_ids:
            del self._track_last_write[tid]

    async def _filter(self, candidates: List[SpatialMemory]) -> List[SpatialMemory]:
        self._purge_old()
        approved, discarded = [], []

        for cand in candidates:
            # 1. Track-ID dedup
            if cand.track_id is not None:
                last = self._track_last_write.get(cand.track_id, 0)
                if time.time() - last < self._window:
                    discarded.append(f"{cand.label}#{cand.track_id}")
                    continue
                self._track_last_write[cand.track_id] = time.time()
                if cand.reid_embedding:
                    self._reid_store.append((cand.reid_embedding, cand.label, time.time()))
                approved.append(cand)
                continue

            # 2. Re-ID cosine dedup
            if cand.reid_embedding:
                from vision import ReIDExtractor
                is_reid_dup = False
                for stored_emb, stored_label, _ in self._reid_store:
                    if stored_label != cand.label:
                        continue
                    dist = ReIDExtractor.cosine_distance(cand.reid_embedding, stored_emb)
                    if dist < self.REID_DISTANCE_THRESHOLD:
                        is_reid_dup = True
                        break
                if is_reid_dup:
                    discarded.append(f"{cand.label}[reid_dup]")
                    continue
                self._reid_store.append((cand.reid_embedding, cand.label, time.time()))
                approved.append(cand)
                continue

            # 3. Spatial proximity fallback
            is_dup = False
            for label, ang, dist, ts in self._recent:
                if label == cand.label:
                    adiff = abs(ang - cand.angle_abs) % 360
                    if adiff > 180: adiff = 360 - adiff
                    if adiff <= self._angle and abs(dist - cand.distance_m) <= self._dist:
                        is_dup = True
                        break
            if is_dup:
                discarded.append(cand.label)
            else:
                approved.append(cand)
                self._recent.append((cand.label, cand.angle_abs, cand.distance_m, time.time()))

        if discarded:
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="debug",
                message=f"Suppressed {len(discarded)} duplicate(s): {', '.join(discarded)}",
                metadata={"discarded": discarded},
            ))
        if approved:
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="info",
                message=f"Approved {len(approved)} memory candidate(s) for write.",
                metadata={"count": len(approved)},
            ))
        return approved


# ═════════════════════════════════════════════════════════════════════════════
# LIBRARIAN AGENT  (v5: EventBus-native)
# ═════════════════════════════════════════════════════════════════════════════

class LibrarianAgent:
    """
    v5 EventBus-native Librarian.

    Subscribes to:  system/query_received
    Publishes to:   memory/search_result
                    memory/confidence_low      (active perception trigger)
                    system/request_camera_pan  (active perception signal)
                    system/agent_log

    Cognitive logic: v4 (probabilistic confidence decay; verbal staleness
    framing; no hard 30-min cliff).
    """
    NAME = "LIBRARIAN"
    FRESH_SECONDS = 300
    RECENT_SECONDS = 3600

    def __init__(self, bus: EventBus, db):
        self._bus = bus
        self._db = db

    def register(self) -> None:
        self._bus.subscribe("system/query_received", self.on_query_received)
        log.debug("LibrarianAgent registered on system/query_received")

    # ── event handler ────────────────────────────────────────────────────────

    async def on_query_received(self, event: Event) -> None:
        payload: QueryPayload = event.payload
        raw_text = payload.raw_text

        # Parse target from query (deterministic — fast, no LLM)
        parsed = _deterministic_parse(raw_text)
        target = parsed["target"]

        results = await self._search(target, min_confidence=0.0)

        if results:
            best = results[0]
            # Active perception: if effective confidence is low, request camera pan
            if best.effective_confidence < 0.30:
                await self._bus.publish(
                    "memory/confidence_low",
                    MemoryConfidenceLowPayload(
                        memory_id=best.memory.id,
                        label=best.memory.label,
                        effective_confidence=best.effective_confidence,
                        age_seconds=best.age_seconds,
                    ),
                    publisher=self.NAME,
                )
                # Estimate pan direction from memory angle
                suggested_pan = best.memory.azimuth_deg if best.memory.azimuth_deg != 0 else 0.0
                await self._bus.publish(
                    "system/request_camera_pan",
                    CameraPanRequestPayload(
                        reason="memory_decay",
                        target_label=best.memory.label,
                        requested_by=self.NAME,
                        suggested_pan_deg=suggested_pan,
                    ),
                    publisher=self.NAME,
                )

        await self._bus.publish(
            "memory/search_result",
            MemorySearchResultPayload(
                results=results,
                query=parsed["target"],
                query_text=raw_text,
            ),
            publisher=self.NAME,
        )

    # ── cognitive core (v4 logic — unchanged) ────────────────────────────────

    async def _search(self, target: str, min_confidence: float = 0.0) -> List[MemorySearchResult]:
        await _bus_emit(self._bus, AgentLog(
            agent=self.NAME, level="info",
            message=f"Step 1: Exact match for '{target}'",
        ))

        raw_results = self._db.search(target, min_confidence=min_confidence)

        if not raw_results:
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="warning",
                message="No exact match. Step 2: Semantic search…",
            ))
            return raw_results

        enriched = []
        now = time.time()
        for r in raw_results:
            age = now - r.memory.timestamp
            orig = r.memory.original_confidence or r.memory.confidence
            eff_conf = probabilistic_confidence(orig, age, r.memory.memory_half_life_hours)
            r.effective_confidence = eff_conf
            r.age_seconds = age
            enriched.append(r)

        best = enriched[0]
        age_str = format_time_ago(best.age_seconds)
        await _bus_emit(self._bus, AgentLog(
            agent=self.NAME, level="success",
            message=(f"Found '{best.memory.label}' via {best.match_type} "
                     f"(score:{best.score:.2f} "
                     f"orig_conf:{best.memory.original_confidence:.0%} "
                     f"eff_conf:{best.effective_confidence:.0%} "
                     f"age:{age_str})"),
            metadata={"match_type": best.match_type, "label": best.memory.label,
                      "age_s": best.age_seconds, "eff_conf": best.effective_confidence},
        ))
        return enriched

    async def get_memory_snapshot(self) -> List[dict]:
        memories = self._db.get_recent(limit=30)
        now = time.time()
        return [
            {
                "id": m.id, "label": m.label, "confidence": m.confidence,
                "angle_abs": m.angle_abs, "distance_m": m.distance_m,
                "timestamp": m.timestamp, "time_ago": format_time_ago(now - m.timestamp),
                "track_id": m.track_id,
                "effective_confidence": probabilistic_confidence(
                    m.original_confidence or m.confidence,
                    now - m.timestamp, m.memory_half_life_hours
                ),
            }
            for m in memories
        ]


# ═════════════════════════════════════════════════════════════════════════════
# AVOIDER AGENT  (v5: EventBus-native)
# ═════════════════════════════════════════════════════════════════════════════

class AvoiderAgent:
    """
    v5 EventBus-native Avoider.

    Subscribes to:  hardware/safety_warning
    Publishes to:   system/agent_log
                    (spoken detour is passed back via bus to VoiceOutput)

    Cognitive logic: v4 (lateral detour vector + spoken instruction).
    Also callable directly by CriticAgent (coordinator_ref pattern).
    """
    NAME = "AVOIDER"

    def __init__(self, bus: EventBus):
        self._bus = bus

    def register(self) -> None:
        self._bus.subscribe("hardware/safety_warning", self.on_safety_warning)
        log.debug("AvoiderAgent registered on hardware/safety_warning")

    # ── event handler ────────────────────────────────────────────────────────

    async def on_safety_warning(self, event: Event) -> None:
        payload: SafetyWarningPayload = event.payload
        if payload.avoidance is None:
            return  # No waypoint computed yet — SafetyCortex will escalate if needed

        speech = await self.build_detour_speech(
            waypoint=payload.avoidance,
            final_target=payload.obstacle_label,
            final_clock=payload.clock_direction,
        )
        # Publish spoken output for VoiceOutput agent / WebSocket broadcaster
        await self._bus.publish(
            "navigation/route_final",
            RouteFinalPayload(
                response_text=speech,
                spatial=None,
                verdict=None,
                query_text="[safety_warning]",
            ),
            publisher=self.NAME,
        )

    # ── cognitive core (v4 logic — unchanged) ────────────────────────────────

    async def build_detour_speech(
        self,
        waypoint: AvoidanceWaypoint,
        final_target: str,
        final_clock: str,
    ) -> str:
        speech = (
            f"Path blocked by {waypoint.obstacle_label} "
            f"at {waypoint.obstacle_distance_m:.1f}m. "
            f"Step {waypoint.strafe_direction} {waypoint.strafe_distance_m:.1f} metres, "
            f"then resume toward {final_clock} to reach your {final_target}."
        )
        await _bus_emit(self._bus, AgentLog(
            agent=self.NAME, level="warning",
            message=f"Detour computed: {speech}",
            metadata={
                "obstacle": waypoint.obstacle_label,
                "strafe_dir": waypoint.strafe_direction,
                "strafe_m": waypoint.strafe_distance_m,
                "final_target": final_target,
            },
        ))
        return speech


# ═════════════════════════════════════════════════════════════════════════════
# COORDINATOR AGENT  (v5: EventBus-native)
# ═════════════════════════════════════════════════════════════════════════════

class CoordinatorAgent:
    """
    v5 EventBus-native Coordinator.

    Subscribes to:  memory/search_result
                    navigation/route_rejected   (re-planning loop)
    Publishes to:   navigation/route_proposed
                    navigation/route_final      (after Critic approves)
                    system/agent_log

    Cognitive logic: v4 (3D azimuth nav, probabilistic confidence, avoidance
    routing, LLM-with-deterministic-fallback response composition).
    """
    NAME = "COORDINATOR"

    def __init__(self, bus: EventBus, llm, world_model: WorldModel):
        self._bus = bus
        self._llm = llm
        self._world = world_model
        # Avoider is instantiated here; CriticAgent will call it via coordinator_ref
        self._avoider = AvoiderAgent(bus)

    def register(self) -> None:
        self._bus.subscribe("memory/search_result", self.on_search_result)
        self._bus.subscribe("navigation/route_rejected", self.on_route_rejected)
        log.debug("CoordinatorAgent registered on memory/search_result + navigation/route_rejected")

    # ── event handlers ───────────────────────────────────────────────────────

    async def on_search_result(self, event: Event) -> None:
        payload: MemorySearchResultPayload = event.payload
        if not payload.results:
            # No memory found — publish empty final with apology
            await self._bus.publish(
                "navigation/route_final",
                RouteFinalPayload(
                    response_text=f"I don't have a memory of {payload.query} yet. "
                                  "Keep moving — I'll tell you when I see it.",
                    spatial=None,
                    verdict=None,
                    query_text=payload.query_text,
                ),
                publisher=self.NAME,
            )
            return

        parsed = await self.parse_query(payload.query_text)
        current_heading = self._world.get_scene_summary().get("compass_heading", 0.0) \
            if hasattr(self._world, "get_compass") else 0.0

        best = payload.results[0]
        spatial = await self.compute_navigation(best, current_heading,
                                                scene_context=self._world.get_scene_summary())
        await self._bus.publish(
            "navigation/route_proposed",
            RouteProposalPayload(
                spatial=spatial,
                results=payload.results,
                scene_context=self._world.get_scene_summary(),
                query_text=payload.query_text,
                attempt=1,
            ),
            publisher=self.NAME,
        )

    async def on_route_rejected(self, event: Event) -> None:
        payload: RouteRejectionPayload = event.payload
        if payload.original_proposal.attempt >= 3:
            # Max re-planning attempts reached — give best-effort response
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="warning",
                message="Max re-planning attempts (3) reached — using deterministic fallback.",
            ))
            spatial = payload.original_proposal.spatial
            fallback_text = spatial.to_speech() if spatial else "Unable to navigate safely."
            await self._bus.publish(
                "navigation/route_final",
                RouteFinalPayload(
                    response_text=fallback_text,
                    spatial=spatial,
                    verdict=None,
                    query_text=payload.original_proposal.query_text,
                ),
                publisher=self.NAME,
            )
            return

        # Re-plan with avoidance waypoint if provided
        orig = payload.original_proposal
        spatial = orig.spatial
        avoidance_waypoint = payload.avoidance_waypoint

        text = await self.compose_response(
            spatial,
            scene_context=orig.scene_context,
            stale_warning=payload.stale_warning,
            avoidance_waypoint=avoidance_waypoint,
        )
        # Re-propose after composing avoidance response
        await self._bus.publish(
            "navigation/route_proposed",
            RouteProposalPayload(
                spatial=spatial,
                results=orig.results,
                scene_context=orig.scene_context,
                query_text=orig.query_text,
                attempt=orig.attempt + 1,
            ),
            publisher=self.NAME,
        )

    async def compose_and_finalize(
        self,
        spatial,
        verdict,
        query_text: str,
        stale_warning: str = "",
        avoidance_waypoint: Optional[AvoidanceWaypoint] = None,
        scene_context: Optional[dict] = None,
    ) -> None:
        """
        Called by CriticAgent after approval to compose the final spoken response
        and publish it on navigation/route_final.

        This is the deliberate exception to full decoupling noted in the
        orchestrator docstring — Critic holds a coordinator_ref for this call.
        """
        response_text = await self.compose_response(
            spatial,
            scene_context=scene_context,
            stale_warning=stale_warning,
            avoidance_waypoint=avoidance_waypoint,
        )
        await self._bus.publish(
            "navigation/route_final",
            RouteFinalPayload(
                response_text=response_text,
                spatial=spatial,
                verdict=verdict,
                query_text=query_text,
            ),
            publisher=self.NAME,
        )

    # ── cognitive core (v4 logic — unchanged) ────────────────────────────────

    async def parse_query(self, raw_query: str) -> dict:
        det = _deterministic_parse(raw_query)
        await _bus_emit(self._bus, AgentLog(
            agent=self.NAME, level="info",
            message=(f"[{det['parser'].upper()}] '{raw_query}' → "
                     f"target='{det['target']}' intent={det['intent']}"),
        ))

        if det["confidence"] >= 0.6 or not self._llm:
            return det

        try:
            from main import COORDINATOR_SYSTEM
            resp = await self._llm.complete(
                system=COORDINATOR_SYSTEM, user=raw_query,
                max_tokens=80, temperature=0.1,
            )
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.strip()).strip()
            parsed = json.loads(clean)
            parsed["parser"] = "llm"
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="success",
                message=f"[LLM] target='{parsed.get('target')}' intent={parsed.get('intent')}",
                metadata=parsed,
            ))
            return parsed
        except Exception as e:
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="warning",
                message=f"LLM parse failed ({e}), using deterministic result.",
            ))
            det["parser"] = "deterministic_fallback"
            return det

    async def compute_navigation(
        self,
        result: MemorySearchResult,
        current_compass: float,
        scene_context: Optional[dict] = None,
    ) -> SpatialResult:
        mem = result.memory
        now = time.time()
        age_s = now - mem.timestamp

        if mem.azimuth_deg != 0.0 and mem.translation_z > 0.0:
            abs_angle = (current_compass + mem.azimuth_deg) % 360
        else:
            abs_angle = mem.angle_abs

        rel = relative_angle(abs_angle, current_compass)
        clock, instruction = to_clock_direction(rel)

        eff_conf = result.effective_confidence if result.effective_confidence > 0.0 else (
            probabilistic_confidence(
                mem.original_confidence or mem.confidence, age_s,
                mem.memory_half_life_hours,
            )
        )

        stale_warning = ""
        is_stale = False
        if age_s > 3600:
            stale_warning = f"I last saw your {mem.label} {format_time_ago(age_s)}"
            is_stale = True

        spatial = SpatialResult(
            label=mem.label,
            distance_m=mem.distance_m,
            angle_abs=abs_angle,
            angle_relative=rel,
            clock_direction=clock,
            turn_instruction=instruction,
            distance_str=format_distance(mem.distance_m),
            time_ago_str=format_time_ago(age_s),
            confidence=eff_conf * result.score,
            memory_id=mem.id,
            translation_x=mem.translation_x,
            translation_y=mem.translation_y,
            translation_z=mem.translation_z,
            azimuth_deg=mem.azimuth_deg,
            age_seconds=age_s,
            is_stale=is_stale,
            stale_message=stale_warning,
        )

        await _bus_emit(self._bus, AgentLog(
            agent=self.NAME, level="info",
            message=(f"Nav: {instruction} → {clock}, "
                     f"{format_distance(mem.distance_m)} "
                     f"(eff_conf:{eff_conf:.0%} age:{format_time_ago(age_s)})"),
            metadata={"angle_relative": rel, "clock": clock,
                      "distance_m": mem.distance_m, "eff_confidence": eff_conf,
                      "azimuth_deg": mem.azimuth_deg},
        ))
        return spatial

    async def compose_response(
        self,
        spatial: SpatialResult,
        scene_context: Optional[dict] = None,
        stale_warning: str = "",
        avoidance_waypoint: Optional[AvoidanceWaypoint] = None,
    ) -> str:
        if avoidance_waypoint:
            return await self._avoider.build_detour_speech(
                avoidance_waypoint, spatial.label, spatial.clock_direction
            )

        scene_str = ""
        if scene_context:
            for o in scene_context.get("active_objects", []):
                if o["label"] == spatial.label:
                    scene_str = (
                        f"Object state: {o['state']}\n"
                        f"Approach velocity: {o.get('velocity_m_s', 0.0):+.2f} m/s "
                        f"(negative=approaching you)\n"
                    )
                    break

        stale_str = f"Staleness note: {spatial.stale_message}\n" if spatial.stale_message else ""
        user_prompt = (
            f"Object: {spatial.label}\n"
            f"Effective confidence: {spatial.confidence:.0%}\n"
            f"Last seen: {spatial.time_ago_str}\n"
            f"{stale_str}"
            f"{scene_str}"
            f"Direction: {spatial.turn_instruction} to {spatial.clock_direction}\n"
            f"Distance: {spatial.distance_str}\n"
            "Compose a 2-sentence spoken response. "
            "If object is moving toward user, warn first. "
            "If stale, include the age caveat naturally in the second sentence."
        )

        try:
            from main import RESPONSE_SYSTEM
            response = await self._llm.complete(
                system=RESPONSE_SYSTEM, user=user_prompt,
                max_tokens=120, temperature=0.3,
            )
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="success",
                message=f'Response: "{response}"',
            ))
            return response
        except Exception:
            fallback = spatial.to_speech()
            await _bus_emit(self._bus, AgentLog(
                agent=self.NAME, level="warning",
                message="LLM unavailable — deterministic fallback.",
            ))
            return fallback


# ═════════════════════════════════════════════════════════════════════════════
# CRITIC AGENT  (v5: EventBus-native)
# ═════════════════════════════════════════════════════════════════════════════

class CriticAgent:
    """
    v5 EventBus-native Critic.

    Subscribes to:  navigation/route_proposed
    Publishes to:   navigation/route_approved  → triggers compose_and_finalize
                    navigation/route_rejected  → triggers CoordinatorAgent.on_route_rejected
                    system/agent_log

    Cognitive logic: v4 (soft staleness warnings; avoidance waypoint instead of
    hard halt for dynamic obstacles; confidence threshold; distance sanity).

    coordinator_ref: deliberate exception to full decoupling.
    After approval, Critic calls coordinator_ref.compose_and_finalize()
    to produce the spoken response. This avoids a round-trip publish/subscribe
    for a synchronous composition step that always follows approval.
    """
    NAME = "CRITIC"
    STALE_ADVISORY_SECONDS = 1800
    STALE_VERY_OLD_SECONDS = 14400

    def __init__(
        self,
        bus: EventBus,
        confidence_threshold: float,
        coordinator_ref: CoordinatorAgent,
    ):
        self._bus = bus
        self._threshold = confidence_threshold
        self._coordinator = coordinator_ref

    def register(self) -> None:
        self._bus.subscribe("navigation/route_proposed", self.on_route_proposed)
        log.debug("CriticAgent registered on navigation/route_proposed")

    # ── event handler ────────────────────────────────────────────────────────

    async def on_route_proposed(self, event: Event) -> None:
        payload: RouteProposalPayload = event.payload
        verdict = await self._validate(
            spatial=payload.spatial,
            results=payload.results,
            scene_context=payload.scene_context,
        )

        if verdict.approved:
            # Trigger compose_and_finalize on the coordinator
            await self._coordinator.compose_and_finalize(
                spatial=payload.spatial,
                verdict=verdict,
                query_text=payload.query_text,
                stale_warning=verdict.stale_warning,
                avoidance_waypoint=verdict.avoidance_waypoint,
                scene_context=payload.scene_context,
            )
            await self._bus.publish(
                "navigation/route_approved",
                RouteApprovalPayload(
                    spatial=payload.spatial,
                    verdict=verdict,
                    query_text=payload.query_text,
                ),
                publisher=self.NAME,
            )
        else:
            await self._bus.publish(
                "navigation/route_rejected",
                RouteRejectionPayload(
                    reason=verdict.reason,
                    original_proposal=payload,
                    avoidance_waypoint=verdict.avoidance_waypoint,
                    stale_warning=verdict.stale_warning,
                ),
                publisher=self.NAME,
            )

    # ── cognitive core (v4 logic — unchanged) ────────────────────────────────

    async def _validate(
        self,
        spatial: SpatialResult,
        results: List[MemorySearchResult],
        scene_context: Optional[dict] = None,
    ) -> CriticVerdict:
        reasons = []
        warnings = []
        approved = True
        avoidance_waypoint = None
        stale_warning = ""

        # Rule 1: Confidence threshold (hard)
        if spatial.confidence < self._threshold:
            approved = False
            reasons.append(
                f"Confidence {spatial.confidence:.0%} < threshold {self._threshold:.0%}"
            )

        # Rule 2: Staleness — soft warning only (v4)
        if results:
            age = time.time() - results[0].memory.timestamp
            if age > self.STALE_VERY_OLD_SECONDS:
                stale_warning = (
                    f"Memory is {format_time_ago(age)} old — "
                    "guiding to last known position."
                )
                warnings.append(stale_warning)
                if spatial.confidence > 0.4:
                    spatial.confidence = 0.4
            elif age > self.STALE_ADVISORY_SECONDS:
                stale_warning = f"Memory is {format_time_ago(age)} old — checking there first."
                warnings.append(stale_warning)

        # Rule 3: Distance sanity
        if spatial.distance_m > 8.0:
            warnings.append(f"Distance {spatial.distance_m:.1f}m may be inaccurate — caution")

        # Rule 4: Approaching obstacle — compute avoidance instead of hard block (v4)
        if scene_context and approved:
            @dataclass
            class _MockTrack:
                label: str
                azimuth_deg: float
                smoothed_distance: float
                translation_x: float
                id: int = 0

            for o in scene_context.get("active_objects", []):
                if o.get("is_approaching") and o["distance_m"] < 1.2:
                    label = o["label"]
                    dynamic_labels = {"person", "dog", "cat", "bicycle", "motorcycle"}
                    if label in dynamic_labels:
                        from vision import DynamicAvoidanceEngine
                        avoider_engine = DynamicAvoidanceEngine()
                        mock = _MockTrack(
                            label=label,
                            azimuth_deg=o.get("azimuth_deg", 0.0),
                            smoothed_distance=o["distance_m"],
                            translation_x=o.get("translation_x", 0.3),
                        )
                        wp = avoider_engine.compute_waypoint(
                            mock, target_azimuth_deg=spatial.azimuth_deg
                        )
                        if wp:
                            avoidance_waypoint = wp
                            warnings.append(
                                f"Approaching {label} at {o['distance_m']:.1f}m — "
                                f"detour: step {wp.strafe_direction} {wp.strafe_distance_m}m"
                            )
                        else:
                            approved = False
                            reasons.append(
                                f"Approaching {label} at {o['distance_m']:.1f}m — hold"
                            )
                    else:
                        approved = False
                        reasons.append(
                            f"Static obstacle {label} at {o['distance_m']:.1f}m — "
                            "do not navigate"
                        )
                    break

        level = "success" if approved else "error"
        all_msgs = reasons + warnings
        verdict_text = (
            ("APPROVED ✓" if approved else "REJECTED ✗")
            + (" — " + "; ".join(all_msgs) if all_msgs else " — All checks passed")
        )

        await _bus_emit(self._bus, AgentLog(
            agent=self.NAME, level=level, message=verdict_text,
            metadata={"approved": approved, "reasons": reasons, "warnings": warnings},
        ))

        return CriticVerdict(
            approved=approved,
            reason="; ".join(reasons) if reasons else "All checks passed",
            avoidance_waypoint=avoidance_waypoint,
            stale_warning=stale_warning,
        )
