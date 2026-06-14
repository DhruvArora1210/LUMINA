"""
Lumina v4 — FastAPI Application + Config + LLM + Database

v4 upgrades over v3:
  ┌───────────────────────────────────────────────────────────────┐
  │  CHANGE 1 — EdgeLLMClient                                     │
  │    Cascade: Groq → OpenAI → Local SLM (llama.cpp/ollama) →   │
  │    Deterministic fallback.  Never raises RuntimeError.        │
  │    Local backend auto-detected from EdgeModelConfig.          │
  │                                                               │
  │  CHANGE 2 — Probabilistic Memory Permanence in Database       │
  │    upsert() stores original_confidence separately.            │
  │    search() returns MemorySearchResult with effective         │
  │    (decayed) confidence already computed.                     │
  │                                                               │
  │  CHANGE 3 — Avoidance WebSocket messages                      │
  │    /ws forwards WSAvoidance frames when Critic returns        │
  │    avoidance_waypoint.                                        │
  │                                                               │
  │  CHANGE 4 — System status includes depth_engine + edge_model │
  └───────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Set

from pydantic_settings import BaseSettings


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    # LLM — cloud
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-70b-8192"
    OPENAI_MODEL: str = "gpt-4o"

    # LLM — local edge (v4)
    EDGE_LLM_BACKEND: str = "none"   # "llama_cpp" | "ollama" | "none"
    EDGE_LLM_MODEL_PATH: str = ""    # Path to .gguf file (llama_cpp)
    EDGE_LLM_MODEL_NAME: str = ""    # Ollama model tag, e.g. "phi3:mini"
    EDGE_LLM_N_CTX: int = 2048
    EDGE_LLM_N_THREADS: int = 4
    EDGE_LLM_TEMPERATURE: float = 0.1
    EDGE_LLM_MAX_TOKENS: int = 256

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_OBJECTS: str = "lumina_objects"
    COLLECTION_ZONES: str = "lumina_zones"
    COLLECTION_ROUTINES: str = "lumina_routines"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Vision
    YOLO_MODEL: str = "yolov8n.pt"
    CAMERA_INDEX: int = 0
    DETECTION_CONFIDENCE: float = 0.50
    VISION_FPS: int = 8
    FRAME_JPEG_QUALITY: int = 70

    # IP Camera (mobile phone on same Wi-Fi)
    # CAMERA_MODE: "local" uses CAMERA_INDEX (laptop webcam)
    #              "ip"    uses CAMERA_IP_URL (phone IP cam)
    # Popular apps:
    #   IP Webcam (Android) → http://<phone-ip>:8080/video
    #   DroidCam            → http://<phone-ip>:4747/video
    #   iVCam / EpocCam     → rtsp://<phone-ip>:8554/live
    CAMERA_MODE: str = "local"          # "local" | "ip"
    CAMERA_IP_URL: str = ""             # e.g. "http://192.168.1.5:8080/video"
    CAMERA_IP_RECONNECT_DELAY: float = 2.0   # seconds between reconnect attempts
    CAMERA_IP_TIMEOUT_MS: int = 5000         # OpenCV read timeout for IP stream

    # Depth engine (v4)
    DEPTH_ENGINE_ENABLED: bool = True
    DEPTH_ONNX_MODEL_PATH: str = ""      # Optional path to midas_v21_small_256.onnx

    # Tracking
    TRACKER_IOU_THRESHOLD: float = 0.35
    TRACKER_MAX_AGE: int = 8
    TRACKER_MIN_HITS: int = 2

    # Spatial Memory
    MEMORY_TTL_SECONDS: int = 3600
    DEDUP_DISTANCE_METERS: float = 0.4
    DEDUP_ANGLE_DEGREES: float = 15.0
    DEDUP_TIME_WINDOW_SECONDS: float = 5.0
    CRITIC_CONFIDENCE_THRESHOLD: float = 0.60

    # Safety Cortex
    SAFETY_CRITICAL_DIST: float = 0.8
    SAFETY_WARNING_DIST: float = 1.5
    SAFETY_CAUTION_DIST: float = 2.5
    SAFETY_ALERT_COOLDOWN: float = 3.0

    # World Model
    WORLD_MODEL_LOST_TIMEOUT: float = 3.0
    WORLD_MODEL_DELETE_TIMEOUT: float = 60.0

    # Persistent Memory
    USER_ID: str = "default_user"
    MEMORY_DECAY_HALF_LIFE_HOURS: float = 2.0   # v4: per-object decay rate
    CROSS_SESSION_ENABLED: bool = True

    # Ambient Narrator
    AMBIENT_MODE_ENABLED: bool = True
    AMBIENT_SPEAK_COOLDOWN: float = 5.0
    AMBIENT_CLOSE_OBJECT_DIST: float = 2.0

    # Routine Engine
    ROUTINE_MIN_OBSERVATIONS: int = 3
    ROUTINE_ANOMALY_ANGLE_DEG: float = 60.0

    # Camera
    CAMERA_FOV_H: float = 62.0

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ══════════════════════════════════════════════════════════════
# LLM CLIENT  (v4 — edge model fallback, never fatal)
# ══════════════════════════════════════════════════════════════

log = logging.getLogger("lumina.llm")


class EdgeLLMBackend:
    """
    Runs a quantized Small Language Model on-device.
    Two supported backends:
      - llama_cpp:  pip install llama-cpp-python; loads a .gguf file
      - ollama:     ollama must be running; uses httpx REST API

    Both are optional. If neither is available, returns None.
    """

    def __init__(self, backend: str, model_path: str, model_name: str,
                 n_ctx: int, n_threads: int, temperature: float, max_tokens: int):
        self._backend = backend
        self._model_path = model_path
        self._model_name = model_name
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._llama = None       # llama_cpp.Llama instance
        self._ollama_url = "http://localhost:11434"
        self._ready = False
        self._try_init()

    def _try_init(self):
        if self._backend == "llama_cpp" and self._model_path:
            try:
                from llama_cpp import Llama
                self._llama = Llama(
                    model_path=self._model_path,
                    n_ctx=self._n_ctx,
                    n_threads=self._n_threads,
                    verbose=False,
                )
                self._ready = True
                log.info(f"llama_cpp edge model loaded: {self._model_path}")
            except ImportError:
                log.warning("llama-cpp-python not installed; edge model disabled")
            except Exception as e:
                log.warning(f"llama_cpp init failed: {e}")

        elif self._backend == "ollama" and self._model_name:
            # Ollama availability is checked at inference time (async)
            self._ready = True
            log.info(f"Ollama edge backend configured: {self._model_name}")

    @property
    def available(self) -> bool:
        return self._ready

    async def complete(self, system: str, user: str) -> Optional[str]:
        if not self._ready:
            return None
        try:
            if self._backend == "llama_cpp" and self._llama:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._llama_complete, system, user
                )
            elif self._backend == "ollama":
                return await self._ollama_complete(system, user)
        except Exception as e:
            log.warning(f"Edge LLM failed: {e}")
        return None

    def _llama_complete(self, system: str, user: str) -> str:
        """Synchronous llama_cpp inference (run in executor)."""
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        result = self._llama(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=["<|user|>", "<|system|>"],
        )
        return result["choices"][0]["text"].strip()

    async def _ollama_complete(self, system: str, user: str) -> str:
        """Async ollama REST API call."""
        try:
            import httpx
            _has_httpx = True
        except ImportError:
            _has_httpx = False

        if not _has_httpx:
            # Synchronous urllib fallback — run in executor so we don't block the event loop
            import urllib.request
            import json as _json

            def _sync_call():
                payload = _json.dumps({
                    "model": self._model_name,
                    "prompt": f"System: {system}\nUser: {user}\nAssistant:",
                    "stream": False,
                    "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
                }).encode()
                req = urllib.request.Request(
                    f"{self._ollama_url}/api/generate",
                    data=payload, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _json.loads(resp.read())
                return data.get("response", "").strip()

            return await asyncio.get_event_loop().run_in_executor(None, _sync_call)

        # httpx is available — use async client
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{self._ollama_url}/api/generate", json={
                "model": self._model_name,
                "prompt": f"System: {system}\nUser: {user}\nAssistant:",
                "stream": False,
                "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
            })
            r.raise_for_status()
            return r.json().get("response", "").strip()


class LLMClient:
    """
    v4 cascade: Groq (cloud) → OpenAI (cloud) → EdgeLLM (local) → Deterministic fallback.
    Never raises RuntimeError. Always returns a usable string.
    """

    def __init__(self, groq_key: str, openai_key: str,
                 groq_model: str, openai_model: str,
                 edge_backend: Optional[EdgeLLMBackend] = None):
        self.groq_key = groq_key
        self.openai_key = openai_key
        self.groq_model = groq_model
        self.openai_model = openai_model
        self._groq = None
        self._openai = None
        self._edge = edge_backend
        self.active_model: str = "none"

    def _get_groq(self):
        if not self._groq and self.groq_key:
            try:
                from groq import AsyncGroq
                self._groq = AsyncGroq(api_key=self.groq_key)
            except ImportError:
                log.warning("groq package not installed")
        return self._groq

    def _get_openai(self):
        if not self._openai and self.openai_key:
            try:
                from openai import AsyncOpenAI
                self._openai = AsyncOpenAI(api_key=self.openai_key)
            except ImportError:
                log.warning("openai package not installed")
        return self._openai

    async def complete(self, system: str, user: str,
                       max_tokens: int = 300, temperature: float = 0.2,
                       fallback_text: Optional[str] = None) -> str:
        import time as _time

        # ── 1. Groq ────────────────────────────────────────────
        groq = self._get_groq()
        if groq:
            try:
                t0 = _time.perf_counter()
                r = await groq.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=temperature,
                )
                text = r.choices[0].message.content.strip()
                log.info(f"[Groq] {(_time.perf_counter()-t0)*1000:.0f}ms — {len(text)} chars")
                self.active_model = "groq"
                return text
            except Exception as e:
                log.warning(f"Groq failed ({e}), trying OpenAI")

        # ── 2. OpenAI ──────────────────────────────────────────
        openai = self._get_openai()
        if openai:
            try:
                t0 = _time.perf_counter()
                r = await openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=temperature,
                )
                text = r.choices[0].message.content.strip()
                log.info(f"[OpenAI] {(_time.perf_counter()-t0)*1000:.0f}ms — {len(text)} chars")
                self.active_model = "openai"
                return text
            except Exception as e:
                log.warning(f"OpenAI failed ({e}), trying edge model")

        # ── 3. Local Edge Model ────────────────────────────────
        if self._edge and self._edge.available:
            try:
                t0 = _time.perf_counter()
                result = await self._edge.complete(system, user)
                if result:
                    log.info(f"[Edge] {(_time.perf_counter()-t0)*1000:.0f}ms — {len(result)} chars")
                    self.active_model = f"edge/{self._edge._backend}"
                    return result
            except Exception as e:
                log.warning(f"Edge LLM failed ({e}), using deterministic fallback")

        # ── 4. Deterministic fallback (never fatal) ────────────
        self.active_model = "deterministic"
        if fallback_text:
            return fallback_text
        # Return empty string — callers must handle this gracefully
        log.warning("All LLM providers failed — returning empty string for caller fallback")
        return ""

    async def health_check(self) -> dict:
        results = {
            "groq": False,
            "openai": False,
            "edge": self._edge.available if self._edge else False,
        }
        for provider, client, model in [
            ("groq", self._get_groq(), self.groq_model),
            ("openai", self._get_openai(), self.openai_model),
        ]:
            if client:
                try:
                    await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=5,
                    )
                    results[provider] = True
                except Exception:
                    pass
        return results


# ── Prompts ───────────────────────────────────────────────────

COORDINATOR_SYSTEM = """You are Lumina's Coordinator — the cognitive core of an assistive navigation system for visually impaired users.

Your role:
1. Parse the user's natural language query to extract the TARGET object they are searching for.
2. Return ONLY a JSON object: {"target": "<object label>", "intent": "find|inventory|describe", "raw_query": "<original query>"}

Rules:
- target must be a single common object noun (lowercase, singular).
- NEVER add explanation outside the JSON.

Examples:
User: "where are my car keys?" → {"target": "keys", "intent": "find", "raw_query": "where are my car keys?"}
User: "I need a drink" → {"target": "bottle", "intent": "find", "raw_query": "I need a drink"}
"""

RESPONSE_SYSTEM = """You are Lumina — a calm, precise spatial intelligence built for one person.

You speak in short, confident sentences. You never say "I think" or "maybe."
You state what you know and how confident you are. Numbers are exact.
When an object is moving toward the user, warn them clearly and first.
When something is dangerous, you are direct and firm.
When a memory is old, include a natural age caveat in the second sentence.
When helping with routine tasks, you are warm and brief.

Given spatial data about a found object, compose a 2-sentence spoken response.
Format: direction + distance → confidence + time context.
If approach_velocity is negative (object moving toward user), lead with that warning.
If the memory is stale, say when you last saw it naturally.

Examples:
"Turn right to 2 o'clock — your bottle is 1.4 metres away. I spotted it 8 minutes ago, 90% confident."
"Straight ahead, 80 centimetres — your phone is right there. Seen 30 seconds ago."
"Warning — that chair is moving toward you. Step back: it's at 11 o'clock, 1.1 metres and closing."
"Head left to 10 o'clock — your keys should be about 2.1 metres away. I last saw them 45 minutes ago, so let's check there first."
"""


# ══════════════════════════════════════════════════════════════
# SPATIAL DATABASE  (v4 — stores 3D vector + Re-ID + decay params)
# ══════════════════════════════════════════════════════════════

_db_log = logging.getLogger("lumina.database")


class SpatialDatabase:
    """
    Qdrant-backed spatial memory.
    v4: stores original_confidence, 3D vector, Re-ID embedding, half_life.
    """

    def __init__(self, host: str, port: int,
                 collection_objects: str, collection_zones: str, collection_routines: str,
                 embedding_model: str, embedding_dim: int,
                 session_id: str, user_id: str = "default_user",
                 cross_session_enabled: bool = True):
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.col_obj = collection_objects
        self.col_zone = collection_zones
        self.col_routines = collection_routines
        self.session_id = session_id
        self.user_id = user_id
        self.cross_session = cross_session_enabled
        self._dim = embedding_dim
        self._client = QdrantClient(host=host, port=port, timeout=5.0)
        self._encoder = SentenceTransformer(embedding_model)
        self._ensure_collections()
        _db_log.info(f"SpatialDatabase ready — user:{user_id} session:{session_id}")

    def _ensure_collections(self):
        from qdrant_client.models import VectorParams, Distance
        existing = {c.name for c in self._client.get_collections().collections}
        for col in [self.col_obj, self.col_zone, self.col_routines]:
            if col not in existing:
                self._client.create_collection(
                    collection_name=col,
                    vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
                )
                _db_log.info(f"Created collection: {col}")

    def upsert(self, memory) -> bool:
        from qdrant_client.models import PointStruct
        try:
            vector = self._encoder.encode(memory.label).tolist()
            payload = {
                "label": memory.label,
                "confidence": memory.confidence,
                "original_confidence": memory.original_confidence or memory.confidence,
                "angle_abs": memory.angle_abs,
                "distance_m": memory.distance_m,
                "frame_x": memory.frame_x_norm,
                "frame_y": memory.frame_y_norm,
                "timestamp": memory.timestamp,
                "session_id": memory.session_id,
                "user_id": memory.user_id,
                "track_id": memory.track_id,
                "approach_velocity": memory.approach_velocity,
                # v4: 3D vector
                "translation_x": memory.translation_x,
                "translation_y": memory.translation_y,
                "translation_z": memory.translation_z,
                "azimuth_deg": memory.azimuth_deg,
                # v4: decay params
                "memory_half_life_hours": memory.memory_half_life_hours,
                # v4: Re-ID (stored but not used as search vector — separate field)
                "reid_embedding": memory.reid_embedding,
            }
            point = PointStruct(id=memory.id, vector=vector, payload=payload)
            result = self._client.upsert(collection_name=self.col_obj, points=[point])
            # qdrant_client returns an UpdateResult; status is "completed" string
            # or a rest.UpdateStatus enum depending on client version — compare by name
            status = result.status
            return str(status).lower() in ("completed", "updatestatus.completed")
        except Exception as e:
            _db_log.error(f"Upsert failed: {e}")
            return False

    def search(self, query: str, session_id: Optional[str] = None,
               top_k: int = 5, min_confidence: float = 0.0,
               cross_session: Optional[bool] = None) -> list:
        use_cross = cross_session if cross_session is not None else self.cross_session
        sid = session_id or self.session_id

        exact = self._exact_search(query, sid, min_confidence, use_cross)
        if exact:
            from models import MemorySearchResult
            return [MemorySearchResult(memory=m, score=1.0, match_type="exact") for m in exact]

        semantic = self._semantic_search(query, sid, top_k, min_confidence, use_cross)
        if semantic:
            from models import MemorySearchResult
            return [MemorySearchResult(memory=m, score=s, match_type="semantic")
                    for m, s in semantic]
        return []

    def _session_filter(self, session_id: str, cross_session: bool):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        if cross_session:
            return Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=self.user_id))
            ])
        return Filter(must=[
            FieldCondition(key="session_id", match=MatchValue(value=session_id))
        ])

    def _exact_search(self, label: str, session_id: str,
                      min_confidence: float, cross_session: bool) -> list:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        try:
            filt = Filter(must=[
                FieldCondition(key="label", match=MatchValue(value=label.lower())),
                *(
                    [FieldCondition(key="user_id", match=MatchValue(value=self.user_id))]
                    if cross_session else
                    [FieldCondition(key="session_id", match=MatchValue(value=session_id))]
                )
            ])
            hits, _ = self._client.scroll(
                collection_name=self.col_obj, scroll_filter=filt,
                limit=10, with_payload=True, with_vectors=False,
            )
            memories = [self._to_memory(h.payload, h.id) for h in hits]
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return [m for m in memories if m.confidence >= min_confidence]
        except Exception as e:
            _db_log.warning(f"Exact search error: {e}")
            return []

    def _semantic_search(self, query: str, session_id: str, top_k: int,
                         min_confidence: float, cross_session: bool) -> list:
        try:
            vector = self._encoder.encode(query).tolist()
            filt = self._session_filter(session_id, cross_session)
            hits = self._client.search(
                collection_name=self.col_obj, query_vector=vector,
                query_filter=filt, limit=top_k, with_payload=True,
            )
            return [(self._to_memory(h.payload, str(h.id)), h.score)
                    for h in hits if h.payload.get("confidence", 0) >= min_confidence]
        except Exception as e:
            _db_log.warning(f"Semantic search error: {e}")
            return []

    def get_recent(self, limit: int = 20, cross_session: bool = False) -> list:
        try:
            filt = self._session_filter(self.session_id, cross_session)
            hits, _ = self._client.scroll(
                collection_name=self.col_obj, scroll_filter=filt,
                limit=100, with_payload=True, with_vectors=False,
            )
            memories = [self._to_memory(h.payload, str(h.id)) for h in hits]
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return memories[:limit]
        except Exception as e:
            _db_log.warning(f"get_recent error: {e}")
            return []

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    @staticmethod
    def _to_memory(payload: dict, point_id) -> object:
        # Import here keeps the circular-import window narrow but avoids
        # re-importing on every single call by relying on sys.modules cache.
        from models import SpatialMemory  # noqa: PLC0415
        return SpatialMemory(
            id=str(point_id),
            label=payload.get("label", "unknown"),
            confidence=payload.get("confidence", 0.0),
            original_confidence=payload.get("original_confidence",
                                             payload.get("confidence", 0.0)),
            angle_abs=payload.get("angle_abs", 0.0),
            distance_m=payload.get("distance_m", 0.0),
            frame_x_norm=payload.get("frame_x", 0.5),
            frame_y_norm=payload.get("frame_y", 0.5),
            timestamp=payload.get("timestamp", time.time()),
            session_id=payload.get("session_id", ""),
            user_id=payload.get("user_id", "default_user"),
            track_id=payload.get("track_id"),
            approach_velocity=payload.get("approach_velocity", 0.0),
            translation_x=payload.get("translation_x", 0.0),
            translation_y=payload.get("translation_y", 0.0),
            translation_z=payload.get("translation_z", 0.0),
            azimuth_deg=payload.get("azimuth_deg", 0.0),
            reid_embedding=payload.get("reid_embedding"),
            memory_half_life_hours=payload.get("memory_half_life_hours", 2.0),
        )


# ══════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_api_log = logging.getLogger("lumina.api")


class ConnectionManager:
    """Fan-out WebSocket broadcaster."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        """Thread-safe snapshot count (best-effort; no lock needed for int read)."""
        return len(self._connections)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        _api_log.info(f"Client connected — total: {len(self._connections)}")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._connections.discard(ws)
        _api_log.info(f"Client disconnected — total: {len(self._connections)}")

    async def broadcast(self, data: dict):
        if not self._connections:
            return
        payload = json.dumps(data)
        dead: Set[WebSocket] = set()
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._connections -= dead


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _api_log.info("Starting Lumina v4 Orchestrator…")
    from orchestrator import orchestrator as _orc
    # Cache on app.state so route handlers don't re-import each call
    app.state.orchestrator = _orc
    _orc.set_broadcaster(manager.broadcast)
    task = asyncio.create_task(_orc.start())
    yield
    _api_log.info("Shutting down…")
    await _orc.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Lumina Spatial AI",
    description="Multi-Agent Embodied AI for Visually Impaired Navigation — v4",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    orchestrator = app.state.orchestrator
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")
            if msg_type == "query":
                text = msg.get("text", "").strip()
                if text:
                    await orchestrator.query(text)
            elif msg_type == "set_open_vocab":
                # v4: allow frontend to set open-vocabulary detection targets
                classes = msg.get("classes", [])
                if classes:
                    orchestrator.set_open_vocab_targets(classes)
            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
            else:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                }))
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)


@app.get("/health")
async def health():
    orchestrator = app.state.orchestrator
    return JSONResponse({
        "status": "ok",
        "session_id": orchestrator.session_id,
        "version": "4.0.0",
        "connections": manager.count,
    })


@app.get("/memory")
async def get_memory():
    orchestrator = app.state.orchestrator
    if not orchestrator._librarian:
        return JSONResponse({"error": "Database not ready"}, status_code=503)
    snapshot = await orchestrator._librarian.get_memory_snapshot()
    return JSONResponse({"objects": snapshot, "session_id": orchestrator.session_id})


@app.get("/scene")
async def get_scene():
    orchestrator = app.state.orchestrator
    if not orchestrator._world_model:
        return JSONResponse({"error": "World model not ready"}, status_code=503)
    return JSONResponse(orchestrator._world_model.get_scene_summary())


@app.get("/status")
async def get_status():
    orchestrator = app.state.orchestrator
    llm_health = {}
    if orchestrator._llm:
        llm_health = await orchestrator._llm.health_check()
    return JSONResponse({
        "session_id": orchestrator.session_id,
        "camera": orchestrator._camera.is_open if orchestrator._camera else False,
        "qdrant": orchestrator._db.health_check() if orchestrator._db else False,
        "groq": llm_health.get("groq", False),
        "openai": llm_health.get("openai", False),
        "edge_model": llm_health.get("edge", False),         # v4
        "depth_engine": (orchestrator._depth_mono.available  # v4
                         if orchestrator._depth_mono else False),
        "model_active": orchestrator._llm.active_model if orchestrator._llm else "none",
        "tracker_tracks": len(orchestrator._tracker.tracks) if orchestrator._tracker else 0,
        "world_objects": (len(orchestrator._world_model._objects)
                          if orchestrator._world_model else 0),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
