# Lumina — Embodied Spatial AI for Assistive Navigation

> **A production-grade Multi-Agent System (MAS) that gives visually impaired users real-time spatial awareness, object memory, and voice-guided navigation — powered by computer vision, monocular depth estimation, and LLM-driven agent negotiation.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Event-Driven Pub/Sub Core](#event-driven-pubsub-core)
  - [Fast Loop vs. Slow Loop](#fast-loop-vs-slow-loop)
  - [Agent Roster](#agent-roster)
  - [Subscription Graph](#subscription-graph)
- [Key Technical Features](#key-technical-features)
  - [v4.1 Vision Upgrades](#v41-vision-upgrades)
  - [v5 Architecture Upgrades](#v5-architecture-upgrades)
- [System Components](#system-components)
- [Data Models](#data-models)
- [API & WebSocket Protocol](#api--websocket-protocol)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)

---

## Overview

Lumina is a real-time assistive navigation system designed for visually impaired users. It uses a camera feed (local webcam or IP camera) to continuously perceive the environment, build a persistent spatial memory, and respond to natural-language queries like _"Where is my phone?"_ or _"Find my bottle"_ with spoken, clock-direction navigation instructions.

The system is built around a true **Multi-Agent System (MAS)** architecture — six autonomous agents communicate exclusively through a central Pub/Sub event bus with no direct inter-agent coupling. This enables genuine agent autonomy, fault isolation, and emergent negotiation behaviour.

**Core capabilities:**

- Real-time object detection and multi-object tracking (YOLOv8 + IoU tracker)
- Monocular depth estimation with RANSAC multi-anchor metric calibration (MiDaS)
- 3D spatial back-projection (X, Y, Z camera-coordinate vectors)
- Persistent spatial memory with probabilistic confidence decay (Qdrant vector DB)
- Illumination-invariant visual Re-ID for cross-frame object deduplication
- Bird's-Eye View occupancy grid for safe lateral obstacle avoidance
- ORB-SLAM visual odometry compass (drift-free heading without IMU)
- LLM-driven query parsing and natural-language response generation
- LLM cascade: Groq → OpenAI → Local edge SLM (llama.cpp / Ollama) → deterministic fallback
- Real-time WebSocket streaming of annotated frames, agent logs, and navigation responses
- Cross-session persistent user memory

---

## Architecture

### Event-Driven Pub/Sub Core

Lumina v5 replaces the v4 procedural orchestrator with a fully async Pub/Sub architecture. All inter-agent communication goes through a single `EventBus` — agents never call each other directly.

```
                          ┌─────────────────────────────────────────┐
                          │             EventBus (Pub/Sub)           │
                          │                                          │
  Camera Frame ──────────►│ vision/new_frame                         │
                          │ hardware/emergency_stop  (high priority) │
  User Query  ──────────►│ system/query_received                    │
                          │ memory/candidates_ready                  │
                          │ memory/write_approved                    │
                          │ memory/search_result                     │
                          │ memory/confidence_low                    │
                          │ navigation/route_proposed                │
                          │ navigation/route_rejected                │
                          │ navigation/route_approved                │
                          │ navigation/route_final   ───────────────►│ WebSocket
                          │ system/agent_log         ───────────────►│ Clients
                          │ system/request_camera_pan                │
                          └─────────────────────────────────────────┘
```

**Why Pub/Sub?**
In v4, the orchestrator called agents `A → B → C → D` sequentially — a conductor pattern that is definitionally **not** a multi-agent system. v5 satisfies both MAS properties:

1. **Autonomy** — each agent decides _for itself_ when to act, based on what it perceives from the bus.
2. **Decentralisation** — no single component holds the global control flow at runtime.

The EventBus supports:
- Wildcard subscriptions (`navigation/*`)
- Two-tier priority dispatch (hardware reflexes bypass the queue)
- Per-topic event history ring buffer (last 20 events)
- Dead-letter logging for unsubscribed topics
- Thread-safe `publish_nowait()` for camera callback contexts

### Fast Loop vs. Slow Loop

```
FAST LOOP (30 FPS — ~33ms budget, NEVER awaits LLM)
┌─────────────────────────────────────────────────────────────────────┐
│  Camera capture → ORB-SLAM heading → YOLO detect → IoU track       │
│  → RANSAC depth calibration → 3D back-projection                    │
│  → BEV occupancy grid update → SafetyCortex                         │
│  → publish "vision/new_frame"      (triggers ArchivistAgent)        │
│  → publish "hardware/emergency_stop" if obstacle < 1m (HIGH PRIO)  │
└─────────────────────────────────────────────────────────────────────┘

SLOW LOOP (1–3 FPS equivalent — EventBus async dispatcher)
┌─────────────────────────────────────────────────────────────────────┐
│  ArchivistAgent    → packages SpatialMemory candidates              │
│  JanitorAgent      → deduplicates via Re-ID + spatial proximity     │
│  LibrarianAgent    → vector DB search + confidence decay            │
│  CoordinatorAgent  → computes 3D navigation + composes LLM response │
│  CriticAgent       → validates route, negotiates avoidance          │
│  AvoiderAgent      → builds detour speech from waypoints            │
└─────────────────────────────────────────────────────────────────────┘
```

The fast loop uses `asyncio.create_task()` for all bus publications, ensuring it **never blocks** on LLM latency or cognitive agent processing time.

### Agent Roster

| Agent | Subscribes To | Publishes To | Role |
|---|---|---|---|
| **ArchivistAgent** | `vision/new_frame` | `memory/candidates_ready` | Packages detected objects as `SpatialMemory` candidates with 3D vectors and Re-ID embeddings |
| **JanitorAgent** | `memory/candidates_ready` | `memory/write_approved` | Deduplicates candidates via Re-ID cosine distance, track-ID windowing, and spatial proximity fallback |
| **LibrarianAgent** | `system/query_received` | `memory/search_result`, `memory/confidence_low`, `system/request_camera_pan` | Exact + semantic Qdrant search; triggers active perception on low-confidence hits |
| **CoordinatorAgent** | `memory/search_result`, `navigation/route_rejected` | `navigation/route_proposed`, `navigation/route_final` | Parses queries (deterministic + LLM), computes 3D azimuth navigation, composes spoken responses |
| **CriticAgent** | `navigation/route_proposed` | `navigation/route_approved`, `navigation/route_rejected` | Validates confidence, staleness, and obstacle proximity; attaches avoidance waypoints instead of hard halts |
| **AvoiderAgent** | `hardware/safety_warning` | `navigation/route_final` | Computes grid-verified lateral strafe instructions; outputs spoken detour guidance |

### Subscription Graph

```
vision/new_frame            ──► ArchivistAgent.on_new_frame
memory/candidates_ready     ──► JanitorAgent.on_candidates_ready
system/query_received       ──► LibrarianAgent.on_query_received
memory/search_result        ──► CoordinatorAgent.on_search_result
navigation/route_proposed   ──► CriticAgent.on_route_proposed
navigation/route_rejected   ──► CoordinatorAgent.on_route_rejected
hardware/safety_warning     ──► AvoiderAgent.on_safety_warning
```

Agent negotiation is emergent: the Coordinator proposes a route, the Critic approves or rejects with a reason, and the Coordinator re-plans autonomously — up to 3 rounds — without the orchestrator's involvement.

---

## Key Technical Features

### v4.1 Vision Upgrades

#### Fix 1 — Multi-Anchor RANSAC Depth Calibration
Replaces single-object depth anchoring (which was corrupted by one wrong-height detection) with a RANSAC consensus across all confirmed tracks per frame. Scale is only updated when ≥3 anchors agree within a 15% tolerance window. The accepted scale is further smoothed by a 1D Kalman filter to suppress frame-to-frame jitter.

#### Fix 2 — Illumination-Invariant Re-ID (LAB + LBP + Spatial Pyramid)
Replaces raw HSV colour histograms with a 128-dimensional fused descriptor:
- **LAB chroma histogram (48-d):** bins only `a*` and `b*` channels, deliberately discarding `L*` (luminance) to eliminate sensitivity to lighting changes.
- **LBP texture descriptor (40-d):** Local Binary Patterns encode micro-texture structurally — invariant to colour and illumination.
- **3×3 spatial pyramid colour layout (40-d):** encodes _where_ colours appear in the object, separating objects with identical colour distributions but different spatial structure.

Final descriptor is L2-normalised. Same-object threshold: cosine distance < 0.20.

#### Fix 3 — Bird's-Eye Occupancy Grid (Dense Floor Map)
`BEVOccupancyGrid` back-projects every detected bounding box bottom edge into world-floor XZ space. Maintains a 2D grid (free / occupied / unknown). Before proposing any lateral strafe, the avoidance engine queries the grid for a clear corridor. Unknown cells are treated as unsafe (conservative / safe-fail). Free cells decay to unknown after 3 seconds for dynamic environments.

#### Fix 4 — ORB-SLAM Visual Compass (Drift-Free Heading)
Replaces cumulative optical-flow integration (unbounded drift) with a full Visual Odometry pipeline: ORB feature extraction → FLANN matching with Lowe ratio test → RANSAC Essential Matrix → `cv2.recoverPose()` for R and t → yaw extraction from the rotation matrix. Keyframe-based loop-closure detection soft-resets accumulated drift whenever the scene is revisited.

### v5 Architecture Upgrades

- **EventBus with two-tier dispatch** — hardware reflexes on `hardware/*` topics bypass the async queue and are dispatched immediately via `create_task()`. LLM-based agents cannot block the safety-critical path.
- **Agent registration pattern** — `agent.register()` attaches all subscriptions in one place. The subscription topology is fully visible and auditable in the orchestrator's `_register_agents()` method.
- **Active perception** — `LibrarianAgent` publishes `system/request_camera_pan` when effective memory confidence falls below 30%, signalling the system that it needs better sensory data. Agents are actors that shape their environment, not just passive processors.
- **Probabilistic memory permanence** — confidence decays exponentially with a 2-hour half-life rather than a hard 30-minute cliff. Memories never reach zero confidence; they become increasingly uncertain.
- **Orchestrator as bootstrap** — after `start()` returns, the orchestrator does nothing except feed camera frames to the bus at 30 FPS. All runtime behaviour is emergent.

---

## System Components

| Component | Class | Description |
|---|---|---|
| Camera Manager | `CameraManager` | Unified local/IP camera source with automatic reconnection and test pattern fallback |
| Visual SLAM Compass | `VisualSLAMCompass` | ORB-SLAM visual odometry with FLANN matching and loop-closure drift correction |
| YOLO Detector | `YOLODetector` | YOLOv8 COCO + optional YOLOWorld open-vocabulary detection |
| IoU Tracker | `IoUTracker` | Frame-to-frame bounding box association with Kalman-smoothed depth |
| Monocular Depth Engine | `MonocularDepthEngine` | MiDaS DPT-Small via ONNX Runtime or PyTorch; RANSAC multi-anchor metric calibration |
| Depth Fusion Engine | `DepthFusionEngine` | Per-track depth Kalman filter + 3D back-projection |
| Re-ID Extractor | `ReIDExtractor` | 128-d LAB+LBP+spatial pyramid descriptor extraction |
| BEV Occupancy Grid | `BEVOccupancyGrid` | 10cm-resolution 2D floor map with free-cell decay |
| Dynamic Avoidance Engine | `DynamicAvoidanceEngine` | Grid-verified lateral strafe waypoint computation |
| Safety Cortex | `SafetyCortex` | Multi-level danger alerting (critical / warning / caution) with avoidance integration |
| World Model | `WorldModel` | Live scene graph with temporal state transitions and event log |
| Spatial Database | `SpatialDatabase` | Qdrant-backed memory with exact + semantic search, probabilistic confidence decay |
| LLM Client | `LLMClient` | Groq → OpenAI → Edge SLM → deterministic cascade, never raises |
| Edge LLM Backend | `EdgeLLMBackend` | llama.cpp (GGUF) and Ollama local inference backends |
| Event Bus | `EventBus` | Fully async Pub/Sub broker with wildcard subscriptions, priority dispatch, dead-letter logging |

---

## Data Models

### Core Spatial Types

```python
# 3D object position in camera space
TrackedDetection:
    translation_x: float    # metres right of camera axis
    translation_y: float    # metres below camera axis
    translation_z: float    # metres forward (depth)
    azimuth_deg: float      # θ = arctan2(X, Z) in degrees
    reid_embedding: List[float]  # 128-d Re-ID descriptor

# Persistent spatial memory entry (stored in Qdrant)
SpatialMemory:
    label, confidence, original_confidence
    angle_abs, distance_m
    translation_x, translation_y, translation_z, azimuth_deg
    reid_embedding
    memory_half_life_hours  # per-object decay rate
    timestamp, session_id, user_id

# Navigation output
SpatialResult:
    clock_direction: str       # e.g. "2 o'clock"
    turn_instruction: str      # e.g. "Turn right"
    distance_str: str          # e.g. "1.4 metres"
    confidence: float          # probabilistically decayed
    is_stale: bool
    stale_message: str

# Dynamic obstacle detour
AvoidanceWaypoint:
    strafe_direction: "left" | "right"
    strafe_distance_m: float
    forward_clearance_m: float
    clock_instruction: str
```

### Probabilistic Memory Permanence

```python
def probabilistic_confidence(original_confidence, age_seconds, half_life_hours=2.0):
    """
    Exponential decay: at age=0 → original_confidence
                       at age=half_life → original_confidence * 0.5
                       at age=8h → ~original_confidence * 0.06 (never zero)
    """
    decayed = original_confidence * exp(-0.693 * age_seconds / (half_life_hours * 3600))
    return max(0.05, decayed)
```

---

## API & WebSocket Protocol

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System health, session ID, connection count |
| `GET` | `/status` | LLM provider health, camera, Qdrant, depth engine, tracker stats |
| `GET` | `/memory` | Recent spatial memory snapshot with effective confidence |
| `GET` | `/scene` | Live world model: active tracks, recent state-change events |

### WebSocket — `/ws`

**Client → Server messages:**

```jsonc
// Natural-language object query
{ "type": "query", "text": "where is my phone?" }

// Set open-vocabulary detection targets (YOLOWorld)
{ "type": "set_open_vocab", "classes": ["coffee mug", "charger"] }

// Keepalive
{ "type": "ping" }
```

**Server → Client message types:**

| Type | Description |
|---|---|
| `frame` | Annotated JPEG (base64) + detection list + compass heading + bus stats |
| `response` | Final navigation text, clock direction, distance, confidence, avoidance vector |
| `safety_alert` | Level (critical/warning/caution), label, distance, clock direction |
| `agent_log` | Per-agent log emission with level and metadata |
| `memory_update` | Updated memory snapshot after new writes |
| `world_update` | Active object list + recent state-change events |
| `system_status` | All component health flags + MAS topology info |
| `camera_pan_request` | Active perception signal: agent requesting camera repositioning |
| `avoidance` | Detour instruction: strafe direction, distance, obstacle info |

---

## Configuration

All settings are loaded from a `.env` file via `pydantic-settings`. Key variables:

```env
# LLM — Cloud (cascade order: Groq → OpenAI)
GROQ_API_KEY=
OPENAI_API_KEY=
GROQ_MODEL=llama3-70b-8192
OPENAI_MODEL=gpt-4o

# LLM — Local Edge (optional; llama_cpp or ollama)
EDGE_LLM_BACKEND=none          # "llama_cpp" | "ollama" | "none"
EDGE_LLM_MODEL_PATH=           # path to .gguf file
EDGE_LLM_MODEL_NAME=           # e.g. "phi3:mini" for ollama

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Vision
YOLO_MODEL=yolov8n.pt
DETECTION_CONFIDENCE=0.50
VISION_FPS=8

# Camera Source
CAMERA_MODE=local              # "local" | "ip"
CAMERA_INDEX=0
CAMERA_IP_URL=                 # e.g. http://192.168.1.5:8080/video

# Depth Engine
DEPTH_ENGINE_ENABLED=true
DEPTH_ONNX_MODEL_PATH=         # optional: path to midas_v21_small_256.onnx

# Spatial Memory
MEMORY_DECAY_HALF_LIFE_HOURS=2.0
CRITIC_CONFIDENCE_THRESHOLD=0.60
CROSS_SESSION_ENABLED=true
USER_ID=default_user

# Safety Distances (metres)
SAFETY_CRITICAL_DIST=0.8
SAFETY_WARNING_DIST=1.5
SAFETY_CAUTION_DIST=2.5
```

### IP Camera Setup

Lumina supports phone cameras as the input source via any MJPEG or RTSP app on the same Wi-Fi network:

| App | Platform | URL Format |
|---|---|---|
| IP Webcam | Android | `http://<phone-ip>:8080/video` |
| DroidCam | Android / iOS | `http://<phone-ip>:4747/video` |
| iVCam / EpocCam | iOS | `rtsp://<phone-ip>:8554/live` |

Set `CAMERA_MODE=ip` and `CAMERA_IP_URL=<url>` in `.env`. The stream reconnects automatically on drop-out.

---

## Installation

### Prerequisites

- Python 3.10+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally (Docker recommended)
- (Optional) CUDA-capable GPU for faster depth inference

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd lumina

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install fastapi uvicorn[standard] pydantic pydantic-settings \
    opencv-python-headless ultralytics qdrant-client \
    sentence-transformers groq openai

# Optional: monocular depth (PyTorch path)
pip install torch torchvision

# Optional: ONNX depth engine (lighter, faster)
pip install onnxruntime   # or onnxruntime-gpu

# Optional: local edge LLM
pip install llama-cpp-python   # for llama_cpp backend
# OR: install Ollama from https://ollama.ai

# 4. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 5. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 6. (Optional) Download MiDaS ONNX model for offline depth
# Place midas_v21_small_256.onnx at the path set in DEPTH_ONNX_MODEL_PATH
```

---

## Running the System

```bash
# Development
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
python main.py
```

The WebSocket endpoint is available at `ws://localhost:8000/ws`.

Connect a frontend or use a WebSocket client to send queries and receive real-time navigation responses.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | FastAPI + Uvicorn |
| **Computer Vision** | OpenCV, YOLOv8 (Ultralytics), MiDaS (Intel) |
| **Object Tracking** | Custom IoU Tracker + Kalman depth filter |
| **Visual Odometry** | ORB-SLAM pipeline via OpenCV |
| **Vector Database** | Qdrant with sentence-transformers embeddings |
| **LLM (Cloud)** | Groq (llama3-70b) + OpenAI (GPT-4o) |
| **LLM (Edge)** | llama.cpp (GGUF) / Ollama |
| **Data Validation** | Pydantic v2 |
| **Async Runtime** | Python asyncio |
| **Depth Estimation** | MiDaS DPT-Small via ONNX Runtime or PyTorch |

---

## Project Structure

```
lumina/
├── main.py            # FastAPI app, LLM client, Settings, SpatialDatabase
├── orchestrator.py    # Bootstrap orchestrator — wires components and starts loops
├── agents.py          # Six autonomous EventBus-native agents + WorldModel
├── event_bus.py       # Async Pub/Sub broker + all payload type definitions
├── vision.py          # Camera, YOLO, tracker, depth, Re-ID, BEV grid, SLAM compass
├── models.py          # All Pydantic/dataclass data models, spatial math utilities
└── .env               # Environment configuration (not committed)
```

---

## Design Principles

**Safety-first dispatch** — emergency stop events are published on a high-priority path that bypasses the async queue. A slow LLM call can never delay a `STOP` command.

**Fault isolation** — every agent handler is wrapped in `_safe_dispatch()`. An exception in one agent does not crash or stall the others.

**Deterministic fallbacks** — the LLM cascade always terminates in a deterministic response. The system never raises `RuntimeError` due to LLM unavailability.

**Probabilistic confidence over binary thresholds** — memories degrade gracefully through exponential decay rather than hard expiry cliffs. Stale memories produce uncertain guidance, not silence.

**Active perception** — agents request better sensory data rather than degrading passively. Low-confidence memory retrieval triggers a camera pan signal to improve the observation.
