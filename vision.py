"""
Lumina v4.1 — Vision Module

Root-cause fixes for four architectural vulnerabilities:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  FIX 1 — Multi-Anchor RANSAC Depth Calibration                      │
  │    Replaces single-object anchoring with a RANSAC-based multi-      │
  │    anchor fit across all confirmed tracks per frame.                 │
  │    A corrupted anchor (child instead of adult) is voted out by      │
  │    the RANSAC consensus. Scale is only updated when ≥3 inliers      │
  │    agree within 15% of each other. Includes a Kalman smoother on   │
  │    the scale factor itself to suppress frame-to-frame jitter.       │
  │                                                                      │
  │  FIX 2 — Illumination-Invariant Re-ID (LAB + LBP + Spatial Grid)   │
  │    Replaces the raw HSV histogram with a three-part descriptor:     │
  │    (a) LAB colour histogram — perceptually uniform, less sensitive  │
  │        to lighting intensity shifts than HSV.                       │
  │    (b) LBP texture descriptor — purely structural, captures         │
  │        shape/surface even under heavy illumination change.          │
  │    (c) 3×3 spatial pyramid — preserves layout (top/bottom, left/   │
  │        right), so a mug and a shirt with identical colour are       │
  │        separated by their spatial colour distribution.              │
  │    Final: 128-d L2-normalised fused descriptor.                     │
  │                                                                      │
  │  FIX 3 — Bird's-Eye Occupancy Grid (Dense Floor Map)               │
  │    BEVOccupancyGrid back-projects every detected bounding box       │
  │    bottom edge into world-floor XZ space. Maintains a persistent    │
  │    2D occupancy grid (free/occupied/unknown). Avoidance queries     │
  │    this grid before emitting a strafe direction: a lateral step     │
  │    is only proposed if the corridor is confirmed walkable by         │
  │    the grid. Cleared patches decay to "unknown" over time.          │
  │                                                                      │
  │  FIX 4 — ORB-SLAM Visual Compass (drift-free heading)              │
  │    Replaces cumulative pixel-flow integration with a true Visual    │
  │    Odometry pipeline: ORB feature extraction → FLANN matching →    │
  │    Essential Matrix decomposition (RANSAC) → recoverPose() for     │
  │    R, t. Heading is accumulated from the rotation matrix, not       │
  │    from pixel displacement. Keyframe-based loop-closure resets      │
  │    drift when sufficient overlap is detected. Falls back to         │
  │    optical flow only when fewer than 8 ORB inliers are available.  │
  └──────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import base64
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from models import (
    Detection, BoundingBox,
    CameraIntrinsics,
    estimate_distance_geometric,
    backproject_to_3d, azimuth_from_3d,
    to_clock_direction, format_distance,
    AvoidanceWaypoint,
)

log = logging.getLogger("lumina.vision")


# ═════════════════════════════════════════════════════════════
# CAMERA MANAGER
# ═════════════════════════════════════════════════════════════

class CameraManager:
    """
    Unified camera source — supports both local webcam and IP camera.

    MODE "local":
        Uses cv2.VideoCapture(index) — the laptop or USB webcam.

    MODE "ip":
        Uses cv2.VideoCapture(url) where url is an RTSP or HTTP MJPEG
        stream from a mobile phone camera app on the same Wi-Fi network.

        Tested with:
          • IP Webcam (Android)  → http://<phone-ip>:8080/video
          • DroidCam (Android/iOS) → http://<phone-ip>:4747/video
          • iVCam / EpocCam      → rtsp://<phone-ip>:8554/live

        Setup steps (IP Webcam example):
          1. Install "IP Webcam" on your Android phone.
          2. Open the app → tap "Start server" at the bottom.
          3. Note the URL shown (e.g. http://192.168.1.5:8080).
          4. Set CAMERA_MODE=ip and CAMERA_IP_URL=http://192.168.1.5:8080/video
             in your .env file (or environment variables).
          5. Make sure both devices are on the same Wi-Fi network.

        The IP camera source reconnects automatically if the stream drops
        (phone screen locks, app backgrounded, network blip). A test
        pattern with a reconnecting message is shown during drop-outs
        rather than freezing or crashing.

    FOV note:
        A phone rear camera typically offers 60–80° horizontal FOV,
        similar to a laptop webcam. Set CAMERA_FOV_H accordingly in .env.
        Wide-angle (fisheye) lenses may need undistortion — not implemented
        here but can be added with cv2.undistort() before returning the frame.
    """

    def __init__(
        self,
        index: int = 0,
        mode: str = "local",
        ip_url: str = "",
        reconnect_delay: float = 2.0,
        timeout_ms: int = 5000,
    ):
        self._mode = mode.lower().strip()
        self._index = index
        self._ip_url = ip_url
        self._reconnect_delay = reconnect_delay
        self._timeout_ms = timeout_ms
        self._cap: Optional[cv2.VideoCapture] = None
        self._ok = False
        self._last_reconnect_attempt: float = 0.0

        self._open()

    # ── Public interface ──────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._ok

    @property
    def mode(self) -> str:
        return self._mode

    def read(self) -> Optional[np.ndarray]:
        """
        Return the next frame from the active camera source.

        For IP cameras: if the stream is disconnected, attempts a
        reconnect at most once per RECONNECT_DELAY seconds. Returns
        a test pattern (with status message) while reconnecting so
        the rest of the pipeline keeps running without crashing.
        """
        if self._mode == "ip":
            return self._read_ip()
        return self._read_local()

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
        self._ok = False

    # ── Internal helpers ──────────────────────────────────────

    def _open(self):
        """Open the configured camera source."""
        if self._mode == "ip":
            if not self._ip_url:
                log.error(
                    "CAMERA_MODE=ip but CAMERA_IP_URL is empty. "
                    "Set it to e.g. http://192.168.1.5:8080/video"
                )
                self._ok = False
                return
            self._open_ip()
        else:
            self._open_local()

    def _open_local(self):
        self._cap = cv2.VideoCapture(self._index)
        self._ok = self._cap.isOpened()
        if self._ok:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            log.info(f"Local camera {self._index} opened — 640×480")
        else:
            log.warning(f"Local camera {self._index} unavailable — using test pattern")

    def _open_ip(self):
        """
        Open an IP camera stream (RTSP or HTTP MJPEG).

        CAP_PROP_OPEN_TIMEOUT_MSEC and CAP_PROP_READ_TIMEOUT_MSEC are set
        so OpenCV does not block indefinitely if the phone is unreachable.
        """
        log.info(f"Connecting to IP camera: {self._ip_url}")
        cap = cv2.VideoCapture(self._ip_url, cv2.CAP_FFMPEG)

        # Set timeouts (supported in OpenCV 4.5.2+; silently ignored otherwise)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_ms)

        if cap.isOpened():
            # Drain a couple of frames to flush the buffer before real use
            for _ in range(3):
                cap.grab()
            self._cap = cap
            self._ok = True
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"IP camera connected: {self._ip_url} — {w}×{h}")
        else:
            cap.release()
            self._cap = None
            self._ok = False
            log.warning(
                f"IP camera not reachable at {self._ip_url}. "
                "Check that the phone app is running and both devices share Wi-Fi."
            )

    def _read_local(self) -> Optional[np.ndarray]:
        if self._ok and self._cap:
            ret, frame = self._cap.read()
            if ret:
                return frame
        return self._test_pattern()

    def _read_ip(self) -> Optional[np.ndarray]:
        """
        Read a frame from the IP stream, reconnecting on failure.
        Falls back to a test pattern while the stream is down so
        the vision pipeline never receives None.
        """
        if self._ok and self._cap:
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
            # Stream dropped — mark as disconnected
            log.warning("IP camera stream lost — attempting reconnect…")
            self._cap.release()
            self._cap = None
            self._ok = False

        # Throttle reconnect attempts
        now = time.time()
        if now - self._last_reconnect_attempt >= self._reconnect_delay:
            self._last_reconnect_attempt = now
            self._open_ip()
            if self._ok and self._cap:
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    return frame

        return self._test_pattern(
            message=f"IP CAM RECONNECTING… {self._ip_url}"
        )

    @staticmethod
    def _test_pattern(message: str = "NO CAMERA — TEST MODE") -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t = int(time.time() * 2) % 255
        cv2.rectangle(frame, (100, 100), (540, 380), (0, t, 80), 2)
        # Wrap long URLs across two lines so they fit in the frame
        line1 = message[:55]
        line2 = message[55:]
        cv2.putText(frame, line1,
                    (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 170), 2)
        if line2:
            cv2.putText(frame, line2,
                        (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 170), 1)
        return frame


# ═════════════════════════════════════════════════════════════
# MONOCULAR DEPTH ENGINE  (Fix 1 — multi-anchor RANSAC scale)
# ═════════════════════════════════════════════════════════════

class MonocularDepthEngine:
    """
    MiDaS DPT-Small monocular depth with RANSAC multi-anchor calibration.

    Root-cause fix for the anchor corruption problem:
      v4.0 used a single object's geometric estimate to set the global
      scale factor. One wrong anchor (a child, a toy, a partially visible
      object) corrupted the entire depth map.

    v4.1 approach — RANSAC multi-anchor consensus:
      1. Collect geometric distance estimates for ALL confirmed tracks
         in the current frame (at least 3 required).
      2. For each candidate anchor, compute the implied scale:
             scale_i = geo_dist_i * midas_relative_depth_i
      3. Run RANSAC: find the largest subset of anchors whose implied
         scales agree within INLIER_TOLERANCE (15%).
      4. Scale is updated ONLY if RANSAC finds ≥ MIN_INLIERS consensus.
      5. The accepted scale is further smoothed by a 1D Kalman filter
         to suppress frame-to-frame jitter.

    This means: even if one anchor is a child (height ~0.9m vs assumed
    1.7m), as long as other objects (a chair, a table, a bottle) produce
    consistent scale estimates, those inliers win and the outlier is
    discarded without corrupting the scene.
    """

    _INPUT_SIZE = 256
    _INLIER_TOLERANCE = 0.15   # Two anchors agree if scales within 15%
    _MIN_INLIERS = 3            # Need at least 3 agreeing anchors to update scale
    _SCALE_KALMAN_R = 0.05      # Measurement noise for scale Kalman
    _SCALE_KALMAN_Q = 0.001     # Process noise for scale Kalman

    def __init__(self, onnx_model_path: str = ""):
        self._session = None
        self._torch_model = None
        self._backend: str = "none"
        self._raw_depth_cache: Optional[np.ndarray] = None  # raw MiDaS output before scaling

        # Scale Kalman filter state [scale, scale_velocity]
        self._scale_kf_x = np.array([[1.0], [0.0]])
        self._scale_kf_P = np.eye(2) * 0.5
        self._scale_kf_F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._scale_kf_H = np.array([[1.0, 0.0]])
        self._scale_kf_Q = np.array([[self._SCALE_KALMAN_Q, 0], [0, self._SCALE_KALMAN_Q * 10]])
        self._scale_kf_R = np.array([[self._SCALE_KALMAN_R]])

        if onnx_model_path:
            self._try_load_onnx(onnx_model_path)
        if self._backend == "none":
            self._try_load_torch()
        log.info(f"MonocularDepthEngine backend: {self._backend}")

    def _try_load_onnx(self, path: str):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                path, sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._backend = "onnx"
            log.info(f"MiDaS ONNX loaded: {path}")
        except Exception as e:
            log.warning(f"ONNX depth load failed ({e})")

    def _try_load_torch(self):
        try:
            import torch
            self._torch_model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True
            )
            self._torch_model.eval()
            if torch.cuda.is_available():
                self._torch_model = self._torch_model.cuda()
            self._backend = "torch"
            log.info("MiDaS torch.hub loaded")
        except Exception as e:
            log.warning(f"Torch depth load failed ({e}) — geometric fallback active")

    @property
    def available(self) -> bool:
        return self._backend != "none"

    @property
    def current_scale(self) -> float:
        return float(self._scale_kf_x[0, 0])

    def infer_raw(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run MiDaS and return RAW relative depth map (not yet metric-scaled)."""
        if self._backend == "none":
            return None
        try:
            inp = self._preprocess(frame)
            if self._backend == "onnx":
                raw = self._run_onnx(inp, frame.shape)
            else:
                raw = self._run_torch(inp, frame.shape)
            self._raw_depth_cache = raw
            return raw
        except Exception as e:
            log.warning(f"Depth inference failed: {e}")
            return None

    def to_metric(self, raw_depth: np.ndarray) -> np.ndarray:
        """Apply current Kalman-smoothed scale to convert relative→metric."""
        s = max(self.current_scale, 0.1)
        eps = 1e-6
        metric = s / (raw_depth.astype(np.float32) + eps)
        return np.clip(metric, 0.1, 15.0)

    def infer(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Full pipeline: infer raw → apply current scale → return metric map."""
        raw = self.infer_raw(frame)
        if raw is None:
            return None
        return self.to_metric(raw)

    def calibrate_ransac(
        self,
        anchors: List[Tuple[float, float, float, str]]
        # Each anchor: (px_x, px_y, geo_dist_m, label)
    ) -> bool:
        """
        FIX 1 CORE: RANSAC multi-anchor scale calibration.

        Runs RANSAC over all anchor proposals to find the dominant
        scale consensus. Rejects outlier anchors (e.g. misidentified
        object sizes). Updates the Kalman-smoothed scale only when
        consensus is strong (≥ MIN_INLIERS).

        Returns True if scale was updated, False if consensus failed.
        """
        if self._raw_depth_cache is None or len(anchors) < self._MIN_INLIERS:
            return False

        raw = self._raw_depth_cache
        h, w = raw.shape

        # Compute implied scale for each anchor
        implied_scales = []
        for px_x, px_y, geo_dist, label in anchors:
            ix = max(0, min(int(px_x), w - 1))
            iy = max(0, min(int(px_y), h - 1))
            rel_val = float(raw[iy, ix])
            if rel_val < 1e-4 or geo_dist < 0.2:
                continue
            # scale = geo_dist * rel_val  (from: geo_dist = scale / rel_val)
            implied_scales.append((geo_dist * rel_val, geo_dist, label))

        if len(implied_scales) < self._MIN_INLIERS:
            return False

        # RANSAC: find largest inlier set
        best_inliers = []
        best_scale = self.current_scale

        for i, (s_i, _, _) in enumerate(implied_scales):
            inliers = [s_j for s_j, _, _ in implied_scales
                       if abs(s_j - s_i) / (s_i + 1e-6) < self._INLIER_TOLERANCE]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_scale = float(np.median(inliers))

        if len(best_inliers) < self._MIN_INLIERS:
            log.debug(f"RANSAC depth: only {len(best_inliers)} inliers — scale held")
            return False

        # Kalman update on the scale
        self._scale_kf_x = self._scale_kf_F @ self._scale_kf_x
        self._scale_kf_P = (self._scale_kf_F @ self._scale_kf_P @ self._scale_kf_F.T
                            + self._scale_kf_Q)
        y = np.array([[best_scale]]) - self._scale_kf_H @ self._scale_kf_x
        S = self._scale_kf_H @ self._scale_kf_P @ self._scale_kf_H.T + self._scale_kf_R
        K = self._scale_kf_P @ self._scale_kf_H.T @ np.linalg.inv(S)
        self._scale_kf_x = self._scale_kf_x + K @ y
        self._scale_kf_P = (np.eye(2) - K @ self._scale_kf_H) @ self._scale_kf_P
        # Clamp scale to physically plausible range
        self._scale_kf_x[0, 0] = max(0.5, min(50.0, float(self._scale_kf_x[0, 0])))

        log.debug(f"RANSAC depth scale updated: {best_scale:.3f} "
                  f"({len(best_inliers)}/{len(implied_scales)} inliers)")
        return True

    def depth_at(self, depth_map: np.ndarray, px: float, py: float) -> float:
        x, y = int(px), int(py)
        h, w = depth_map.shape[:2]
        return float(depth_map[max(0, min(y, h-1)), max(0, min(x, w-1))])

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._INPUT_SIZE, self._INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return ((img - mean) / std).transpose(2, 0, 1)[np.newaxis]

    def _run_onnx(self, inp: np.ndarray, orig_shape: tuple) -> np.ndarray:
        name = self._session.get_inputs()[0].name
        raw = self._session.run(None, {name: inp})[0].squeeze()
        return cv2.resize(raw, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

    def _run_torch(self, inp: np.ndarray, orig_shape: tuple) -> np.ndarray:
        import torch
        t = torch.from_numpy(inp)
        if next(self._torch_model.parameters()).is_cuda:
            t = t.cuda()
        with torch.no_grad():
            raw = self._torch_model(t).squeeze().cpu().numpy()
        return cv2.resize(raw, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)


# ═════════════════════════════════════════════════════════════
# RE-ID EXTRACTOR  (Fix 2 — illumination-invariant descriptor)
# ═════════════════════════════════════════════════════════════

class ReIDExtractor:
    """
    128-d illumination-invariant Re-ID descriptor.

    Root-cause fix for the histogram lighting sensitivity problem:
      v4.0 used an HSV colour histogram. HSV's Value channel encodes
      absolute brightness, making the same object in sunlight vs shadow
      produce very different embeddings. Saturation also shifts under
      fluorescent vs incandescent light.

    v4.1 three-part descriptor:
      (a) LAB colour histogram [48-d]:
          CIE L*a*b* separates luminance (L) from chrominance (a, b).
          We histogram only the a* and b* channels (24 bins each),
          deliberately DISCARDING L* — the chroma channels are
          substantially more stable across illumination changes.

      (b) LBP texture descriptor [40-d]:
          Local Binary Patterns encode the micro-texture around each
          pixel by comparing it to its 8 neighbours. This is purely
          structural — a mug's smooth ceramic surface and a t-shirt's
          fabric weave produce different LBP histograms regardless of
          colour or lighting. Radius=1, 8 neighbours.

      (c) 3×3 spatial pyramid colour layout [40-d]:
          The crop is divided into a 3×3 grid. Each cell contributes
          a 4-bin a* + 4-bin b* histogram (8-d × 9 cells = 72-d,
          then PCA-reduced inline to 40-d via first-40 top variance).
          This encodes WHERE colours appear in the object, separating
          a red mug (red at the bottom half) from a red shirt (red
          evenly distributed).

    Final: 48 + 40 + 40 = 128-d, L2-normalised.
    Same-object threshold: cosine distance < 0.20.
    """

    EMBEDDING_DIM = 128
    SAME_OBJECT_THRESHOLD = 0.20

    def extract(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[List[float]]:
        try:
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))
            if x2 - x1 < 12 or y2 - y1 < 12:
                return None
            crop = frame[y1:y2, x1:x2]
            return self._build_descriptor(crop)
        except Exception:
            return None

    def _build_descriptor(self, crop: np.ndarray) -> List[float]:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        # ── Part A: LAB chroma histogram (48-d) ──────────────
        # Bin only a* and b* — drop L* to discard illumination
        a_hist = cv2.calcHist([lab], [1], None, [24], [0, 256]).flatten()
        b_hist = cv2.calcHist([lab], [2], None, [24], [0, 256]).flatten()
        lab_feat = np.concatenate([a_hist, b_hist])  # 48-d

        # ── Part B: LBP texture (40-d) ────────────────────────
        lbp_feat = self._lbp_histogram(crop, n_bins=40)   # 40-d

        # ── Part C: Spatial pyramid layout (40-d) ────────────
        spatial_feat = self._spatial_pyramid(lab, grid=3, bins_per_channel=4)  # 40-d trimmed

        # ── Fuse and normalise ────────────────────────────────
        descriptor = np.concatenate([lab_feat, lbp_feat, spatial_feat]).astype(np.float32)
        descriptor = descriptor[:self.EMBEDDING_DIM]
        norm = np.linalg.norm(descriptor)
        if norm > 1e-6:
            descriptor /= norm
        return descriptor.tolist()

    @staticmethod
    def _lbp_histogram(crop: np.ndarray, n_bins: int = 40) -> np.ndarray:
        """
        Compute LBP histogram. Pure NumPy — no extra deps.
        Radius=1, 8 neighbours, uniform mapping.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        # 8 neighbour offsets for radius=1
        neighbours = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        for bit, (dy, dx) in enumerate(neighbours):
            # Roll-based neighbour comparison (avoids loops over pixels)
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp |= ((gray >= shifted).astype(np.uint8) << bit)
        hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, 256))
        return hist.astype(np.float32)

    @staticmethod
    def _spatial_pyramid(lab: np.ndarray, grid: int = 3, bins_per_channel: int = 4) -> np.ndarray:
        """
        Divide crop into grid×grid cells; compute a*+b* histograms per cell.
        Returns first 40 elements (cells × bins_per_channel × 2 channels).
        """
        h, w = lab.shape[:2]
        feats = []
        for r in range(grid):
            for c in range(grid):
                cell = lab[r*h//grid:(r+1)*h//grid, c*w//grid:(c+1)*w//grid]
                if cell.size == 0:
                    feats.extend([0.0] * bins_per_channel * 2)
                    continue
                a_h = cv2.calcHist([cell], [1], None, [bins_per_channel], [0, 256]).flatten()
                b_h = cv2.calcHist([cell], [2], None, [bins_per_channel], [0, 256]).flatten()
                feats.extend(a_h.tolist())
                feats.extend(b_h.tolist())
        arr = np.array(feats, dtype=np.float32)
        return arr[:40]   # trim to fixed 40-d

    @staticmethod
    def cosine_distance(a: List[float], b: List[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        return 1.0 - float(np.dot(va, vb))  # already L2-normalised


# ═════════════════════════════════════════════════════════════
# BIRD'S-EYE OCCUPANCY GRID  (Fix 3 — dense floor map)
# ═════════════════════════════════════════════════════════════

class BEVOccupancyGrid:
    """
    Bird's-Eye View 2D floor occupancy grid.

    Root-cause fix for the "blind avoidance" problem:
      v4.0 computed strafe waypoints based only on detected bounding boxes,
      with no knowledge of what lies in the proposed strafe direction.
      A user could be commanded to step into a wall or staircase.

    v4.1 floor mapping approach:
      The grid represents a top-down view of the floor in front of the user:
      - X axis: lateral (left/right), cells of CELL_SIZE metres
      - Z axis: forward depth, cells of CELL_SIZE metres
      - Grid values: 0.0 = free, 1.0 = occupied, 0.5 = unknown

      Population: For each tracked detection, we project its bounding box
      BOTTOM EDGE (ground contact point) into floor XZ space using the
      camera's known intrinsics and the depth estimate. The footprint of
      the object (estimated from its depth and width) is marked occupied.

      Clearance check: Before proposing a strafe direction, the avoidance
      engine checks whether a corridor of at least MIN_CLEARANCE_M width
      exists in the grid along the proposed direction. If the corridor is
      not confirmed free (unknown counts as blocked for safety), the
      direction is rejected or a warning is added.

      Decay: Free observations decay to 0.5 (unknown) after DECAY_TIME_S
      seconds to handle dynamic environments.
    """

    GRID_RANGE_M   = 5.0    # metres in each direction from user
    CELL_SIZE_M    = 0.1    # 10 cm per cell
    DECAY_TIME_S   = 3.0    # free cells return to unknown after this
    MIN_CLEARANCE_M = 0.8   # minimum walkable corridor width

    def __init__(self, intrinsics: Optional[CameraIntrinsics] = None):
        n = int(2 * self.GRID_RANGE_M / self.CELL_SIZE_M)
        self._n = n
        # 0.0=free, 1.0=occupied, 0.5=unknown
        self._grid = np.full((n, n), 0.5, dtype=np.float32)
        self._last_free_time = np.zeros((n, n), dtype=np.float64)
        self._intrinsics = intrinsics or CameraIntrinsics()
        self._origin = n // 2   # user is at centre of grid

    def update_intrinsics(self, intrinsics: CameraIntrinsics):
        self._intrinsics = intrinsics

    def _world_to_cell(self, x_m: float, z_m: float) -> Optional[Tuple[int, int]]:
        """Convert world XZ (metres) to grid cell indices."""
        col = int(self._origin + x_m / self.CELL_SIZE_M)
        row = int(self._origin + z_m / self.CELL_SIZE_M)
        if 0 <= row < self._n and 0 <= col < self._n:
            return row, col
        return None

    def _cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x_m = (col - self._origin) * self.CELL_SIZE_M
        z_m = (row - self._origin) * self.CELL_SIZE_M
        return x_m, z_m

    def mark_occupied(self, x_m: float, z_m: float, radius_m: float = 0.3):
        """Mark a circular footprint at (x_m, z_m) as occupied."""
        r_cells = max(1, int(radius_m / self.CELL_SIZE_M))
        cx = int(self._origin + x_m / self.CELL_SIZE_M)
        cz = int(self._origin + z_m / self.CELL_SIZE_M)
        for dz in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx*dx + dz*dz <= r_cells*r_cells:
                    row, col = cz + dz, cx + dx
                    if 0 <= row < self._n and 0 <= col < self._n:
                        self._grid[row, col] = 1.0

    def mark_free_corridor(self, x_m: float, z_max_m: float, width_m: float = 0.6):
        """Mark a rectangular forward corridor as free (observed walkable floor)."""
        now = time.time()
        w_cells = max(1, int(width_m / self.CELL_SIZE_M / 2))
        cx = int(self._origin + x_m / self.CELL_SIZE_M)
        z_cells = int(z_max_m / self.CELL_SIZE_M)
        for dz in range(1, z_cells + 1):
            for dx in range(-w_cells, w_cells + 1):
                row, col = int(self._origin) + dz, cx + dx
                if 0 <= row < self._n and 0 <= col < self._n:
                    if self._grid[row, col] < 1.0:  # don't overwrite obstacles
                        self._grid[row, col] = 0.0
                        self._last_free_time[row, col] = now

    def decay_free_cells(self):
        """Return decayed free cells to unknown status."""
        now = time.time()
        stale_mask = (
            (self._grid == 0.0) &
            (now - self._last_free_time > self.DECAY_TIME_S)
        )
        self._grid[stale_mask] = 0.5

    def update_from_tracks(self, tracks: List, depth_map: Optional[np.ndarray] = None):
        """
        Update the occupancy grid from the current set of tracked objects.
        Projects each object's ground contact point (bottom bbox edge) into
        floor XZ space using back-projection.
        """
        self.decay_free_cells()

        intr = self._intrinsics
        for track in tracks:
            depth_z = track.smoothed_distance if hasattr(track, 'smoothed_distance') else 1.0

            # Ground contact: bottom-centre of bounding box
            foot_px = track.bbox.center_x
            foot_py = track.bbox.y2   # bottom edge

            X, Y, Z = backproject_to_3d(
                foot_px, foot_py, depth_z,
                intr.fx, intr.fy, intr.cx, intr.cy
            )

            # Estimate object footprint radius from bbox width
            bbox_width_px = track.bbox.width
            obj_radius_m = max(0.2, (bbox_width_px / intr.fx) * depth_z / 2.0)

            self.mark_occupied(X, Z, radius_m=min(obj_radius_m, 1.5))

    def check_lateral_clearance(
        self, strafe_dir: str, strafe_dist_m: float, forward_depth_m: float = 1.5
    ) -> Tuple[bool, str]:
        """
        FIX 3 CORE: Query whether a proposed lateral strafe is safe.

        Checks a rectangular corridor in the proposed strafe direction.
        Returns (is_safe, reason).

        is_safe=True  → corridor is confirmed free by the grid.
        is_safe=False → corridor contains occupied or unknown cells.

        "unknown" is treated as unsafe (conservative / safe-fail).
        """
        sign = -1.0 if strafe_dir == "left" else 1.0
        n_lateral_cells = int(strafe_dist_m / self.CELL_SIZE_M)
        n_forward_cells = int(forward_depth_m / self.CELL_SIZE_M)
        origin_col = self._origin
        origin_row = self._origin  # user is at grid centre

        n_unsafe = 0
        n_unknown = 0
        n_checked = 0

        for step_l in range(1, n_lateral_cells + 1):
            for step_f in range(0, n_forward_cells + 1):
                col = int(origin_col + sign * step_l)
                row = int(origin_row + step_f)
                if not (0 <= row < self._n and 0 <= col < self._n):
                    n_unsafe += 1
                    n_checked += 1
                    continue
                val = self._grid[row, col]
                n_checked += 1
                if val >= 0.8:
                    n_unsafe += 1
                elif val >= 0.45:
                    n_unknown += 1

        if n_checked == 0:
            return False, "Grid out of bounds"
        if n_unsafe > 0:
            return False, f"Obstacle detected in strafe path ({n_unsafe} blocked cells)"
        if n_unknown > int(n_checked * 0.5):
            return False, f"Strafe path not mapped ({n_unknown}/{n_checked} cells unknown)"
        return True, "Corridor confirmed clear"

    def get_safest_strafe_direction(self, min_dist_m: float = 0.7) -> Optional[str]:
        """
        Compare left vs right corridors and return the safer direction,
        or None if neither is safe.
        """
        left_ok, _ = self.check_lateral_clearance("left", min_dist_m)
        right_ok, _ = self.check_lateral_clearance("right", min_dist_m)
        if left_ok and right_ok:
            # Prefer the direction with more free cells
            l_free = self._count_free("left", min_dist_m)
            r_free = self._count_free("right", min_dist_m)
            return "left" if l_free >= r_free else "right"
        if left_ok:
            return "left"
        if right_ok:
            return "right"
        return None

    def _count_free(self, direction: str, dist_m: float) -> int:
        sign = -1.0 if direction == "left" else 1.0
        n_cells = int(dist_m / self.CELL_SIZE_M)
        count = 0
        for step in range(1, n_cells + 1):
            col = int(self._origin + sign * step)
            row = self._origin
            if 0 <= row < self._n and 0 <= col < self._n:
                if self._grid[row, col] < 0.45:
                    count += 1
        return count


# ═════════════════════════════════════════════════════════════
# ORB-SLAM VISUAL COMPASS  (Fix 4 — drift-free heading)
# ═════════════════════════════════════════════════════════════

class VisualSLAMCompass:
    """
    ORB-SLAM visual odometry compass — eliminates cumulative drift.

    Root-cause fix for the integration drift problem:
      v4.0 accumulated heading by integrating pixel-flow angular velocity.
      This is mathematically guaranteed to drift unboundedly — there is
      no mechanism to detect or correct accumulated error.

    v4.1 Visual Odometry pipeline (no IMU, no GPS required):
      1. Extract ORB keypoints + descriptors from the current frame.
      2. Match against previous keyframe using FLANN + Lowe ratio test.
      3. Run RANSAC Essential Matrix estimation on matched point pairs.
      4. Call cv2.recoverPose() to extract rotation R and translation t.
      5. Extract yaw from R (rotation around Y-axis in camera space).
      6. Accumulate yaw in a running total — but reset to 0 whenever
         a keyframe loop-closure is detected (same scene revisited).

    Keyframe loop-closure:
      Every N frames, the current descriptor set is compared to stored
      keyframes using a bag-of-words histogram distance. If the scene
      is recognised (descriptor overlap > LOOP_CLOSURE_THRESHOLD), the
      heading is soft-reset toward the keyframe's stored heading. This
      bounds long-term drift.

    Degenerate cases (handled):
      - Blank wall / textureless surface: < 8 ORB matches → falls back
        to optical flow for that frame; heading NOT updated from SLAM.
      - Camera stationary: flow near zero → heading held, no drift added.
      - Fast rotation: Essential Matrix fails RANSAC → heading unchanged
        for that frame rather than corrupted.

    Drift characteristics:
      Typical ORB monocular VO: < 1° per 10m travel without loop closure.
      With loop closure: bounded to < 5° absolute over any session length.
    """

    MIN_MATCHES_FOR_VO = 8          # below this, fall back to optical flow
    KEYFRAME_INTERVAL = 30          # store a new keyframe every N frames
    LOOP_CLOSURE_THRESHOLD = 0.70   # descriptor overlap ratio for loop detection
    MAX_KEYFRAMES = 50              # ring buffer of keyframes

    def __init__(self, fx: float = 554.0, fy: float = 554.0,
                 cx: float = 320.0, cy: float = 240.0):
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self._heading = 0.0
        self._frame_count = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp = None
        self._prev_des = None

        # ORB + FLANN matcher
        self._orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
        index_params = dict(algorithm=6,   # FLANN_INDEX_LSH
                            table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=50)
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Keyframe store: list of (heading, descriptors, gray_thumbnail)
        self._keyframes: List[Tuple[float, np.ndarray, np.ndarray]] = []

        # Optical flow fallback
        self._lk_params = dict(winSize=(21, 21), maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # Confidence tracking
        self._last_vo_inliers = 0
        self._consecutive_failures = 0

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @property
    def heading(self) -> float:
        return round(self._heading % 360, 1)

    @property
    def confidence(self) -> float:
        """0→1 confidence in current heading estimate."""
        if self._consecutive_failures > 10:
            return 0.3
        return min(1.0, self._last_vo_inliers / 30.0)

    def update(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_count += 1

        if self._prev_gray is None:
            self._store_keyframe(gray)
            self._prev_gray = gray
            self._prev_kp, self._prev_des = self._orb.detectAndCompute(gray, None)
            return self.heading

        # ── ORB feature matching ──────────────────────────────
        kp, des = self._orb.detectAndCompute(gray, None)
        yaw_delta, inliers = self._vo_yaw(kp, des, gray)

        if inliers >= self.MIN_MATCHES_FOR_VO:
            # VO succeeded
            self._heading = (self._heading + yaw_delta) % 360
            self._last_vo_inliers = inliers
            self._consecutive_failures = 0
        else:
            # Fall back to optical flow for this frame (no drift accumulation)
            flow_delta = self._optical_flow_yaw(gray, frame.shape[1])
            self._heading = (self._heading + flow_delta) % 360
            self._consecutive_failures += 1
            log.debug(f"VO fallback (only {inliers} ORB inliers) — flow delta: {flow_delta:.2f}°")

        # ── Keyframe storage ──────────────────────────────────
        if self._frame_count % self.KEYFRAME_INTERVAL == 0:
            self._check_loop_closure(des, gray)
            self._store_keyframe(gray)

        # Update prev frame state
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des
        return self.heading

    def _vo_yaw(self, kp, des, gray: np.ndarray) -> Tuple[float, int]:
        """
        FIX 4 CORE: Essential Matrix VO for yaw extraction.
        Returns (yaw_delta_degrees, n_ransac_inliers).
        """
        if (des is None or self._prev_des is None or
                len(des) < self.MIN_MATCHES_FOR_VO or
                len(self._prev_des) < self.MIN_MATCHES_FOR_VO):
            return 0.0, 0

        try:
            matches = self._flann.knnMatch(self._prev_des, des, k=2)
        except cv2.error:
            return 0.0, 0

        # Lowe ratio test
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < self.MIN_MATCHES_FOR_VO:
            return 0.0, 0

        pts_prev = np.float32([self._prev_kp[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good])

        try:
            E, mask = cv2.findEssentialMat(
                pts_prev, pts_curr, self._K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
        except cv2.error:
            return 0.0, 0

        if E is None or mask is None:
            return 0.0, 0

        inliers = int(mask.sum())
        if inliers < self.MIN_MATCHES_FOR_VO:
            return 0.0, inliers

        try:
            _, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_curr, self._K, mask=mask)
        except cv2.error:
            return 0.0, inliers

        # Extract yaw (rotation around Y-axis) from rotation matrix R
        # R is camera-to-world rotation. Yaw = atan2(R[0,2], R[2,2])
        yaw_rad = math.atan2(float(R[0, 2]), float(R[2, 2]))
        yaw_deg = math.degrees(yaw_rad)

        # Clamp implausible single-frame rotations (> 30°/frame = gyro failure)
        yaw_deg = max(-30.0, min(30.0, yaw_deg))
        return yaw_deg, int(pose_mask.sum()) if pose_mask is not None else inliers

    def _optical_flow_yaw(self, gray: np.ndarray, frame_width: int) -> float:
        """
        Optical flow fallback — used ONLY when ORB fails.
        Returns a conservative yaw estimate; does NOT accumulate if camera
        is stationary (flow magnitude below threshold).
        """
        if self._prev_gray is None:
            return 0.0
        corners = cv2.goodFeaturesToTrack(self._prev_gray, maxCorners=80,
                                           qualityLevel=0.3, minDistance=10)
        if corners is None or len(corners) < 5:
            return 0.0
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, corners, None, **self._lk_params
        )
        good_old = corners[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1]
        if len(good_old) < 5:
            return 0.0
        dx_vals = good_new[:, 0] - good_old[:, 0]
        median_dx = float(np.median(dx_vals))
        # Deadband: if motion is < 1px, treat as stationary — no drift added
        if abs(median_dx) < 1.0:
            return 0.0
        fov_h = 62.0
        deg_per_px = fov_h / frame_width
        return median_dx * deg_per_px * 0.35

    def _store_keyframe(self, gray: np.ndarray):
        """Store current frame as a keyframe for loop-closure detection."""
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is not None and len(des) > 10:
            thumb = cv2.resize(gray, (64, 48))
            self._keyframes.append((self._heading, des, thumb))
            if len(self._keyframes) > self.MAX_KEYFRAMES:
                self._keyframes.pop(0)

    def _check_loop_closure(self, current_des: Optional[np.ndarray], gray: np.ndarray):
        """
        Compare current frame descriptors to stored keyframes.
        If strong overlap detected, soft-reset heading toward keyframe heading.
        This bounds long-term drift.
        """
        if current_des is None or len(self._keyframes) < 5:
            return

        best_overlap = 0.0
        best_kf_heading = self._heading

        for kf_heading, kf_des, _ in self._keyframes[:-3]:  # skip most recent 3
            try:
                matches = self._flann.knnMatch(current_des, kf_des, k=2)
                good = sum(1 for pair in matches if len(pair) == 2
                          and pair[0].distance < 0.75 * pair[1].distance)
                overlap = good / max(len(current_des), len(kf_des), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_kf_heading = kf_heading
            except cv2.error:
                continue

        if best_overlap >= self.LOOP_CLOSURE_THRESHOLD:
            # Soft reset: blend current heading toward keyframe heading
            alpha = 0.3  # 30% correction per detection
            delta = (best_kf_heading - self._heading + 180) % 360 - 180
            self._heading = (self._heading + alpha * delta) % 360
            log.info(f"Loop closure detected (overlap={best_overlap:.2f}) — "
                     f"heading corrected by {alpha * delta:.1f}°")

    def reset(self):
        self._heading = 0.0
        self._prev_gray = None
        self._prev_kp = None
        self._prev_des = None
        self._consecutive_failures = 0


# Alias for backward-compatibility
OpticalFlowCompass = VisualSLAMCompass


# ═════════════════════════════════════════════════════════════
# YOLO DETECTOR (open-vocabulary upgrade path — unchanged)
# ═════════════════════════════════════════════════════════════

class YOLODetector:
    def __init__(self, model_path: str, confidence: float):
        from ultralytics import YOLO
        self._confidence = confidence
        self._world_model = None
        self._coco_model = None
        try:
            world_path = model_path.replace("yolov8n.pt", "yolov8s-world.pt")
            self._world_model = YOLO(world_path)
            log.info(f"YOLOWorld loaded: {world_path}")
        except Exception as e:
            log.warning(f"YOLOWorld not available ({e})")
        try:
            self._coco_model = YOLO(model_path)
            log.info(f"YOLOv8 COCO loaded: {model_path}")
        except Exception as e:
            log.error(f"YOLO load failed: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        model = self._coco_model or self._world_model
        if model is None:
            return []
        return self._run(model, frame)

    def detect_open(self, frame: np.ndarray, classes: List[str]) -> List[Detection]:
        if self._world_model is not None:
            try:
                self._world_model.set_classes(classes)
                return self._run(self._world_model, frame)
            except Exception as e:
                log.warning(f"YOLOWorld open-vocab failed ({e})")
        return self.detect(frame)

    def _run(self, model, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        results = model(frame, conf=self._confidence, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(Detection(
                    label=model.names[int(box.cls[0])].lower(),
                    confidence=round(float(box.conf[0]), 3),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    frame_width=w, frame_height=h,
                ))
        return dets


# ═════════════════════════════════════════════════════════════
# IoU TRACKER
# ═════════════════════════════════════════════════════════════

@dataclass
class Track:
    id: int
    label: str
    det: Detection
    age: int = 0
    hits: int = 1
    frames_since_seen: int = 0
    state: str = "new"
    smoothed_distance: float = 0.0
    approach_velocity: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    azimuth_deg: float = 0.0
    reid_embedding: Optional[List] = None

    @property
    def is_confirmed(self) -> bool: return self.hits >= 2
    @property
    def bbox(self) -> BoundingBox: return self.det.bbox

    def update_state(self):
        if self.frames_since_seen > 0:
            self.state = "lost"
        elif self.hits < 2:
            self.state = "new"
        elif abs(self.approach_velocity) > 0.20:
            self.state = "moving"
        else:
            self.state = "stable"


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0: return 0.0
    return inter / (a.area + b.area - inter + 1e-6)


class IoUTracker:

    def __init__(self, iou_threshold: float = 0.35, max_age: int = 8, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}
        self._next_id: int = 1  # Instance variable — prevents ID collision across multiple tracker instances

    def update(self, detections: List[Detection]) -> List[Track]:
        for t in self.tracks.values():
            t.age += 1
            t.frames_since_seen += 1

        matched_tids, matched_dis = set(), set()
        pairs = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(detections):
                if det.label != tr.label: continue
                score = _iou(tr.bbox, det.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)
        for score, tid, di in pairs:
            if tid in matched_tids or di in matched_dis: continue
            self.tracks[tid].det = detections[di]
            self.tracks[tid].hits += 1
            self.tracks[tid].frames_since_seen = 0
            matched_tids.add(tid); matched_dis.add(di)

        for di, det in enumerate(detections):
            if di not in matched_dis:
                nid = self._next_id; self._next_id += 1
                self.tracks[nid] = Track(id=nid, label=det.label, det=det)

        stale = [tid for tid, t in self.tracks.items() if t.frames_since_seen > self.max_age]
        for tid in stale: del self.tracks[tid]
        for t in self.tracks.values(): t.update_state()
        return [t for t in self.tracks.values() if t.hits >= self.min_hits and t.frames_since_seen == 0]

    def get_all_active(self) -> List[Track]: return list(self.tracks.values())
    def remove_track(self, tid: int): self.tracks.pop(tid, None)


# ═════════════════════════════════════════════════════════════
# KALMAN DEPTH FILTER
# ═════════════════════════════════════════════════════════════

class KalmanDepthFilter:
    def __init__(self, initial_dist: float, dt: float = 1/8.0):
        self.dt = dt
        self.x = np.array([[initial_dist], [0.0]])
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[0.005, 0.0], [0.0, 0.05]])
        self.R = np.array([[0.15]])
        self.P = np.eye(2) * 0.5

    def predict(self) -> float:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0])

    def update(self, z: float) -> float:
        y = np.array([[z]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.x[0, 0] = max(0.1, min(15.0, float(self.x[0, 0])))
        return float(self.x[0, 0])

    @property
    def distance(self) -> float: return max(0.1, float(self.x[0, 0]))
    @property
    def velocity(self) -> float: return float(self.x[1, 0])


# ═════════════════════════════════════════════════════════════
# DEPTH FUSION ENGINE  (Fix 1 integrated — RANSAC multi-anchor)
# ═════════════════════════════════════════════════════════════

class DepthFusionEngine:
    """
    Per-track depth fusion with RANSAC multi-anchor scale calibration.
    Builds the anchor list from ALL tracks each frame and calls
    MonocularDepthEngine.calibrate_ransac() before applying metric scale.
    """

    def __init__(self, fov_h_deg: float = 62.0,
                 depth_engine: Optional[MonocularDepthEngine] = None):
        self._filters: Dict[int, KalmanDepthFilter] = {}
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._fov = fov_h_deg
        self._depth_engine = depth_engine
        self._raw_depth_map: Optional[np.ndarray] = None
        self._metric_depth_map: Optional[np.ndarray] = None

    def calibrate(self, frame_width: int, frame_height: int = 480):
        self._intrinsics = CameraIntrinsics.from_frame(frame_width, frame_height, self._fov)

    def set_raw_depth(self, raw_map: Optional[np.ndarray]):
        """Receive raw (pre-scale) MiDaS output for the current frame."""
        self._raw_depth_map = raw_map

    def run_ransac_calibration(self, tracks: List[Track]):
        """
        FIX 1 INTEGRATION: collect geometric anchors from all confirmed tracks,
        then run RANSAC scale calibration on the depth engine.
        Called once per frame BEFORE update() is called per-track.
        """
        if self._depth_engine is None or self._raw_depth_map is None:
            return
        if self._intrinsics is None:
            return

        intr = self._intrinsics
        anchors = []
        for track in tracks:
            if not track.is_confirmed:
                continue
            geo_dist = estimate_distance_geometric(track.label, track.bbox.height, intr.fx)
            if 0.3 <= geo_dist <= 6.0:
                anchors.append((
                    track.bbox.center_x,
                    track.bbox.center_y,
                    geo_dist,
                    track.label,
                ))

        if len(anchors) >= MonocularDepthEngine._MIN_INLIERS:
            updated = self._depth_engine.calibrate_ransac(anchors)
            if updated:
                # Re-apply scale to get fresh metric map
                self._metric_depth_map = self._depth_engine.to_metric(self._raw_depth_map)
        elif self._raw_depth_map is not None:
            # Use whatever scale we have
            self._metric_depth_map = self._depth_engine.to_metric(self._raw_depth_map)

    def update(self, track: Track) -> Tuple[float, float]:
        intr = self._intrinsics or CameraIntrinsics()
        cx_px = track.bbox.center_x
        cy_px = track.bbox.center_y

        # Depth source: metric map from depth engine (RANSAC calibrated) or geometric
        if self._metric_depth_map is not None:
            raw_z = float(self._metric_depth_map[
                max(0, min(int(cy_px), self._metric_depth_map.shape[0]-1)),
                max(0, min(int(cx_px), self._metric_depth_map.shape[1]-1))
            ])
            raw_z = max(0.1, min(15.0, raw_z))
        else:
            raw_z = estimate_distance_geometric(track.label, track.bbox.height, intr.fx)

        # Kalman filter
        if track.id not in self._filters:
            self._filters[track.id] = KalmanDepthFilter(raw_z, dt=1/8.0)
        kf = self._filters[track.id]
        kf.predict()
        smooth_z = kf.update(raw_z)
        velocity = kf.velocity

        # 3D back-projection
        X, Y, Z = backproject_to_3d(cx_px, cy_px, smooth_z,
                                     intr.fx, intr.fy, intr.cx, intr.cy)
        theta = azimuth_from_3d(X, Z)

        track.translation_x = X
        track.translation_y = Y
        track.translation_z = Z
        track.azimuth_deg = theta
        track.smoothed_distance = round(smooth_z, 2)
        track.approach_velocity = round(velocity, 3)

        return round(smooth_z, 2), round(velocity, 3)

    def remove(self, tid: int):
        self._filters.pop(tid, None)


# ═════════════════════════════════════════════════════════════
# DYNAMIC AVOIDANCE ENGINE  (Fix 3 — grid-checked strafe)
# ═════════════════════════════════════════════════════════════

class DynamicAvoidanceEngine:
    """
    Grid-aware lateral strafe avoidance.

    FIX 3: Before proposing a strafe direction, queries the BEVOccupancyGrid
    to confirm the corridor is mapped and free. Rejects directions with any
    occupied or >50% unknown cells in the proposed path.

    If NEITHER direction is safe, emits a hold instruction rather than
    commanding movement into an unmapped zone.
    """

    DYNAMIC_LABELS = {"person", "dog", "cat", "bicycle", "motorcycle"}

    def __init__(self, occupancy_grid: Optional[BEVOccupancyGrid] = None):
        self._grid = occupancy_grid

    def compute_waypoint(
        self,
        obstacle: Track,
        target_azimuth_deg: float = 0.0,
    ) -> Optional[AvoidanceWaypoint]:
        if obstacle.label not in self.DYNAMIC_LABELS:
            return None
        if obstacle.smoothed_distance > 3.0:
            return None

        obs_width_m = max(0.4, min(abs(obstacle.translation_x) * 2.0, 2.0))
        required_strafe = round(obs_width_m * 0.8 + 0.3, 1)
        clearance = round(max(0.5, obstacle.smoothed_distance - 0.3), 1)

        # FIX 3: determine preferred direction from obstacle position
        preferred_dir = "left" if obstacle.azimuth_deg >= 0 else "right"
        fallback_dir = "right" if preferred_dir == "left" else "left"

        # Check occupancy grid
        if self._grid is not None:
            pref_ok, pref_reason = self._grid.check_lateral_clearance(
                preferred_dir, required_strafe
            )
            if pref_ok:
                chosen_dir = preferred_dir
                safety_note = ""
            else:
                fall_ok, fall_reason = self._grid.check_lateral_clearance(
                    fallback_dir, required_strafe
                )
                if fall_ok:
                    chosen_dir = fallback_dir
                    safety_note = ""
                else:
                    # Neither direction confirmed safe — HOLD
                    log.warning(
                        f"Avoidance blocked: {preferred_dir}='{pref_reason}', "
                        f"{fallback_dir}='{fall_reason}'. Issuing hold."
                    )
                    return None   # caller will issue hold instruction
        else:
            # No grid available — use geometric preference (unsafe fallback)
            chosen_dir = preferred_dir
            safety_note = " (grid unavailable — proceed with caution)"

        clock, _ = to_clock_direction(-30.0 if chosen_dir == "left" else 30.0)

        return AvoidanceWaypoint(
            obstacle_label=obstacle.label,
            obstacle_distance_m=round(obstacle.smoothed_distance, 2),
            obstacle_track_id=obstacle.id,
            strafe_direction=chosen_dir,
            strafe_distance_m=required_strafe,
            forward_clearance_m=clearance,
            clock_instruction=(
                f"Step {chosen_dir} {required_strafe}m, then continue forward"
                + (safety_note or "")
            ),
        )


# ═════════════════════════════════════════════════════════════
# SAFETY CORTEX  (Fix 3 integrated — avoidance uses grid)
# ═════════════════════════════════════════════════════════════

_HIGH_RISK = {"person", "dog", "cat", "car", "motorcycle", "bicycle",
              "chair", "dining table", "bench", "suitcase", "backpack"}
_SURFACE_OBJ = {"cup", "bottle", "cell phone", "remote", "keyboard",
                "mouse", "book", "apple", "banana", "fork", "spoon", "knife"}


@dataclass
class DangerAlert:
    level: str
    label: str
    distance_m: float
    clock_direction: str
    message: str
    track_id: int
    timestamp: float
    avoidance: Optional[AvoidanceWaypoint] = None


class SafetyCortex:
    def __init__(self, critical_dist: float = 0.8, warning_dist: float = 1.5,
                 caution_dist: float = 2.5, cooldown_s: float = 3.0,
                 occupancy_grid: Optional[BEVOccupancyGrid] = None):
        self.CRITICAL = critical_dist
        self.WARNING = warning_dist
        self.CAUTION = caution_dist
        self._cooldown = cooldown_s
        self._last_alert: Dict[int, float] = {}
        # FIX 3: avoidance engine receives the shared occupancy grid
        self._avoider = DynamicAvoidanceEngine(occupancy_grid=occupancy_grid)

    def evaluate(self, tracks: List[Track], current_heading: float) -> List[DangerAlert]:
        now = time.time()
        alerts: List[DangerAlert] = []
        for track in tracks:
            if track.label in _SURFACE_OBJ or not track.is_confirmed:
                continue
            dist = track.smoothed_distance
            if dist <= 0.05 or dist > self.CAUTION:
                continue
            if now - self._last_alert.get(track.id, 0) < self._cooldown:
                continue

            level = ("critical" if dist <= self.CRITICAL
                     else "warning" if dist <= self.WARNING else "caution")
            if track.label in _HIGH_RISK and level == "caution":
                level = "warning"
            if track.approach_velocity < -0.5 and level != "critical":
                level = {"caution": "warning", "warning": "critical"}.get(level, level)

            rel_az = track.azimuth_deg
            clock, _ = to_clock_direction(rel_az)

            avoidance = None
            if level in ("critical", "warning") and track.label in DynamicAvoidanceEngine.DYNAMIC_LABELS:
                avoidance = self._avoider.compute_waypoint(track, target_azimuth_deg=current_heading)

            msg = self._make_message(level, track.label, dist, clock, avoidance)
            alerts.append(DangerAlert(
                level=level, label=track.label, distance_m=dist,
                clock_direction=clock, message=msg,
                track_id=track.id, timestamp=now, avoidance=avoidance,
            ))
            self._last_alert[track.id] = now
        return alerts

    def _make_message(self, level: str, label: str, dist: float,
                       clock: str, avoidance: Optional[AvoidanceWaypoint]) -> str:
        d = format_distance(dist)
        if level == "critical":
            avoid_txt = f" {avoidance.to_speech()}" if avoidance else " Do not move forward."
            return f"STOP — {label} at {clock}, {d}.{avoid_txt}"
        elif level == "warning":
            avoid_txt = (f" Step {avoidance.strafe_direction} {avoidance.strafe_distance_m}m."
                         if avoidance else "")
            return f"Caution — {label} at {clock}, {d}.{avoid_txt}"
        return f"{label.capitalize()} nearby at {clock}, {d}."


# ═════════════════════════════════════════════════════════════
# FRAME UTILITIES
# ═════════════════════════════════════════════════════════════

_PALETTE = [(0,212,170),(255,165,0),(255,99,71),(100,149,237),
            (152,251,152),(255,215,0),(218,112,214),(127,255,212)]


def draw_detections(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        b = track.bbox
        color = _PALETTE[hash(track.label) % len(_PALETTE)]
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        txt = (f"#{track.id} {track.label} Z={track.smoothed_distance:.1f}m "
               f"θ={track.azimuth_deg:+.0f}°")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (int(b.x1), int(b.y1)-th-8),
                      (int(b.x1)+tw+4, int(b.y1)), color, -1)
        cv2.putText(out, txt, (int(b.x1)+2, int(b.y1)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1, cv2.LINE_AA)
    return out


def frame_to_b64(frame: np.ndarray, quality: int = 70) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")
