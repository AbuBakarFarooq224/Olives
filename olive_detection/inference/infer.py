"""
=============================================================
 OliveVision — Real-Time Inference Engine
 Supports:
   • Webcam / video file / image folder
   • ByteTrack-based persistent olive counting
   • Region of Interest (ROI) counting
   • FPS display and output recording
=============================================================
"""

import cv2
import time
import yaml
import argparse
import numpy as np
from pathlib import Path
from collections import deque, defaultdict

import torch
from ultralytics import RTDETR


# ─────────────────────────────────────────────────────────────
# Simple Kalman Filter (for ByteTrack-lite tracking)
# ─────────────────────────────────────────────────────────────

class KalmanTracker:
    """
    Lightweight Kalman filter tracker for single object.
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    count = 0

    def __init__(self, bbox: np.ndarray, score: float):
        KalmanTracker.count += 1
        self.id    = KalmanTracker.count
        self.hits  = 1
        self.misses = 0
        self.score = score

        x, y, w, h = bbox
        self.state = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)

        # Transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1

        # Observation matrix
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1

        # Covariances
        self.P = np.eye(8, dtype=np.float32) * 10
        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 1.0

    def predict(self):
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q
        self.misses += 1
        return self.state[:4]

    def update(self, bbox: np.ndarray, score: float):
        self.score = score
        self.hits += 1
        self.misses = 0

        z = bbox.reshape(4, 1)
        y = z - (self.H @ self.state).reshape(4, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y).flatten()
        self.P     = (np.eye(8) - K @ self.H) @ self.P

    @property
    def bbox(self) -> np.ndarray:
        return self.state[:4].copy()

    @property
    def box_xyxy(self) -> np.ndarray:
        x, y, w, h = self.state[:4]
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])


# ─────────────────────────────────────────────────────────────
# Multi-Object Tracker (ByteTrack-lite)
# ─────────────────────────────────────────────────────────────

def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of [x1,y1,x2,y2] boxes."""
    A, B = len(a), len(b)
    iou  = np.zeros((A, B), dtype=np.float32)
    for i in range(A):
        x1 = np.maximum(a[i, 0], b[:, 0])
        y1 = np.maximum(a[i, 1], b[:, 1])
        x2 = np.minimum(a[i, 2], b[:, 2])
        y2 = np.minimum(a[i, 3], b[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (a[i, 2]-a[i, 0]) * (a[i, 3]-a[i, 1])
        area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
        iou[i] = inter / (area_a + area_b - inter + 1e-6)
    return iou


class ByteTracker:
    """
    Lightweight ByteTrack-inspired tracker.
    Maintains track ID continuity across frames for stable counting.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_thresh: float = 0.3):
        self.max_age   = max_age      # frames before track is deleted
        self.min_hits  = min_hits     # frames before track is confirmed
        self.iou_thresh = iou_thresh
        self.trackers: list[KalmanTracker] = []
        KalmanTracker.count = 0

    def update(self, detections: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Args:
            detections: (N, 4) [xc, yc, w, h] normalized boxes (pixel coords)
            scores:     (N,)

        Returns:
            active_tracks: (M, 5) [x1, y1, x2, y2, track_id]
        """
        # Predict all existing tracks
        predicted = np.array([t.predict() for t in self.trackers]) if self.trackers else np.empty((0, 4))

        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted, scores
        )

        # Update matched trackers
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(detections[d_idx], scores[d_idx])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanTracker(detections[d_idx], scores[d_idx]))

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.misses <= self.max_age]

        # Return confirmed tracks
        active = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.misses == 0:
                x, y, w, h = t.bbox
                active.append([x - w/2, y - h/2, x + w/2, y + h/2, t.id])

        return np.array(active) if active else np.empty((0, 5))

    def _associate(self, dets, preds, scores):
        if len(preds) == 0 or len(dets) == 0:
            return [], list(range(len(dets))), list(range(len(preds)))

        # Convert xywh → xyxy for IoU
        def xywh2xyxy_np(b):
            return np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2,
                             b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], axis=1)

        det_xyxy  = xywh2xyxy_np(dets)   if dets.ndim == 2 else np.empty((0,4))
        pred_xyxy = xywh2xyxy_np(preds)  if preds.ndim == 2 else np.empty((0,4))

        iou = iou_batch(det_xyxy, pred_xyxy)  # (N_det, N_trk)

        matched_pairs = []
        used_dets, used_trks = set(), set()
        for _ in range(min(len(dets), len(preds))):
            idx = np.unravel_index(np.argmax(iou), iou.shape)
            d, t = idx
            if iou[d, t] < self.iou_thresh:
                break
            matched_pairs.append((t, d))
            used_dets.add(d)
            used_trks.add(t)
            iou[d, :] = -1
            iou[:, t] = -1

        unmatched_dets = [i for i in range(len(dets))  if i not in used_dets]
        unmatched_trks = [i for i in range(len(preds)) if i not in used_trks]
        return matched_pairs, unmatched_dets, unmatched_trks


# ─────────────────────────────────────────────────────────────
# Olive Inference Engine
# ─────────────────────────────────────────────────────────────

class OliveInferenceEngine:
    """
    Real-time olive detection and counting engine.

    Usage:
        engine = OliveInferenceEngine("models/checkpoints/best.pt", config)
        engine.run(source=0)  # webcam
        engine.run(source="video.mp4")
    """

    def __init__(self, weights_path: str, config: dict):
        self.config     = config
        inf_cfg         = config["inference"]
        cnt_cfg         = inf_cfg["counting"]

        self.conf_thresh = inf_cfg["conf_threshold"]
        self.iou_thresh  = inf_cfg["iou_threshold"]
        self.input_size  = inf_cfg["input_size"]
        self.device      = inf_cfg["device"]
        self.display_fps = inf_cfg["display_fps"]
        self.display_cnt = inf_cfg["display_count"]
        self.save_output = inf_cfg["save_output"]
        self.output_path = Path(inf_cfg["output_path"])

        self.use_tracking = cnt_cfg["use_tracking"]
        self.roi_enabled  = cnt_cfg["roi_enabled"]
        self.roi_coords   = cnt_cfg["roi_coords"]

        # Load model
        print(f"🔄 Loading model from {weights_path} ...")
        self.model = RTDETR(weights_path)
        print(f"✅ Model loaded on {self.device}")

        # Tracker
        self.tracker = ByteTracker(max_age=30, min_hits=3) if self.use_tracking else None

        # FPS smoothing (rolling window)
        self.fps_buffer = deque(maxlen=30)

    def _preprocess(self, frame: np.ndarray) -> tuple:
        """Resize + pad frame to model input size."""
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))
        canvas  = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale, (w, h)

    def _postprocess(self, results, orig_size, scale):
        """Extract boxes + scores from Ultralytics result object."""
        boxes, scores = [], []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            # Ultralytics returns xyxy as numpy array, no need for .cpu()
            b = r.boxes.xyxy if isinstance(r.boxes.xyxy, np.ndarray) else r.boxes.xyxy.cpu().numpy()   # (N, 4) pixel coords at input_size
            s = r.boxes.conf if isinstance(r.boxes.conf, np.ndarray) else r.boxes.conf.cpu().numpy()   # (N,)

            # Scale back to original image
            b /= scale
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, orig_size[0])
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, orig_size[1])

            boxes.append(b)
            scores.append(s)

        if boxes:
            return np.concatenate(boxes), np.concatenate(scores)
        return np.empty((0, 4)), np.empty((0,))

    def _apply_roi(self, boxes: np.ndarray, scores: np.ndarray):
        """Filter boxes that fall within the ROI."""
        if not self.roi_enabled or len(boxes) == 0:
            return boxes, scores
        x1r, y1r, x2r, y2r = self.roi_coords
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        mask = (cx >= x1r) & (cx <= x2r) & (cy >= y1r) & (cy <= y2r)
        return boxes[mask], scores[mask]

    def _draw(self, frame, boxes, scores, count, fps):
        """Annotate frame with detections."""
        for box, score in zip(boxes.astype(int), scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
            lbl = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 200, 80), -1)
            cv2.putText(frame, lbl, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # ROI rectangle
        if self.roi_enabled:
            x1r, y1r, x2r, y2r = self.roi_coords
            cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), (255, 200, 0), 2)
            cv2.putText(frame, "ROI", (x1r + 4, y1r + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # Count overlay
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, 55), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Olives: {count}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 100), 2, cv2.LINE_AA)

        # FPS
        if fps is not None:
            fps_str = f"FPS: {fps:.1f}"
            (fw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(frame, fps_str, (w - fw - 10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 2, cv2.LINE_AA)

        return frame

    def infer_image(self, image_path: str) -> dict:
        """
        Run detection on a single image.

        Returns:
            dict with 'count', 'boxes', 'scores', 'annotated_image'
        """
        frame    = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        # Extract detections from results
        if results[0].boxes:
            boxes = results[0].boxes.xyxy if isinstance(results[0].boxes.xyxy, np.ndarray) else results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf if isinstance(results[0].boxes.conf, np.ndarray) else results[0].boxes.conf.cpu().numpy()
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))

        boxes, scores = self._apply_roi(boxes, scores)
        count         = len(boxes)
        annotated     = self._draw(frame.copy(), boxes, scores, count, None)

        return {
            "count":            count,
            "boxes":            boxes,
            "scores":           scores,
            "annotated_image":  annotated,
        }

    def run(self, source=None) -> None:
        """
        Run real-time inference loop.

        Args:
            source: 0 for webcam, or path to video/image file
        """
        if source is None:
            source = self.config["inference"]["source"]

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if self.save_output:
            self.output_path.mkdir(parents=True, exist_ok=True)
            out_file = self.output_path / "output.mp4"
            fourcc   = cv2.VideoWriter.fourcc(*"mp4v")
            writer   = cv2.VideoWriter(str(out_file), fourcc, fps_src, (w, h))

        print(f"\n🎥 Starting inference on source: {source}")
        print(f"   Press 'q' to quit | 's' to save screenshot | 'r' to toggle ROI")

        frame_idx = 0
        try:
            while True:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                # Inference
                results = self.model.predict(
                    frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.input_size,
                    device=self.device,
                    verbose=False,
                )

                
                if results[0].boxes:
                    boxes  = results[0].boxes.xyxy if isinstance(results[0].boxes.xyxy, np.ndarray) else results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf if isinstance(results[0].boxes.conf, np.ndarray) else results[0].boxes.conf.cpu().numpy()
                else:
                    boxes = np.empty((0, 4))
                    scores = np.empty((0,))

                # ROI filter
                boxes, scores = self._apply_roi(boxes, scores)

                # Tracking
                if self.tracker is not None and len(boxes) > 0:
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2
                    bw = boxes[:, 2] - boxes[:, 0]
                    bh = boxes[:, 3] - boxes[:, 1]
                    dets_xywh = np.stack([cx, cy, bw, bh], axis=1)
                    tracks = self.tracker.update(dets_xywh, scores)
                    count  = len(tracks)
                    # Remap boxes to tracked boxes
                    if len(tracks) > 0:
                        boxes  = tracks[:, :4]
                        scores = np.ones(len(tracks))
                else:
                    count = len(boxes)

                # FPS
                t1  = time.perf_counter()
                self.fps_buffer.append(1.0 / max(t1 - t0, 1e-6))
                fps = np.mean(self.fps_buffer)

                # Annotate
                annotated = self._draw(frame, boxes, scores, count, fps)

                cv2.imshow("OliveVision — Real-Time Detection", annotated)

                if writer:
                    writer.write(annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    shot_path = self.output_path / f"screenshot_{frame_idx:04d}.jpg"
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(shot_path), annotated)
                    print(f"📸 Screenshot saved → {shot_path}")
                elif key == ord("r"):
                    self.roi_enabled = not self.roi_enabled
                    print(f"ROI: {'ON' if self.roi_enabled else 'OFF'}")

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"\n✅ Inference finished. Processed {frame_idx} frames.")


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="OliveVision Real-Time Inference")
    parser.add_argument("--weights", type=str, default="models/checkpoints/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--source",  type=str, default="0",
                        help="Video source (0=webcam, or video path)")
    parser.add_argument("--config",  type=str, default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--conf",    type=float, default=None,
                        help="Confidence threshold (overrides config)")
    parser.add_argument("--save",    action="store_true",
                        help="Save output video")
    parser.add_argument("--image",   type=str, default=None,
                        help="Run on a single image instead of video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    source = int(args.source) if args.source.isdigit() else args.source
    if args.conf:
        config["inference"]["conf_threshold"] = args.conf
    if args.save:
        config["inference"]["save_output"] = True

    engine = OliveInferenceEngine(args.weights, config)

    if args.image:
        result = engine.infer_image(args.image)
        print(f"\n🫒 Olives detected: {result['count']}")
        cv2.imshow("Detection Result", result["annotated_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        engine.run(source=source)
