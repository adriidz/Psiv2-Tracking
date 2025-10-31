# python
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from tracker import *

# Must be set before importing ultralytics
os.environ["ULTRALYTICS_HOME"] = str(Path(__file__).resolve().parent)

import cv2
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from VehicleCounter import VehicleCounter
CAR_CLASS_ID = 2  # COCO: 2 = car

YOLO_DIR = Path(__file__).resolve().parent
SETTINGS["runs_dir"] = str(YOLO_DIR / "runs")
SETTINGS["weights_dir"] = str(YOLO_DIR / "weights")
Path(SETTINGS["runs_dir"]).mkdir(parents=True, exist_ok=True)
Path(SETTINGS["weights_dir"]).mkdir(parents=True, exist_ok=True)


# --- Constants for playback and persistence ---
PLAYBACK_OPTIONS = ["fast", "1x"]
PLAYBACK = PLAYBACK_OPTIONS[0]  # "1x" for real-time, "fast" for as-fast-as-possible
PERSIST_FRAMES = 3              # keep last detections visible for N frames
# ---------------------------------------------

def draw_boxes(frame, result, label_suffix=""):
    if result is None or result.boxes is None:
        return frame
    names = result.names
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls_id = int(b.cls)
        conf = float(b.conf)
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}{label_suffix}"
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def window_closed(win_name: str) -> bool:
    # Returns True if the window was closed by the user (clicking the X)
    try:
        v = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        a = cv2.getWindowProperty(win_name, cv2.WND_PROP_AUTOSIZE)
        return v < 1 or a < 0
    except cv2.error:
        return True

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=str(YOLO_DIR / "weights" / "yolo11n.pt"))
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--skip", type=int, default=2, help="Run inference every N frames")
    p.add_argument("--display", action="store_true", default=True, help="Show window (press Q to quit)")  # default ON
    p.add_argument("--reuse-last", action="store_true", default=True, help="Draw last detections on skipped frames")  # default True
    return p.parse_args()

def open_capture(video_path: Path):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {video_path}")
    return cap

def prepare_writer(cap: cv2.VideoCapture):
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_dir = Path(SETTINGS["runs_dir"]) / "cars_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cars_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
    if not writer.isOpened():
        out_path = out_dir / f"cars_{ts}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
    return writer, out_path, width, height, fps_in

def init_model(weights_path: str):
    return YOLO(weights_path)

def setup_display_if_needed(display: bool, width: int, height: int):
    if not display:
        return
    win_name = "cars"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        from ctypes import windll
        user32 = windll.user32
        user32.SetProcessDPIAware()
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except Exception:
        screen_w, screen_h = 1920, 1080

    scale = min(0.8 * screen_w / width, 0.8 * screen_h / height, 1.0)
    win_w = max(1, int(width * scale))
    win_h = max(1, int(height * scale))
    cv2.resizeWindow(win_name, win_w, win_h)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except cv2.error:
        pass
    return win_name


def process_frames(cap: cv2.VideoCapture, writer: cv2.VideoWriter, model, args, width: int, height: int, fps_in: float,
                   out_path: Path, tracker: Tracker):
    # Contador horizontal (cuenta ambas direcciones: arriba-abajo)
    counter_horizontal = VehicleCounter(line_position=2 / 3, margin=5, orientation='horizontal')
    counter_horizontal.set_line_position(height)

    # Línea vertical izquierda (solo cuenta der->izq)
    counter_left = VehicleCounter(line_position=0.3, margin=3, orientation='vertical', direction='right_to_left')
    counter_left.set_line_position(height, width)

    # Línea vertical derecha (solo cuenta izq->der)
    counter_right = VehicleCounter(line_position=0.85, margin=3, orientation='vertical', direction='left_to_right')
    counter_right.set_line_position(height, width)

    frame_period = 1.0 / (fps_in if fps_in > 0 else 30.0)
    next_frame_ts = time.perf_counter() + frame_period

    last_result = None
    frame_idx = 0
    t0 = time.time()
    win_name = "cars" if args.display else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_shape = frame.shape  # Dimensiones del frame

        if args.skip <= 1 or frame_idx % args.skip == 0:
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                classes=[CAR_CLASS_ID],
                verbose=False
            )
            last_result = results[0] if results else None
            if last_result is not None:
                print(f"Frame {frame_idx}: {len(last_result.boxes)} cars")

            detections = yolo_result_to_detections(last_result) if last_result is not None else []
            track_ids = tracker.update(frame, detections)

        # Actualizar los tres contadores con frame_shape y límites de línea
        for track_id, car in track_ids.items():
            if car.centroids:
                center_x, center_y = car.centroids[-1]

                # Contador horizontal (línea verde completa)
                counter_horizontal.update(
                    track_id,
                    center_x=int(center_x),
                    center_y=int(center_y),
                    line_start=0.0,
                    line_end=1.0,
                    frame_shape=frame_shape
                )

                # Línea vertical izquierda (roja) - estirada de 0.2 a 0.85
                counter_left.update(
                    track_id,
                    center_x=int(center_x),
                    center_y=int(center_y),
                    line_start=0.2,
                    line_end=0.4,
                    frame_shape=frame_shape
                )

                # Línea vertical derecha (azul) - estirada de 0.2 a 0.85
                counter_right.update(
                    track_id,
                    center_x=int(center_x),
                    center_y=int(center_y),
                    line_start=0.2,
                    line_end=0.4,
                    frame_shape=frame_shape
                )

        annotated = tracker.draw_tracks(frame.copy(), min_hits=1)

        # Dibujar líneas con los mismos límites
        counter_horizontal.draw(annotated, color=(0, 255, 0), label_y_start=30, line_start=0.0, line_end=1.0)
        counter_left.draw(annotated, color=(255, 0, 0), label_y_start=90, line_start=0.2, line_end=0.4)
        counter_right.draw(annotated, color=(0, 0, 255), label_y_start=120, line_start=0.2, line_end=0.4)

        writer.write(annotated)

        if args.display:
            if window_closed(win_name):
                break
            cv2.imshow(win_name, annotated)

            if PLAYBACK == "1x":
                now = time.perf_counter()
                delay_s = max(0.0, next_frame_ts - now)
                delay_ms = max(1, int(delay_s * 1000))
                key = cv2.waitKey(delay_ms) & 0xFF
                next_frame_ts += frame_period
            else:
                key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"Frames: {frame_idx} | Elapsed: {elapsed:.1f}s | Out: {out_path}")
    print(f"Arriba->Abajo: {counter_horizontal.count_forward} | Abajo->Arriba: {counter_horizontal.count_backward}")
    print(f"Der->Izq: {counter_left.count_backward} | Izq->Der: {counter_right.count_forward}")
    return frame_idx, elapsed, out_path