import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Must be set before importing ultralytics
os.environ["ULTRALYTICS_HOME"] = str(Path(__file__).resolve().parent)

import cv2
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

CAR_CLASS_ID = 2  # COCO: 2 = car

YOLO_DIR = Path(__file__).resolve().parent
SETTINGS["runs_dir"] = str(YOLO_DIR / "runs")
SETTINGS["weights_dir"] = str(YOLO_DIR / "weights")
Path(SETTINGS["runs_dir"]).mkdir(parents=True, exist_ok=True)
Path(SETTINGS["weights_dir"]).mkdir(parents=True, exist_ok=True)

VIDEO_PATH = Path(r"Psiv2-Tracking\videos\output7.mp4")  # change this

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

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=str(YOLO_DIR / "weights" / "yolo11n.pt"))
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--skip", type=int, default=20, help="Run inference every N frames")
    p.add_argument("--display", action="store_true", default=True, help="Show window (press Q to quit)")  # default ON
    p.add_argument("--reuse-last", action="store_true", default=True, help="Draw last detections on skipped frames")  # default True
    # removed: --playback and --persist (use constants above)
    args = p.parse_args()

    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO_PATH))  # ensure string path
    if not cap.isOpened():
        print(f"Could not open video source: {VIDEO_PATH}")
        sys.exit(1)

    # Prepare output video writer
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
        # Fallback to AVI if MP4 encoder unavailable
        out_path = out_dir / f"cars_{ts}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))

    # Real-time pacing base
    frame_period = 1.0 / (fps_in if fps_in > 0 else 30.0)
    next_frame_ts = time.perf_counter() + frame_period

    model = YOLO(args.weights)

    if args.display:
        cv2.namedWindow("cars", cv2.WINDOW_NORMAL)

        # Fit window to 80% of screen while preserving the video's aspect ratio
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
        cv2.resizeWindow("cars", win_w, win_h)

        # Keep aspect ratio even if the user resizes the window
        try:
            cv2.setWindowProperty("cars", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except cv2.error:
            pass

    last_result = None
    last_age = 0  # frames since last inference
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.skip <= 1 or frame_idx % args.skip == 0:
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                classes=[CAR_CLASS_ID],
                verbose=False
            )
            last_result = results[0] if results else None
            annotated = draw_boxes(frame.copy(), last_result)
            last_age = 0
            if last_result is not None:
                print(f"Frame {frame_idx}: {len(last_result.boxes)} cars")
        else:
            annotated = frame.copy()
            last_age += 1
            if args.reuse_last and last_result is not None and last_age <= PERSIST_FRAMES:
                annotated = draw_boxes(annotated, last_result, label_suffix="*")

        writer.write(annotated)

        if args.display:
            if window_closed("cars"):
                break
            cv2.imshow("cars", annotated)

            if PLAYBACK == "1x":
                now = time.perf_counter()
                delay_s = max(0.0, next_frame_ts - now)
                delay_ms = max(1, int(delay_s * 1000))
                key = cv2.waitKey(delay_ms) & 0xFF
                next_frame_ts += frame_period
            else:
                key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"Frames: {frame_idx} | Elapsed: {elapsed:.1f}s | Out: {out_path}")

if __name__ == "__main__":
    main()