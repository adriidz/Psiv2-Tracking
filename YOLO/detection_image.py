import os
from pathlib import Path
import argparse
import sys

# Force Ultralytics to use this folder for weights/runs (set BEFORE importing ultralytics)
os.environ["ULTRALYTICS_HOME"] = str(Path(__file__).resolve().parent)

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

CAR_CLASS_ID = 2  # COCO: 2 = car

# Pin outputs and weights inside the YOLO directory
YOLO_DIR = Path(__file__).resolve().parent
SETTINGS["runs_dir"] = str(YOLO_DIR / "runs")
SETTINGS["weights_dir"] = str(YOLO_DIR / "weights")
Path(SETTINGS["runs_dir"]).mkdir(parents=True, exist_ok=True)
Path(SETTINGS["weights_dir"]).mkdir(parents=True, exist_ok=True)

# Default image path (no CLI flag needed)
IMAGE_PATH = Path(r"Psiv2-Tracking\images\image.png")  # change this

def main():
    p = argparse.ArgumentParser()
    # Force download/lookup to YOLO/weights/yolo11n.pt by default
    p.add_argument("--weights", type=str, default=str(YOLO_DIR / "weights" / "yolo11n.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    args = p.parse_args()

    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}\nUpdate IMAGE_PATH in detection.py.")
        sys.exit(1)

    model = YOLO(args.weights)
    results = model.predict(
        source=str(IMAGE_PATH),
        conf=args.conf,
        imgsz=args.imgsz,
        classes=[CAR_CLASS_ID],
        save=True,
        project=SETTINGS["runs_dir"],
        name="cars",
        exist_ok=True
    )

    r = results[0]
    print(f"Using image: {IMAGE_PATH}")
    print(f"Cars detected: {len(r.boxes)}")
    print(f"Annotated image(s) saved to: {r.save_dir}")
    for b in r.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        print(f"{r.names[cls_id]} conf={conf:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

if __name__ == "__main__":
    main()