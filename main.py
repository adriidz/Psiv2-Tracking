
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from detection_frames import *

VIDEO_PATH = Path(r"videos\output7.mp4")

def main():
    args = parse_args()

    try:
        cap = open_capture(VIDEO_PATH)
    except Exception as e:
        print(e)
        sys.exit(1)

    writer, out_path, width, height, fps_in = prepare_writer(cap)
    model = init_model(args.weights)
    setup_display_if_needed(args.display, width, height)

    tracker = Trackermalo(iou_threshold=0.15, max_lost=15, min_hits=1)

    process_frames(cap, writer, model, args, width, height, fps_in, out_path, tracker)

if __name__ == "__main__":
    main()