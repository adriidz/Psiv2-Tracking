# trackers/basic_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import math
import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

def iou(boxA: BBox, boxB: BBox) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-9)

def bbox_center(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def compute_hsv_hist(frame: np.ndarray, bbox: BBox, bins: int = 16) -> np.ndarray:
    """Histograma HSV normalizado (H,S,V) concatenado (bins por canal)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((bins*3,), dtype=np.float32)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist