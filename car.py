# trackers/basic_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import math
import cv2
import numpy as np
from utilities import *

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

@dataclass
class Car:
    track_id: int
    bbox: BBox
    first_bbox: BBox
    confidence: float
    speed_x: float = 0
    speed_y: float = 0
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    hits: int = 1                 # nº de veces emparejado con una detección
    lost: int = 0                 # nº de frames consecutivos sin ser emparejado
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    hsv_hist: Optional[np.ndarray] = None
    grad_hist: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray, bbox: BBox, confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.hits += 1
        self.lost = 0
        self.centroids.append(bbox_center(bbox))
        # (opcional) refrescar histograma cada n actualizaciones para ahorrar coste
        if self.hsv_hist is None or (self.hits % 10 == 0):
            self.hsv_hist = compute_hsv_hist(frame, bbox)
        if self.grad_hist is None or (self.hits % 10 == 0):
            self.grad_hist = compute_grad_hist(frame, bbox)
        self.speed_x, self.speed_y = self.calc_speed()

    def mark_missed(self):
        self.lost += 1

    def calc_speed(self):
        x1, y1 = self.centroids[-1]
        x2, y2 = self.centroids[-2]
        vx = x1 - x2
        vy = y1 - y2
        return vx, vy

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    def current_direction(self) -> Optional[str]:
        """
        Devuelve 'up', 'down' o None según el desplazamiento vertical predominante.
        Pensado para calles verticales (y crece hacia abajo en imágenes).
        """
        if len(self.centroids) < 2:
            return None
        # vector medio de los últimos k desplazamientos
        k = min(5, len(self.centroids) - 1)
        dy = 0.0
        for i in range(-k, -1):
            y_prev = self.centroids[i][1]
            y_next = self.centroids[i + 1][1]
            dy += (y_next - y_prev)
        if abs(dy) < 1e-3:
            return None
        return "down" if dy > 0 else "up"