# trackers/basic_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import math
import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

# ----------------------------- Utilidades -----------------------------

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

# ------------------------------- Car ----------------------------------

@dataclass
class Car:
    track_id: int
    bbox: BBox
    confidence: float
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    hits: int = 1                 # nº de veces emparejado con una detección
    lost: int = 0                 # nº de frames consecutivos sin ser emparejado
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    hsv_hist: Optional[np.ndarray] = None

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

    def mark_missed(self):
        self.lost += 1

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

# ------------------------------ Tracker -------------------------------

class Tracker:
    """
    Tracker sencillo basado en IoU.
    - iou_threshold: mínimo IoU para asociar detecciones con tracks existentes.
    - max_lost: nº de frames que un track puede estar sin detección antes de eliminarse.
    - min_hits: nº de emparejamientos requeridos para considerar un track 'confiable' (puede usarse en la fase de conteo).
    """
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 15, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks: Dict[int, Car] = {}

    def _create_track(self, frame: np.ndarray, bbox: BBox, conf: float) -> Car:
        t = Car(track_id=self._next_id, bbox=bbox, confidence=conf)
        t.centroids.append(bbox_center(bbox))
        t.hsv_hist = compute_hsv_hist(frame, bbox)
        self.tracks[self._next_id] = t
        self._next_id += 1
        return t

    def _match(self, detections: List[Tuple[BBox, float]]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Empareja tracks existentes con detecciones por IoU (greedy).
        returns:
            assignments: dict {track_id -> det_idx}
            unassigned_tracks: list[track_id]
            unassigned_dets: list[det_idx]
        """
        track_ids = list(self.tracks.keys())
        if not track_ids or not detections:
            return {}, track_ids, list(range(len(detections)))

        # Matriz IoU [num_tracks x num_dets]
        iou_mat = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for ti, tid in enumerate(track_ids):
            tb = self.tracks[tid].bbox
            for di, (db, _) in enumerate(detections):
                iou_mat[ti, di] = iou(tb, db)

        assignments: Dict[int, int] = {}
        used_tracks = set()
        used_dets = set()

        # Asignación greedy por IoU máximo
        while True:
            ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            best = iou_mat[ti, di]
            if best < self.iou_threshold:
                break
            if ti in used_tracks or di in used_dets:
                iou_mat[ti, di] = -1  # invalidar y continuar
                continue
            track_id = track_ids[ti]
            assignments[track_id] = di
            used_tracks.add(ti)
            used_dets.add(di)
            iou_mat[ti, :] = -1
            iou_mat[:, di] = -1

        unassigned_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in used_tracks]
        unassigned_dets = [i for i in range(len(detections)) if i not in used_dets]
        return assignments, unassigned_tracks, unassigned_dets

    def update(self, frame: np.ndarray, detections: List[Tuple[BBox, float]]) -> Dict[int, Car]:
        """
        Actualiza el conjunto de tracks con las detecciones del frame actual.
        detections: lista de (bbox, conf) con bbox=(x1,y1,x2,y2)
        Devuelve un dict {track_id: Car} con los tracks vigentes tras la actualización.
        """
        # 1) Emparejar
        assignments, un_tracks, un_dets = self._match(detections)

        # 2) Actualizar tracks emparejados
        for track_id, det_idx in assignments.items():
            bbox, conf = detections[det_idx]
            self.tracks[track_id].update(frame, bbox, conf)

        # 3) Marcar como perdidos los no emparejados
        for tid in un_tracks:
            self.tracks[tid].mark_missed()

        # 4) Crear nuevos tracks para detecciones no emparejadas
        for det_idx in un_dets:
            bbox, conf = detections[det_idx]
            self._create_track(frame, bbox, conf)

        # 5) Eliminar tracks vencidos
        to_delete = [tid for tid, t in self.tracks.items() if t.lost > self.max_lost]
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks

    # ------------------------- Helpers visualización -------------------------

    def draw_tracks(self, frame: np.ndarray, min_hits: Optional[int] = None) -> np.ndarray:
        """Dibuja bbox + ID + dirección. min_hits permite ocultar tracks muy recientes."""
        if min_hits is None:
            min_hits = self.min_hits
        for t in self.tracks.values():
            if t.hits < min_hits:
                continue
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
            direction = t.current_direction()
            lbl = f"ID {t.track_id}"
            if direction:
                lbl += f" ({direction})"
            cv2.putText(frame, lbl, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
            # trayectoria (últimos puntos)
            if len(t.centroids) >= 2:
                for p, q in zip(t.centroids[-15:-1], t.centroids[-14:]):
                    cv2.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0, 215, 255), 2)
        return frame

# ------------------- Integración con Ultralytics YOLO -------------------

def yolo_result_to_detections(result) -> List[Tuple[BBox, float]]:
    """
    Convierte un 'result' de Ultralytics a lista de (bbox, conf).
    Solo usa 'boxes' presentes en result (ya filtradas por clase en tu inferencia).
    """
    dets: List[Tuple[BBox, float]] = []
    if result is None or result.boxes is None:
        return dets
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf)
        dets.append(((x1, y1, x2, y2), conf))
    return dets
