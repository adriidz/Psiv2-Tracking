# trackers/basic_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import math
import cv2
import numpy as np
from typing import Any

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

def compute_grad_hist(frame: np.ndarray, bbox: Tuple[int, int, int, int], bins: int = 9) -> np.ndarray:
    """
    Calcula un histograma de orientaciones de gradiente (HOG simplificado)
    dentro de una bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)
    patch = frame[y1:y2, x1:x2]

    # Convertir a escala de grises
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Gradientes Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitud y orientación
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Histograma de orientaciones (0–180 grados)
    hist, _ = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)
    hist = cv2.normalize(hist, hist).flatten()  # normalizamos
    return hist

# ---------------- Additional scoring utilities for hybrid tracker ----------------

def predict_center(track: Any) -> Tuple[float, float]:
    """Predict next centroid using last motion; falls back to last center."""
    if hasattr(track, 'centroids') and len(track.centroids) >= 2:
        (x_prev, y_prev) = track.centroids[-2]
        (x_last, y_last) = track.centroids[-1]
        vx = x_last - x_prev
        vy = y_last - y_prev
        return (x_last + 1.1 * vx * track.lost, y_last + 1.1 * vy * track.lost)
    if hasattr(track, 'centroids') and len(track.centroids) == 1:
        return track.centroids[-1]
    # fallback to bbox center
    return bbox_center(track.bbox)

def predict_bbox(track):
    """
    Predice el siguiente bbox desplazando el último bbox según la velocidad calculada
    a partir de los dos últimos centroides.
    """
    # # Si hay al menos dos centroides, podemos calcular la velocidad
    # if hasattr(track, 'centroids'):
    #     if len(track.centroids) >= 2:
    #         (x_prev, y_prev) = track.centroids[-2]
    #         (x_last, y_last) = track.centroids[-1]

    #         # Velocidad (diferencia entre los dos últimos centros)
    #         vx = x_last - x_prev
    #         vy = y_last - y_prev

    #         # Factor según cuántos frames lleva perdido
    #         factor = 1.1 * (track.lost + 1) if hasattr(track, 'lost') else 1.0

    #         # Desplazamiento total
    #         dx = vx * factor
    #         dy = vy * factor

    #         # Último bbox
    #         x1, y1, x2, y2 = track.bbox

    #         # Simplemente trasladamos el bbox completo
    #         new_bbox = [int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)]
    #         return new_bbox
        
    if hasattr(track, 'centroids') and len(track.centroids) >= 2:
        # Usar últimos N centroides para estimar velocidad
        n = min(4, len(track.centroids))
        recent = track.centroids[-n:]
        
        # Calcular velocidad promedio
        vx_total = 0.0
        vy_total = 0.0
        for i in range(1, len(recent)):
            vx_total += recent[i][0] - recent[i-1][0]
            vy_total += recent[i][1] - recent[i-1][1]
        
        vx = vx_total / n
        vy = vy_total / n

        # Factor según cuántos frames lleva perdido
        factor = 1.1 * (track.lost + 1) if hasattr(track, 'lost') else 1.0
        # Predecir posición
        # Desplazamiento total
        dx = vx * factor
        dy = vy * factor

        # Último bbox
        x1, y1, x2, y2 = track.bbox

        # Simplemente trasladamos el bbox completo
        new_bbox = [int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)]
        return new_bbox   


    # Si solo tiene un centroide o bbox, lo devolvemos sin cambios
    return track.bbox

def aspect_score(track: Any, det_bbox: BBox) -> float:
    tw = track.bbox[2] - track.bbox[0]
    th = track.bbox[3] - track.bbox[1]
    dw = det_bbox[2] - det_bbox[0]
    dh = det_bbox[3] - det_bbox[1]
    if th <= 0 or dh <= 0:
        return 0.0
    ar_t = tw / max(1e-6, th)
    ar_d = dw / max(1e-6, dh)
    score = 1.0 - abs(ar_t - ar_d) / max(ar_t, ar_d)
    return float(max(0.0, min(1.0, score)))

def distance_score(track: Any, det_center: Tuple[float, float]) -> float:
    pred = predict_center(track)
    dx = pred[0] - det_center[0]
    dy = pred[1] - det_center[1]
    dist = math.hypot(dx, dy)
    w = max(1.0, (track.bbox[2] - track.bbox[0]))
    h = max(1.0, (track.bbox[3] - track.bbox[1]))
    diag = math.hypot(w, h)
    max_dist = max(1.0, 5.0 * diag)
    score = max(0.0, 1.0 - (dist / max_dist))
    return float(score)

def direction_score(track: Any, det_center: Tuple[float, float]) -> float:
    if not hasattr(track, 'centroids') or len(track.centroids) < 2:
        return 0.5
    (x_prev, y_prev) = track.centroids[-2]
    (x_last, y_last) = track.centroids[-1]
    vx = x_last - x_prev
    vy = y_last - y_prev
    ux = det_center[0] - x_last
    uy = det_center[1] - y_last
    norm_v = math.hypot(vx, vy)
    norm_u = math.hypot(ux, uy)
    if norm_v == 0 or norm_u == 0:
        return 0.5
    cosang = (vx * ux + vy * uy) / (norm_v * norm_u)
    score = (cosang + 1.0) / 2.0
    return float(max(0.0, min(1.0, score)))

def appearance_score(frame: np.ndarray, track: Any, det_bbox: BBox) -> float:
    """HSV histogram correlation mapped to [0,1]. Returns 0.0 if not available."""
    try:
        det_hist = compute_hsv_hist(frame, det_bbox)
    except Exception:
        return 0.0
    hsv_hist = getattr(track, 'hsv_hist', None)
    if hsv_hist is None:
        return 0.0
    corr = cv2.compareHist(hsv_hist.astype('float32'), det_hist.astype('float32'), cv2.HISTCMP_CORREL)
    corr = max(-1.0, min(1.0, float(corr)))
    return float(min(1.0, corr))

def shape_score(frame: np.ndarray, track: Any, det_bbox: Tuple[int, int, int, int]) -> float:
    """
    Compara la similitud de forma entre el objeto del track y la detección
    usando histogramas de orientaciones de gradiente.
    Devuelve un score entre 0 y 1.
    """
    try:
        det_hist = compute_grad_hist(frame, det_bbox)
    except Exception:
        return 0.0

    grad_hist = getattr(track, 'grad_hist', None)
    if grad_hist is None:
        return 0.0

    corr = cv2.compareHist(grad_hist.astype('float32'), det_hist.astype('float32'), cv2.HISTCMP_CORREL)
    corr = max(-1.0, min(1.0, float(corr)))

    # Reescalar de [-1,1] a [0,1]
    score = (corr + 1.0) / 2.0
    return float(max(0.0, min(1.0, score)))