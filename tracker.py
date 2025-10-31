# trackers/basic_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import math
import cv2
import numpy as np
from car import Car
from utilities import *
from utilities import predict_center, distance_score, aspect_score, direction_score, appearance_score
import random

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

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
        t = Car(track_id=self._next_id, bbox=bbox, confidence=conf, first_bbox=bbox)
        t.centroids.append(bbox_center(bbox))
        # t.hsv_hist = compute_hsv_hist(frame, bbox)
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
            # no dibujar si lleva más de 8 frames perdido (no afecta a la lógica de tracking)
            if t.lost > 8:
                continue
            if t.hits < min_hits:
                continue

            x1, y1, x2, y2 = t.bbox

            # color según estado
            if t.lost > 0: color = (0, 0, 255)     # rojo para tracks perdidos recientemente
            else: color = (0, 215, 255)   # amarillo para activos

            # dibujar bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # dirección y etiqueta
            direction = t.current_direction()
            lbl = f"ID {t.track_id}"
            if direction: lbl += f" ({direction})"

            # etiqueta sobre la caja
            cv2.putText(frame, lbl, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # trayectoria (últimos puntos)
            if len(t.centroids) >= 2:
                for p, q in zip(t.centroids[-15:-1], t.centroids[-14:]):
                    cv2.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 2)

        return frame

    def draw_prediction(self, frame: np.ndarray, bbox, min_hits: Optional[int] = None) -> np.ndarray:
        """Dibuja bbox + ID + dirección. min_hits permite ocultar tracks muy recientes."""
        if min_hits is None:
            min_hits = self.min_hits
        for t in self.tracks.values():
            # no dibujar si lleva más de 8 frames perdido (no afecta a la lógica de tracking)
            if t.lost > 8:
                continue
            if t.hits < min_hits:
                continue

            x1, y1, x2, y2 = bbox

            # color según estado
            if t.lost > 0: color = (255, 0, 0)     # rojo para tracks perdidos recientemente
            else: color = (0, 215, 255)   # amarillo para activos

            # dibujar bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # dirección y etiqueta
            direction = t.current_direction()
            lbl = f"ID {t.track_id}"
            if direction: lbl += f" ({direction})"

            # etiqueta sobre la caja
            cv2.putText(frame, lbl, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # trayectoria (últimos puntos)
            if len(t.centroids) >= 2:
                for p, q in zip(t.centroids[-15:-1], t.centroids[-14:]):
                    cv2.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 2)

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


class TrackerHíbrido(Tracker):
    def __init__(self,
                 iou_threshold: float = 0.3,
                 max_lost: int = 15,
                 min_hits: int = 1,
                 appearance_threshold: float = 0.6,
                 cascade_threshold: float = 0.45,
                 weights: dict = None,
                 debug: bool = False):
        """
        weights: diccionario con pesos para cada heurística (suma 1.0).
                 keys: 'appearance', 'distance', 'aspect', 'direction'
        appearance_threshold: umbral mínimo para considerar similitud de apariencia (opcional)
        cascade_threshold: umbral compuesto para aceptar una asociación en la fase heurística
        debug: imprime info de debug si True
        """
        super().__init__(iou_threshold=iou_threshold, max_lost=max_lost, min_hits=min_hits)
        self.appearance_threshold = appearance_threshold
        self.cascade_threshold = cascade_threshold
        self.debug = debug
        if weights is None:
            self.weights = {'appearance': 0.3, 'distance': 0.4, 'aspect': 0.1, 'direction': 0.2}
        else:
            self.weights = weights


    def _match(self, detections: List[Tuple[BBox, float]], frame: Optional[np.ndarray] = None):
        """
        Primero hace match por IoU (método base). Después intenta emparejar
        los tracks y detecciones que quedaron sin asignar usando heurísticas.
        Nota: frame es requerido para la similitud de apariencia; si no se proporciona,
              omitimos la similitud de apariencia.
        """
        # -- 1) Fase IoU usando implementación de la clase base --
        base_assignments, un_tracks, un_dets = super()._match(detections)

        # Convertir un_tracks a lista mutable y un_dets a lista mutable
        remaining_tracks = list(un_tracks)
        remaining_dets = list(un_dets)

        # Precalcular centros de detecciones
        det_centers = {}
        for di in remaining_dets:
            db, _ = detections[di]
            det_centers[di] = bbox_center(db)

        # Greedy matching sobre score compuesto
        extra_assignments: Dict[int, int] = {}
        used_tracks = set()
        used_dets = set()

        # Construir lista de candidatos con score
        candidate_scores = []
        for tid in remaining_tracks:
            track = self.tracks.get(tid)
            if track is None:
                continue
            for di in remaining_dets:
                if di in used_dets:
                    continue
                db, _ = detections[di]
                center = det_centers[di]

                # calculo de sub-scores
                app_score = 0.0
                if frame is not None:
                    app_score = appearance_score(frame, track, db)
                dist_score = distance_score(track, center)
                asp_score = aspect_score(track, db)
                dir_score = direction_score(track, center)

                # compuesta por pesos
                w = self.weights
                composite = (w['appearance'] * app_score +
                             w['distance'] * dist_score +
                             w['aspect'] * asp_score +
                             w['direction'] * dir_score)

                candidate_scores.append((composite, tid, di, app_score, dist_score, asp_score, dir_score))

        # Ordenar candidatos por score desc
        candidate_scores.sort(key=lambda x: x[0], reverse=True)

        for composite, tid, di, app_s, dist_s, asp_s, dir_s in candidate_scores:
            if tid in used_tracks or di in used_dets:
                continue
            if composite < self.cascade_threshold:
                # al ser ordenado, si este candidato no alcanza umbral, los siguientes tampoco
                break
            # opcional: exigir cierta mínima similitud de apariencia si está configurada
            if frame is not None and self.appearance_threshold is not None:
                if app_s < min(0.0, self.appearance_threshold * 0):  # NO obligamos; línea placeholder
                    pass  # no forzamos descarte por apariencia aquí, lo dejamos para info
            # aceptar la asignación
            extra_assignments[tid] = di
            used_tracks.add(tid)
            used_dets.add(di)
            if self.debug:
                print(f"[Hybrid] Assign T{tid} <-> D{di} score={composite:.3f} (app={app_s:.2f}, dist={dist_s:.2f}, asp={asp_s:.2f}, dir={dir_s:.2f})")

        # Combinar asignaciones: las de IoU (base_assignments) tienen preferencia,
        # pero si por alguna razón un track aparece también en extra_assignments, preservamos la asignación IoU.
        final_assignments = dict(base_assignments)  # copia
        for tid, di in extra_assignments.items():
            if tid in final_assignments:
                continue
            # si la detección ya fue tomada por IoU, saltar
            if di in final_assignments.values():
                continue
            final_assignments[tid] = di

        # Recalcular listas de no asignados
        assigned_track_ids = set(final_assignments.keys())
        assigned_det_idxs = set(final_assignments.values())

        new_unassigned_tracks = [tid for tid in self.tracks.keys() if tid not in assigned_track_ids]
        new_unassigned_dets = [i for i in range(len(detections)) if i not in assigned_det_idxs]

        return final_assignments, new_unassigned_tracks, new_unassigned_dets


class Tracker_predict(Tracker):
    def __init__(self, iou_threshold = 0.3, max_lost = 15, min_hits = 1):
        super().__init__(iou_threshold, max_lost, min_hits)
        self.avg_speed = None
        self.speeds = np.array([])

    def _match(self, detections: List[Tuple[BBox, float]], frame) -> Tuple[Dict[int, int], List[int], List[int]]:
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
            tb = predict_bbox(self.tracks[tid])
            self.draw_prediction(frame, tb, self.min_hits)
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
        assignments, un_tracks, un_dets = self._match(detections, frame)

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
            self.speeds = np.append(self.speeds, self.tracks[tid].speed_x)
            self.avg_speed = np.mean(self.speeds)
            del self.tracks[tid]

        return self.tracks
    
    def _create_track(self, frame, bbox, conf):
        t =  super()._create_track(frame, bbox, conf)
        if self.avg_speed:
            t.speed = self.avg_speed
        return t
    

class Tracker_color(Tracker):
    def __init__(self, iou_threshold = 0.3, max_lost = 15, min_hits = 1, appearance_threshold=0.6):
        super().__init__(iou_threshold, max_lost, min_hits)
        self.appearance_threshold = appearance_threshold
    
    def _create_track(self, frame: np.ndarray, bbox: BBox, conf: float) -> Car:
        t = Car(track_id=self._next_id, bbox=bbox, confidence=conf, first_bbox=bbox)
        t.centroids.append(bbox_center(bbox))
        t.hsv_hist = compute_hsv_hist(frame, bbox)
        self.tracks[self._next_id] = t
        self._next_id += 1
        return t
    
    def _match(self, detections: List[Tuple[BBox, float]], frame) -> Tuple[Dict[int, int], List[int], List[int]]:
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
            tb = predict_bbox(self.tracks[tid])
            self.draw_prediction(frame, tb, self.min_hits)
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

        # Reasignacion por colores
        if unassigned_tracks and unassigned_dets:
            app_mat = np.zeros((len(unassigned_tracks), len(unassigned_dets)), dtype=np.float32)
            for ti, tid in enumerate(unassigned_tracks):
                track = self.tracks[tid]
                for dj, det_idx in enumerate(unassigned_dets):
                    db, _ = detections[det_idx]
                    app_mat[ti, dj] = appearance_score(frame, track, db)

            while True:
                ti, di = np.unravel_index(np.argmax(app_mat), app_mat.shape)
                best = app_mat[ti, di]
                if best < getattr(self, "appearance_threshold", 0.6):
                    break
                track_id = unassigned_tracks[ti]
                det_idx = unassigned_dets[di]
                assignments[track_id] = det_idx
                used_tracks.add(track_ids.index(track_id))
                used_dets.add(det_idx)
                app_mat[ti, :] = -1
                app_mat[:, di] = -1

            # Recalcular los no asignados tras esta segunda fase
            unassigned_tracks = [tid for tid in track_ids if tid not in assignments]
            unassigned_dets = [i for i in range(len(detections)) if i not in used_dets]

        return assignments, unassigned_tracks, unassigned_dets
    
    def update(self, frame: np.ndarray, detections: List[Tuple[BBox, float]]) -> Dict[int, Car]:
        """
        Actualiza el conjunto de tracks con las detecciones del frame actual.
        detections: lista de (bbox, conf) con bbox=(x1,y1,x2,y2)
        Devuelve un dict {track_id: Car} con los tracks vigentes tras la actualización.
        """
        # 1) Emparejar
        assignments, un_tracks, un_dets = self._match(detections, frame)

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
    
class Tracker_grad(Tracker):
    def __init__(self, iou_threshold = 0.3, max_lost = 15, min_hits = 1, shape_threshold=0.55):
        super().__init__(iou_threshold, max_lost, min_hits)
        self.shape_threshold = shape_threshold

    def _create_track(self, frame: np.ndarray, bbox: BBox, conf: float) -> Car:
        t = Car(track_id=self._next_id, bbox=bbox, confidence=conf, first_bbox=bbox)
        t.centroids.append(bbox_center(bbox))
        t.grad_hist = compute_grad_hist(frame, bbox)
        self.tracks[self._next_id] = t
        self._next_id += 1
        return t

    def _match(self, detections: List[Tuple[BBox, float]], frame) -> Tuple[Dict[int, int], List[int], List[int]]:
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
            tb = predict_bbox(self.tracks[tid])
            self.draw_prediction(frame, tb, self.min_hits)
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

        # Reasignacion por colores
        if unassigned_tracks and unassigned_dets:
            app_mat = np.zeros((len(unassigned_tracks), len(unassigned_dets)), dtype=np.float32)
            for ti, tid in enumerate(unassigned_tracks):
                track = self.tracks[tid]
                for dj, det_idx in enumerate(unassigned_dets):
                    db, _ = detections[det_idx]
                    app_mat[ti, dj] = shape_score(frame, track, db)

            while True:
                ti, di = np.unravel_index(np.argmax(app_mat), app_mat.shape)
                best = app_mat[ti, di]
                if best < getattr(self, "shape_threshold", 0.55):
                    break
                track_id = unassigned_tracks[ti]
                det_idx = unassigned_dets[di]
                assignments[track_id] = det_idx
                used_tracks.add(track_ids.index(track_id))
                used_dets.add(det_idx)
                app_mat[ti, :] = -1
                app_mat[:, di] = -1

            # Recalcular los no asignados tras esta segunda fase
            unassigned_tracks = [tid for tid in track_ids if tid not in assignments]
            unassigned_dets = [i for i in range(len(detections)) if i not in used_dets]

        return assignments, unassigned_tracks, unassigned_dets
    
    def update(self, frame: np.ndarray, detections: List[Tuple[BBox, float]]) -> Dict[int, Car]:
        """
        Actualiza el conjunto de tracks con las detecciones del frame actual.
        detections: lista de (bbox, conf) con bbox=(x1,y1,x2,y2)
        Devuelve un dict {track_id: Car} con los tracks vigentes tras la actualización.
        """
        # 1) Emparejar
        assignments, un_tracks, un_dets = self._match(detections, frame)

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