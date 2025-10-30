# improved_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math
import cv2
import numpy as np
from car import Car
from utilities import *

# Nueva importación para el Tracker de Kalman
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

BBox = Tuple[int, int, int, int]

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


class OcclusionTracker(Tracker):
    """
    Tracker simple con manejo de oclusiones mediante buffer.
    Cuando un ID desaparece, lo guarda en un buffer por N frames.
    Si aparece una nueva detección compatible con la trayectoria predicha, lo reasigna.
    """
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_lost: int = 30,
                 min_hits: int = 1,
                 buffer_frames: int = 25,          # Frames que mantiene IDs perdidos en buffer
                 search_radius: float = 100.0,     # Radio de búsqueda base en píxeles (MUY CONSERVADOR)
                 min_match_score: float = 0.50,    # Score mínimo para reasociar (MUY ESTRICTO)
                 skip_frames: int = 1,             # Frames saltados entre detecciones
                 fragment_iou: float = 0.1,        # IoU mínimo para considerar fragmentos
                 debug: bool = False):
        """
        Args:
            buffer_frames: Número de frames que mantiene tracks perdidos antes de eliminarlos
            search_radius: Radio base en píxeles para buscar detecciones
            min_match_score: Score mínimo para aceptar una reasociación (0-1)
            skip_frames: Cuántos frames se saltan entre detecciones (para ajustar predicción)
            fragment_iou: IoU mínimo entre detecciones cercanas para detectar fragmentación
        """
        super().__init__(iou_threshold, max_lost, min_hits)
        self.buffer_frames = buffer_frames
        self.search_radius = search_radius
        self.min_match_score = min_match_score
        self.skip_frames = skip_frames
        self.fragment_iou = fragment_iou
        self.debug = debug
    
    def _merge_fragments(self, detections: List[Tuple[BBox, float]]) -> List[Tuple[BBox, float]]:
        """
        Detecta y fusiona SOLO detecciones claramente fragmentadas.
        Criterios MUY ESTRICTOS para evitar fusionar coches diferentes.
        """
        if len(detections) <= 1 or self.fragment_iou <= 0.0:
            return detections  # Desactivado si fragment_iou = 0
        
        merged = []
        used = set()
        
        for i, (bbox1, conf1) in enumerate(detections):
            if i in used:
                continue
            
            # Buscar fragmentos cercanos
            fragments = [(i, bbox1, conf1)]
            
            for j, (bbox2, conf2) in enumerate(detections):
                if j <= i or j in used:
                    continue
                
                # Calcular IoU
                iou_val = iou(bbox1, bbox2)
                
                # Criterios MÁS ESTRICTOS:
                # 1. Debe haber ALGO de solapamiento (IoU > 0.05)
                # 2. Y estar MUY cerca
                if iou_val < 0.05:
                    continue
                
                # Distancia entre centros
                c1 = bbox_center(bbox1)
                c2 = bbox_center(bbox2)
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                
                # Tamaños similares (evitar fusionar coche completo con fragmento pequeño)
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                area_ratio = min(area1, area2) / max(area1, area2)
                
                # Solo fusionar si:
                # - Áreas similares (ratio > 0.3)
                # - Distancia pequeña (< menor dimensión)
                min_dim = min(bbox1[2] - bbox1[0], bbox1[3] - bbox1[1],
                             bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])
                
                if area_ratio > 0.3 and dist < min_dim * 0.8:
                    fragments.append((j, bbox2, conf2))
                    used.add(j)
            
            # Si encontramos fragmentos, fusionar
            if len(fragments) > 1:
                all_x1 = [b[0] for _, b, _ in fragments]
                all_y1 = [b[1] for _, b, _ in fragments]
                all_x2 = [b[2] for _, b, _ in fragments]
                all_y2 = [b[3] for _, b, _ in fragments]
                
                merged_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))
                merged_conf = max(c for _, _, c in fragments)
                
                merged.append((merged_bbox, merged_conf))
                used.add(i)
                
                if self.debug:
                    print(f"  [MERGE] {len(fragments)} fragmentos fusionados")
            else:
                merged.append((bbox1, conf1))
                used.add(i)
        
        return merged
    
    def _is_static(self, track: Car, threshold: float = 3.0) -> bool:
        """
        Determina si un track es estático (aparcado) basándose en movimiento.
        
        Args:
            threshold: Movimiento promedio en píxeles para considerar móvil
        """
        if len(track.centroids) < 4:
            return False  # Necesita historial
        
        # Calcular movimiento total en últimos 4-6 centroides
        n = min(6, len(track.centroids))
        recent = track.centroids[-n:]
        
        total_dist = 0.0
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            total_dist += math.hypot(dx, dy)
        
        avg_movement = total_dist / (len(recent) - 1)
        
        return avg_movement < threshold
    
    def _predict_position(self, track: Car, frames_ahead: int = 1) -> Tuple[float, float]:
        """
        Predice la posición futura del centroide basándose en velocidad reciente.
        Para objetos estáticos, devuelve la última posición conocida.
        
        Args:
            frames_ahead: Número de frames a predecir (útil cuando se saltan frames)
        """
        if len(track.centroids) < 2:
            return track.centroids[-1] if track.centroids else (0, 0)
        
        # Si es estático, no predecir movimiento
        if self._is_static(track):
            return track.centroids[-1]
        
        # Usar últimos N centroides para estimar velocidad
        n = min(5, len(track.centroids))
        recent = track.centroids[-n:]
        
        # Calcular velocidad promedio
        vx_total = 0.0
        vy_total = 0.0
        for i in range(1, len(recent)):
            vx_total += recent[i][0] - recent[i-1][0]
            vy_total += recent[i][1] - recent[i-1][1]
        
        vx = vx_total / (len(recent) - 1)
        vy = vy_total / (len(recent) - 1)
        
        # Predecir posición: usar frames_ahead en vez de track.lost
        # porque track.lost no refleja frames saltados
        predicted_x = track.centroids[-1][0] + vx * frames_ahead
        predicted_y = track.centroids[-1][1] + vy * frames_ahead
        
        return (predicted_x, predicted_y)
    
    def _compute_match_score(self, 
                            frame: np.ndarray,
                            track: Car, 
                            det_bbox: BBox) -> float:
        """
        Calcula un score BALANCEADO para emparejar track con detección.
        VALIDACIONES ESTRICTAS para evitar saltos imposibles.
        """
        det_center = bbox_center(det_bbox)
        
        # Detectar si es estático
        is_static = self._is_static(track)
        
        # Predecir considerando frames perdidos Y frames saltados
        frames_ahead = max(1, track.lost * (self.skip_frames + 1))
        predicted_pos = self._predict_position(track, frames_ahead)
        
        # 1. VALIDACIÓN CRÍTICA: Distancia física razonable
        dist = math.hypot(det_center[0] - predicted_pos[0], 
                         det_center[1] - predicted_pos[1])
        
        # Radio MUCHO MÁS ESTRICTO para prevenir saltos
        if is_static:
            # Estáticos: radio MUY pequeño
            dynamic_radius = min(self.search_radius * 0.3, 40.0)
        elif track.lost == 1:
            # Perdido 1 frame: radio pequeño
            dynamic_radius = self.search_radius * 0.8
        elif track.lost == 2:
            # Perdido 2 frames: radio moderado
            dynamic_radius = self.search_radius * 1.0
        else:
            # Perdidos 3+: radio crece poco a poco
            growth_factor = min(1.4, 1.0 + track.lost * 0.08)
            dynamic_radius = self.search_radius * growth_factor
        
        if dist > dynamic_radius:
            if self.debug:
                print(f"  REJECT T{track.track_id}: dist={dist:.0f}px > radius={dynamic_radius:.0f}px")
            return 0.0
        
        # 2. VALIDACIÓN CRÍTICA: Coherencia de velocidad
        if len(track.centroids) >= 2:
            # Calcular velocidad previa (usar más puntos para más estabilidad)
            if len(track.centroids) >= 3:
                recent = track.centroids[-3:]
                prev_speed = math.hypot(recent[-1][0] - recent[0][0],
                                       recent[-1][1] - recent[0][1]) / 2.0
            else:
                last_two = track.centroids[-2:]
                prev_speed = math.hypot(last_two[1][0] - last_two[0][0],
                                       last_two[1][1] - last_two[0][1])
            
            # Velocidad implícita hacia la detección
            implied_speed = dist / max(1, frames_ahead)
            
            # VALIDACIÓN MUY ESTRICTA: Cambio de velocidad limitado
            if prev_speed > 1.0:  # Si hay movimiento previo
                # Tracks establecidos: cambio mínimo
                if track.hits >= 8:
                    max_speed_change = 2.0  # Solo 2x cambio
                elif track.hits >= 5:
                    max_speed_change = 2.5  # 2.5x cambio
                else:
                    max_speed_change = 3.0  # Tracks nuevos: 3x
                
                if implied_speed > prev_speed * max_speed_change:
                    if self.debug:
                        print(f"  REJECT T{track.track_id}: velocidad implícita {implied_speed:.1f} "
                              f"> {max_speed_change}x velocidad previa {prev_speed:.1f}")
                    return 0.0
                
                # También rechazar desaceleración brusca (frenar de golpe)
                if implied_speed < prev_speed * 0.3 and prev_speed > 5.0:
                    if self.debug:
                        print(f"  REJECT T{track.track_id}: desaceleración brusca "
                              f"{implied_speed:.1f} < 0.3x {prev_speed:.1f}")
                    return 0.0
            
            # Para objetos estáticos, no permitir saltos
            if is_static and dist > 30:
                if self.debug:
                    print(f"  REJECT T{track.track_id}: objeto estático no puede saltar {dist:.0f}px")
                return 0.0
        
        dist_score = 1.0 - (dist / dynamic_radius)
        
        # 3. Score de apariencia (más importante ahora)
        app_score = 0.3  # Penalizar por defecto si no hay histograma
        try:
            det_hist = compute_hsv_hist(frame, det_bbox)
            if track.hsv_hist is not None and det_hist is not None:
                corr = cv2.compareHist(track.hsv_hist.astype('float32'), 
                                      det_hist.astype('float32'), 
                                      cv2.HISTCMP_CORREL)
                # Normalizar: correlación [-1, 1] -> score [0, 1]
                app_score = max(0.0, (corr + 1.0) / 2.0)
                
                # VALIDACIÓN: Rechazar si apariencia es MUY diferente
                if track.hits >= 5 and corr < -0.3:  # Tracks establecidos
                    if self.debug:
                        print(f"  REJECT T{track.track_id}: apariencia muy diferente (corr={corr:.2f})")
                    return 0.0
        except:
            pass
        
        # 4. Score de dirección - VALIDACIÓN MUY ESTRICTA
        dir_score = 0.5
        if len(track.centroids) >= 3:  # Necesita más historial
            # Usar últimos 4 puntos para dirección más estable
            n = min(4, len(track.centroids))
            recent = track.centroids[-n:]
            prev_vx = recent[-1][0] - recent[0][0]
            prev_vy = recent[-1][1] - recent[0][1]
            prev_speed = math.hypot(prev_vx, prev_vy)
            
            curr_vx = det_center[0] - track.centroids[-1][0]
            curr_vy = det_center[1] - track.centroids[-1][1]
            curr_speed = math.hypot(curr_vx, curr_vy)
            
            if prev_speed > 2.0 and curr_speed > 2.0:  # Solo si hay movimiento claro
                cos_angle = (prev_vx * curr_vx + prev_vy * curr_vy) / (prev_speed * curr_speed)
                
                # VALIDACIÓN MUY ESTRICTA: Limitar cambios de dirección
                if track.hits >= 8:
                    # Tracks muy establecidos: máximo 75° de cambio
                    min_cos = 0.259  # cos(75°)
                    if cos_angle < min_cos:
                        if self.debug:
                            angle_deg = math.acos(max(-1, min(1, cos_angle))) * 180 / math.pi
                            print(f"  REJECT T{track.track_id}: cambio dirección {angle_deg:.0f}° (track muy establecido)")
                        return 0.0
                elif track.hits >= 5:
                    # Tracks establecidos: máximo 85° de cambio
                    min_cos = 0.087  # cos(85°)
                    if cos_angle < min_cos:
                        if self.debug:
                            angle_deg = math.acos(max(-1, min(1, cos_angle))) * 180 / math.pi
                            print(f"  REJECT T{track.track_id}: cambio dirección {angle_deg:.0f}° (track establecido)")
                        return 0.0
                else:
                    # Tracks nuevos: máximo 100° de cambio
                    min_cos = -0.174  # cos(100°)
                    if cos_angle < min_cos:
                        if self.debug:
                            angle_deg = math.acos(max(-1, min(1, cos_angle))) * 180 / math.pi
                            print(f"  REJECT T{track.track_id}: cambio dirección {angle_deg:.0f}°")
                        return 0.0
                
                # Score proporcional al alineamiento (favorece movimiento recto)
                dir_score = max(0.0, (cos_angle + 1.0) / 2.0)
            
            # VALIDACIÓN ADICIONAL: Rechazar movimientos perpendiculares
            # Si el track va en una dirección y la detección está perpendicular
            if prev_speed > 5.0:  # Solo para tracks con movimiento claro
                # Calcular distancia perpendicular a la trayectoria
                # Vector unitario de la dirección previa
                ux = prev_vx / prev_speed
                uy = prev_vy / prev_speed
                
                # Vector desde última posición a detección
                dx = det_center[0] - track.centroids[-1][0]
                dy = det_center[1] - track.centroids[-1][1]
                
                # Proyección perpendicular
                perp_dist = abs(-uy * dx + ux * dy)
                
                # Si la detección está muy perpendicular (lateral), rechazar
                if perp_dist > self.search_radius * 0.6:
                    if self.debug:
                        print(f"  REJECT T{track.track_id}: detección muy perpendicular (perp={perp_dist:.0f}px)")
                    return 0.0
        
        # 5. Validación de tamaño MÁS ESTRICTA
        tw = track.bbox[2] - track.bbox[0]
        th = track.bbox[3] - track.bbox[1]
        dw = det_bbox[2] - det_bbox[0]
        dh = det_bbox[3] - det_bbox[1]
        
        if tw > 0 and th > 0:
            width_ratio = max(tw, dw) / min(tw, dw)
            height_ratio = max(th, dh) / min(th, dh)
            
            # RECHAZO: Tamaños muy diferentes
            # Para tracks establecidos, ser más estricto
            if track.hits >= 5:
                max_ratio = 1.6  # Solo 60% diferencia
            else:
                max_ratio = 1.8  # 80% diferencia para nuevos
            
            if width_ratio > max_ratio or height_ratio > max_ratio:
                if self.debug:
                    print(f"  REJECT T{track.track_id}: tamaño muy diferente (w={width_ratio:.1f}x, h={height_ratio:.1f}x)")
                return 0.0
            
            size_score = 1.0 - (abs(width_ratio - 1.0) + abs(height_ratio - 1.0)) / 4.0
        else:
            size_score = 0.5
        
        # Score compuesto con DISTANCIA como factor dominante
        # La distancia es el indicador más confiable
        score = 0.45 * dist_score + 0.25 * app_score + 0.20 * dir_score + 0.10 * size_score
        
        # VALIDACIÓN FINAL: Score mínimo absoluto incluso antes de threshold
        if score < 0.35:  # Rechazar scores muy bajos
            if self.debug:
                print(f"  REJECT T{track.track_id}: score total muy bajo ({score:.2f})")
            return 0.0
        
        if self.debug:
            print(f"  T{track.track_id}: dist={dist_score:.2f}({dist:.0f}px), "
                  f"app={app_score:.2f}, dir={dir_score:.2f}, size={size_score:.2f} -> {score:.2f}")
        
        return score
    
    def _match(self, detections: List[Tuple[BBox, float]], frame: Optional[np.ndarray] = None):
        """
        Matching BALANCEADO en 2 fases:
        1. IoU para tracks activos (estándar)
        2. Score-based para tracks perdidos (CON VALIDACIONES ESTRICTAS)
        """
        if not detections:
            return {}, list(self.tracks.keys()), []
        
        # FASE 0: Fusionar fragmentos (si está activado)
        detections = self._merge_fragments(detections)
        
        # Separar tracks activos y perdidos
        active_tracks = {tid: t for tid, t in self.tracks.items() if t.lost == 0}
        lost_tracks = {tid: t for tid, t in self.tracks.items() if t.lost > 0}
        
        assignments = {}
        used_dets = set()
        
        # ============ FASE 1: IoU para tracks activos ============
        if active_tracks:
            track_ids = list(active_tracks.keys())
            iou_mat = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            
            for ti, tid in enumerate(track_ids):
                tb = active_tracks[tid].bbox
                for di, (db, _) in enumerate(detections):
                    iou_mat[ti, di] = iou(tb, db)
            
            # Asignación greedy por IoU
            while True:
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                best_iou = iou_mat[ti, di]
                
                if best_iou < self.iou_threshold:
                    break
                
                tid = track_ids[ti]
                assignments[tid] = di
                used_dets.add(di)
                iou_mat[ti, :] = -1
                iou_mat[:, di] = -1
        
        # ============ FASE 2: Re-identificación con VALIDACIONES ============
        if lost_tracks and frame is not None:
            candidates = []
            
            for tid, track in lost_tracks.items():
                # Filtro 1: Solo tracks con BUEN historial
                if track.hits < 4:  # Aumentado de 3 a 4
                    continue
                
                # Filtro 2: No reasignar tracks perdidos hace mucho
                if track.lost > 10:  # Reducido de 15 a 10
                    continue
                
                is_static = self._is_static(track)
                
                for di, (db, _) in enumerate(detections):
                    if di in used_dets:
                        continue
                    
                    # Calcular score (incluye validaciones internas)
                    score = self._compute_match_score(frame, track, db)
                    
                    if score == 0.0:
                        continue  # Rechazado por validaciones
                    
                    # Umbrales MÁS ESTRICTOS
                    if is_static:
                        threshold = self.min_match_score * 0.85  # Estáticos: algo más bajo
                    elif track.lost == 1:
                        threshold = self.min_match_score * 0.90  # Recién perdido: casi igual
                    elif track.lost == 2:
                        threshold = self.min_match_score * 0.95  # 2 frames: casi normal
                    else:
                        threshold = self.min_match_score * 1.05  # 3+: MÁS ESTRICTO
                    
                    if score >= threshold:
                        # Prioridad: score alto > establecidos > menos perdidos > estáticos
                        # (estáticos al final porque son más propensos a errores)
                        priority = (score, track.hits >= 8, -track.lost, is_static)
                        candidates.append((priority, score, tid, di, track.lost, is_static))
            
            # Ordenar por prioridad (score primero)
            candidates.sort(reverse=True)
            
            # Asignar (greedy por score)
            for _, score, tid, di, lost_frames, is_static in candidates:
                if tid in assignments or di in used_dets:
                    continue
                
                assignments[tid] = di
                used_dets.add(di)
                
                if self.debug:
                    status = "STATIC" if is_static else "mobile"
                    print(f"[ReID] T{tid} ({status}, hits={self.tracks[tid].hits}) <- Det{di} "
                          f"(score={score:.2f}, lost={lost_frames})")
        
        unassigned_tracks = [tid for tid in self.tracks.keys() if tid not in assignments]
        unassigned_dets = [i for i in range(len(detections)) if i not in used_dets]
        
        return assignments, unassigned_tracks, unassigned_dets
    
    def update(self, frame: np.ndarray, detections: List[Tuple[BBox, float]]) -> Dict[int, Car]:
        """
        Actualiza tracks con enfoque BALANCEADO.
        Prioridad: coherencia física > reutilizar IDs
        """
        # Matching en 2 fases (con validaciones estrictas)
        assignments, un_tracks, un_dets = self._match(detections, frame)
        
        # Actualizar tracks emparejados
        for track_id, det_idx in assignments.items():
            bbox, conf = detections[det_idx]
            self.tracks[track_id].update(frame, bbox, conf)
        
        # Marcar como perdidos los no emparejados
        for tid in un_tracks:
            self.tracks[tid].mark_missed()
        
        # ============ CREAR NUEVOS TRACKS ============
        # Crear directamente - las validaciones en _match ya son suficientemente estrictas
        for det_idx in un_dets:
            bbox, conf = detections[det_idx]
            new_track = self._create_track(frame, bbox, conf)
            if self.debug:
                print(f"[NEW] T{new_track.track_id} created")
        
        # Eliminar tracks que superaron el buffer
        to_delete = [tid for tid, t in self.tracks.items() 
                     if t.lost > self.buffer_frames]
        for tid in to_delete:
            if self.debug:
                print(f"[Delete] Track {tid} (lost for {self.tracks[tid].lost} frames)")
            del self.tracks[tid]
        
        return self.tracks
    
    def draw_tracks(self, frame: np.ndarray, min_hits: Optional[int] = None) -> np.ndarray:
        """
        Dibuja tracks de forma simple.
        Tracks activos en amarillo, tracks en buffer de oclusión en naranja.
        """
        if min_hits is None:
            min_hits = self.min_hits
        
        for t in self.tracks.values():
            # No dibujar tracks muy recientes
            if t.hits < min_hits:
                continue
            
            # No dibujar si lleva demasiado tiempo perdido
            if t.lost > 8:
                continue
            
            x1, y1, x2, y2 = t.bbox
            
            # Color según estado
            if t.lost > 0:
                color = (0, 100, 255)  # Naranja para perdidos
            else:
                color = (0, 215, 255)  # Amarillo para activos
            
            # Dibujar bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta con ID y dirección
            direction = t.current_direction()
            lbl = f"ID {t.track_id}"
            if direction:
                lbl += f" ({direction})"
            if t.lost > 0:
                lbl += f" [lost:{t.lost}]"
            
            cv2.putText(frame, lbl, (x1, max(0, y1 - 6)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            
            # Trayectoria (últimos 15 puntos)
            if len(t.centroids) >= 2:
                points = t.centroids[-15:]
                for p, q in zip(points[:-1], points[1:]):
                    cv2.line(frame, (int(p[0]), int(p[1])), 
                            (int(q[0]), int(q[1])), color, 2)
        
        return frame

# ============================================================================
# 🚀 KALMAN TRACKER - ENFOQUE BASADO EN MODELOS
# ============================================================================

def create_kalman_filter():
    """
    Crea un Filtro de Kalman OPTIMIZADO con modelo de velocidad constante.
    
    CAMBIOS CLAVE vs versión anterior:
    - Ruido de proceso (Q) MÁS ALTO: permite seguir aceleraciones
    - Ruido de medida (R) AJUSTADO: confía más en YOLO
    - Covarianza inicial (P) más conservadora
    
    Estado (dim_x=7): [x, y, w, h, vx, vy, vw] 
        - x, y: posición del centro
        - w, h: ancho y alto del bbox
        - vx, vy: velocidad del centro
        - vw: velocidad de cambio de escala (w y h escalan juntos)
    Medida (dim_z=4): [x, y, w, h] (medimos posición y tamaño desde YOLO)
    """
    kf = KalmanFilter(dim_x=7, dim_z=4)
    
    # Matriz de transición de estado (F)
    # Modelo de velocidad constante para posición y escala
    dt = 1.0  # dt=1 frame
    kf.F = np.array([
        [1, 0, 0, 0, dt, 0,  0],   # x' = x + vx*dt
        [0, 1, 0, 0, 0,  dt, 0],   # y' = y + vy*dt
        [0, 0, 1, 0, 0,  0,  dt],  # w' = w + vw*dt
        [0, 0, 0, 1, 0,  0,  dt],  # h' = h + vw*dt (misma velocidad de escala)
        [0, 0, 0, 0, 1,  0,  0],   # vx' = vx
        [0, 0, 0, 0, 0,  1,  0],   # vy' = vy
        [0, 0, 0, 0, 0,  0,  1]    # vw' = vw
    ], dtype=np.float32)
    
    # Matriz de medida (H) - mapea estado a medida
    # Medimos x, y, w, h directamente
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ], dtype=np.float32)
    
    # Covarianza del ruido del proceso (Q) - OPTIMIZADA PARA PREDICCIÓN
    # Balance: confiar en el modelo durante oclusiones
    kf.Q = np.eye(7, dtype=np.float32)
    kf.Q[:2, :2] *= 0.1     # 🔥 MUY bajo ruido en posición (confiar en predicción)
    kf.Q[2:4, 2:4] *= 0.1   # Bajo ruido en tamaño (cambio gradual)
    kf.Q[4:6, 4:6] *= 5.0   # 🔥 Ruido moderado en velocidad (permite cambios suaves)
    kf.Q[6, 6] *= 0.5       # Bajo ruido en velocidad de escala
    
    # Covarianza del ruido de la medida (R) - AJUSTADA
    # Confiar bastante en YOLO cuando está presente
    kf.R = np.eye(4, dtype=np.float32)
    kf.R[:2, :2] *= 5.0    # 🔥 Ruido moderado en posición 
    kf.R[2:, 2:] *= 15.0   # Más ruido en tamaño (YOLO menos preciso aquí)
    
    # Covarianza inicial del estado (P) - MODERADA
    kf.P = np.eye(7, dtype=np.float32)
    kf.P[:4, :4] *= 20.0    # Incertidumbre moderada inicial
    kf.P[4:6, 4:6] *= 100.0 # Alta incertidumbre en velocidades (hasta estimar bien)
    kf.P[6, 6] *= 50.0      # Incertidumbre moderada en escala
    
    return kf

class KalmanTracker(Tracker):
    """
    Tracker que utiliza Filtros de Kalman para predecir trayectorias y manejar oclusiones.
    
    MODO HÍBRIDO: Combina Kalman con validaciones físicas para mejor robustez.
    """
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 15, min_hits: int = 3,
                 mahalanobis_threshold: float = 3.5, debug: bool = False):
        super().__init__(iou_threshold, max_lost, min_hits)
        self.mahalanobis_threshold = mahalanobis_threshold
        self.debug = debug
        self.frame_count = 0

    def _create_track(self, frame: np.ndarray, bbox: BBox, conf: float) -> Car:
        """Crea un nuevo track y le asigna un Filtro de Kalman MEJORADO."""
        t = super()._create_track(frame, bbox, conf)
        
        # Inicializar filtro de Kalman
        t.kalman_filter = create_kalman_filter()
        
        # Estado inicial: [x, y, w, h, vx, vy, vw]
        center = bbox_center(bbox)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # Inicializar con velocidad cero
        t.kalman_filter.x = np.array([
            center[0],  # x
            center[1],  # y
            w,          # w
            h,          # h
            0.0,        # vx
            0.0,        # vy
            0.0         # vw
        ], dtype=np.float32)
        
        return t

    def _mahalanobis_distance(self, track: Car, det_bbox: BBox) -> float:
        """
        Calcula la distancia de Mahalanobis MEJORADA entre predicción y detección.
        Ahora considera posición Y tamaño, con validaciones físicas adicionales.
        """
        kf = track.kalman_filter
        if kf is None: 
            return np.inf
        
        # Convertir detección a formato [x, y, w, h]
        det_center = bbox_center(det_bbox)
        det_w = det_bbox[2] - det_bbox[0]
        det_h = det_bbox[3] - det_bbox[1]
        z = np.array([det_center[0], det_center[1], det_w, det_h], dtype=np.float32)
        
        # Predicción del estado
        predicted = kf.H @ kf.x
        
        # 🔥 VALIDACIÓN FÍSICA: Distancia euclidiana primero (rápida)
        pred_center_x, pred_center_y = predicted[:2]
        dist_euclidean = math.hypot(det_center[0] - pred_center_x, 
                                     det_center[1] - pred_center_y)
        
        # 🔥 RELAJADO: Distancia física más permisiva para oclusiones
        # Adaptar según tiempo perdido del track
        if hasattr(track, 'lost'):
            # Más tiempo perdido = mayor radio de búsqueda
            max_physical_dist = 150.0 + (track.lost * 15.0)  # Crece con tiempo perdido
            max_physical_dist = min(max_physical_dist, 400.0)  # Límite máximo
        else:
            max_physical_dist = 250.0  # píxeles por defecto
            
        if dist_euclidean > max_physical_dist:
            if self.debug:
                print(f"    [Reject] T{track.track_id}: dist={dist_euclidean:.0f}px > max={max_physical_dist:.0f}px")
            return np.inf
        
        # 🔥 RELAJADO: Validar cambio de tamaño más permisivo
        pred_w, pred_h = predicted[2:4]
        size_ratio_w = max(det_w, pred_w) / max(1, min(det_w, pred_w))
        size_ratio_h = max(det_h, pred_h) / max(1, min(det_h, pred_h))
        
        # Más permisivo: 3.0x en vez de 2.5x
        if size_ratio_w > 3.0 or size_ratio_h > 3.0:
            if self.debug:
                print(f"    [Reject] T{track.track_id}: size_ratio w={size_ratio_w:.1f}x, h={size_ratio_h:.1f}x")
            return np.inf  # Cambio de tamaño irreal
        
        # Proyectar la covarianza del estado en el espacio de medida
        H = kf.H
        P = kf.P
        S = H @ P @ H.T + kf.R
        
        # Innovación (diferencia entre medida y predicción)
        y = z - predicted
        
        # Distancia de Mahalanobis
        try:
            S_inv = np.linalg.inv(S)
            dist_sq = y.T @ S_inv @ y
            
            # Normalizar por número de dimensiones (chi-cuadrado)
            mahal_dist = np.sqrt(max(0, dist_sq))
            
            if self.debug and mahal_dist < self.mahalanobis_threshold:
                print(f"    [Mahal] T{track.track_id}: dist={mahal_dist:.2f} "
                      f"(eucl={dist_euclidean:.0f}px, size_ratio={size_ratio_w:.1f}x)")
            
            return mahal_dist
            
        except np.linalg.LinAlgError:
            # Si S no es invertible, usar distancia euclidiana normalizada
            fallback_dist = dist_euclidean / 50.0  # Normalizar
            if self.debug:
                print(f"    [Mahal] T{track.track_id}: FALLBACK euclidean={dist_euclidean:.0f}px")
            return fallback_dist

    def update(self, frame: np.ndarray, detections: List[Tuple[BBox, float]]) -> Dict[int, Car]:
        """
        Actualiza los tracks usando FILTRO DE KALMAN MEJORADO:
        1. Predicción: Todos los filtros predicen el siguiente estado
        2. Emparejamiento en 2 fases (IoU + Mahalanobis)
        3. Actualización: Corrección de Kalman con mediciones
        """
        self.frame_count += 1
        
        # 1. PREDICCIÓN para todos los tracks existentes
        for track in self.tracks.values():
            if track.kalman_filter:
                track.kalman_filter.predict()
                
                # Extraer estado predicho [x, y, w, h, vx, vy, vw]
                px, py, pw, ph = track.kalman_filter.x[:4]
                
                # VALIDACIÓN: Evitar valores absurdos
                pw = max(10, min(1000, pw))  # Ancho razonable
                ph = max(10, min(1000, ph))  # Alto razonable
                
                # Convertir a bbox predicho [x1, y1, x2, y2]
                track.predicted_bbox = (
                    int(px - pw/2), 
                    int(py - ph/2), 
                    int(px + pw/2), 
                    int(py + ph/2)
                )
                
                if self.debug and self.frame_count % 30 == 0:
                    vx, vy = track.kalman_filter.x[4:6]
                    print(f"  [Kalman] T{track.track_id}: pos=({px:.0f},{py:.0f}), "
                          f"vel=({vx:.1f},{vy:.1f}), size=({pw:.0f}x{ph:.0f})")

        # 2. Separar tracks activos y perdidos
        active_tracks = [tid for tid, t in self.tracks.items() if t.lost == 0]
        lost_tracks = [tid for tid, t in self.tracks.items() if t.lost > 0]
        
        assignments = {}
        
        # --- FASE 1: Emparejamiento por IoU para tracks activos ---
        if active_tracks and detections:
            iou_matrix = np.zeros((len(active_tracks), len(detections)), dtype=np.float32)
            
            for i, tid in enumerate(active_tracks):
                # Usar bbox PREDICHO (más preciso que el anterior)
                pred_bbox = self.tracks[tid].predicted_bbox or self.tracks[tid].bbox
                for j, (det_bbox, _) in enumerate(detections):
                    iou_matrix[i, j] = iou(pred_bbox, det_bbox)
            
            # Algoritmo Húngaro para asignación óptima
            # Maximizar IoU = minimizar -IoU
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    tid = active_tracks[r]
                    assignments[tid] = c

        used_dets = set(assignments.values())
        unassigned_dets = [i for i in range(len(detections)) if i not in used_dets]

        # --- FASE 2: Emparejamiento por Mahalanobis para tracks perdidos ---
        if lost_tracks and unassigned_dets:
            dist_matrix = np.full((len(lost_tracks), len(unassigned_dets)), np.inf, dtype=np.float32)
            
            for i, tid in enumerate(lost_tracks):
                track = self.tracks[tid]
                
                # 🔥 CAMBIO: Permitir tracks más nuevos (hits >= 2 en vez de min_hits)
                if track.hits < 2:
                    continue
                
                # 🔥 CAMBIO: Threshold adaptativo según tiempo perdido
                if track.lost > 15:
                    threshold_multiplier = 1.5  # Más permisivo
                else:
                    threshold_multiplier = 1.0
                
                for j, det_idx in enumerate(unassigned_dets):
                    det_bbox = detections[det_idx][0]
                    dist = self._mahalanobis_distance(track, det_bbox)
                    
                    # Usar threshold adaptativo
                    effective_threshold = self.mahalanobis_threshold * threshold_multiplier
                    
                    if dist < effective_threshold:
                        dist_matrix[i, j] = dist
                        
                        if self.debug and dist < self.mahalanobis_threshold:
                            print(f"  [Match candidate] T{tid} (lost={track.lost}) <-> Det{det_idx}: dist={dist:.2f}")
            
            # Solo aplicar húngaro si hay al menos una asignación válida
            if np.any(np.isfinite(dist_matrix)):
                try:
                    row_ind, col_ind = linear_sum_assignment(dist_matrix)
                    
                    for r, c in zip(row_ind, col_ind):
                        # 🔥 Usar threshold adaptativo también aquí
                        track = self.tracks[lost_tracks[r]]
                        threshold_multiplier = 1.5 if track.lost > 15 else 1.0
                        effective_threshold = self.mahalanobis_threshold * threshold_multiplier
                        
                        if dist_matrix[r, c] < effective_threshold:
                            tid = lost_tracks[r]
                            det_idx = unassigned_dets[c]
                            assignments[tid] = det_idx
                            
                            if self.debug:
                                print(f"  [✅ RE-ID] T{tid} (lost={track.lost}) <-> Det{det_idx}: dist={dist_matrix[r,c]:.2f}")
                except (ValueError, np.linalg.LinAlgError):
                    pass  # Matriz inviable, continuar sin más asignaciones

        # 3. ACTUALIZACIÓN de estados
        
        # A) Tracks emparejados: actualizar con CORRECCIÓN de Kalman
        for tid, det_idx in assignments.items():
            track = self.tracks[tid]
            bbox, conf = detections[det_idx]
            
            # Actualizar filtro de Kalman con medición [x, y, w, h]
            if track.kalman_filter:
                center = bbox_center(bbox)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                z = np.array([center[0], center[1], w, h], dtype=np.float32)
                
                # Actualización de Kalman
                track.kalman_filter.update(z)
                
                # 🔥 MODO HÍBRIDO ADAPTATIVO: Mezcla Kalman + YOLO según confianza
                if track.hits >= 3:  # Empezar híbrido antes (era 5)
                    # Track con historial: usar Kalman
                    px, py, pw, ph = track.kalman_filter.x[:4]
                    
                    # Validación de sanidad
                    if 10 < pw < 1000 and 10 < ph < 1000:
                        corrected_bbox = (
                            int(px - pw/2), 
                            int(py - ph/2), 
                            int(px + pw/2), 
                            int(py + ph/2)
                        )
                        
                        # 🔥 Peso adaptativo: más hits = más confianza en Kalman
                        if track.hits >= 10:
                            kalman_weight = 0.80  # Muy establecido: 80% Kalman
                        elif track.hits >= 5:
                            kalman_weight = 0.70  # Establecido: 70% Kalman
                        else:
                            kalman_weight = 0.50  # Nuevo: 50-50
                        
                        yolo_weight = 1.0 - kalman_weight
                        
                        final_bbox = (
                            int(kalman_weight * corrected_bbox[0] + yolo_weight * bbox[0]),
                            int(kalman_weight * corrected_bbox[1] + yolo_weight * bbox[1]),
                            int(kalman_weight * corrected_bbox[2] + yolo_weight * bbox[2]),
                            int(kalman_weight * corrected_bbox[3] + yolo_weight * bbox[3])
                        )
                        bbox = final_bbox
                    # Si falla validación, usar bbox original de YOLO
                else:
                    # Track muy nuevo: usar YOLO directamente
                    pass
                
                if self.debug and tid % 5 == 0:
                    print(f"  [Update] T{tid}: hits={track.hits}, conf={conf:.2f}")
            
            # Actualizar track con bbox (híbrido o YOLO)
            track.update(frame, bbox, conf)

        # B) Tracks no emparejados: marcar como perdidos
        unassigned_tracks = [tid for tid in self.tracks if tid not in assignments]
        for tid in unassigned_tracks:
            self.tracks[tid].mark_missed()

        # C) Detecciones no emparejadas: crear nuevos tracks
        used_dets = set(assignments.values())
        new_det_indices = [i for i in range(len(detections)) if i not in used_dets]
        for det_idx in new_det_indices:
            bbox, conf = detections[det_idx]
            self._create_track(frame, bbox, conf)

        # 4. LIMPIEZA: Eliminar tracks perdidos por mucho tiempo
        to_delete = [tid for tid, t in self.tracks.items() if t.lost > self.max_lost]
        for tid in to_delete:
            if self.debug:
                print(f"  [Delete] T{tid}: lost={self.tracks[tid].lost}, hits={self.tracks[tid].hits}")
            del self.tracks[tid]
        
        # Debug periódico
        if self.debug and self.frame_count % 50 == 0:
            active = sum(1 for t in self.tracks.values() if t.lost == 0)
            lost = sum(1 for t in self.tracks.values() if t.lost > 0)
            print(f"\n[Frame {self.frame_count}] Tracks: {len(self.tracks)} total "
                  f"({active} active, {lost} lost), Detections: {len(detections)}\n")
            
        return self.tracks

    def draw_tracks(self, frame: np.ndarray, min_hits: Optional[int] = None) -> np.ndarray:
        """
        Dibuja los tracks con visualización MEJORADA de Kalman.
        - Verde: bbox actual (detección)
        - Azul: bbox predicho por Kalman
        - Rojo: bbox durante oclusión (solo predicción)
        """
        if min_hits is None:
            min_hits = self.min_hits
        
        for t in self.tracks.values():
            if t.hits < min_hits:
                continue
            if t.lost > 8:
                continue

            # Determinar qué bbox dibujar
            if t.lost == 0:
                # Track ACTIVO: dibujar detección actual (verde)
                color = (0, 255, 0)  # Verde
                bbox_to_draw = t.bbox
                thickness = 2
                style = "-"
                
                # OPCIONAL: Dibujar también predicción en azul claro (para debug)
                if t.predicted_bbox:
                    px1, py1, px2, py2 = t.predicted_bbox
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 200, 0), 1)  # Azul claro
                    
            else:
                # Track PERDIDO: dibujar predicción (rojo/naranja)
                color = (0, 100, 255)  # Naranja-rojo
                bbox_to_draw = t.predicted_bbox or t.bbox
                thickness = 1
                style = "- -"  # Discontinua

            # Dibujar bbox
            if bbox_to_draw:
                x1, y1, x2, y2 = bbox_to_draw
                
                if t.lost > 0:
                    # Línea discontinua para perdidos
                    self._draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
                else:
                    # Línea continua para activos
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Etiqueta
                lbl = f"ID {t.track_id}"
                if t.kalman_filter and t.lost == 0:
                    # Mostrar velocidad estimada
                    vx, vy = t.kalman_filter.x[4:6]
                    speed = math.hypot(vx, vy)
                    lbl += f" {speed:.1f}px/f"
                if t.lost > 0:
                    lbl += f" [lost:{t.lost}]"
                
                cv2.putText(frame, lbl, (x1, max(0, y1 - 6)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Trayectoria
            if len(t.centroids) >= 2:
                points = t.centroids[-15:]
                for i, (p, q) in enumerate(zip(points[:-1], points[1:])):
                    # Color degradado: más reciente = más brillante
                    alpha = (i + 1) / len(points)
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, (int(p[0]), int(p[1])), 
                            (int(q[0]), int(q[1])), trail_color, 2)
        
        return frame
    
    def _draw_dashed_rect(self, img, pt1, pt2, color, thickness=1, gap=10):
        """Dibuja un rectángulo con líneas discontinuas."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Líneas horizontales
        for x in range(x1, x2, gap*2):
            cv2.line(img, (x, y1), (min(x+gap, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x+gap, x2), y2), color, thickness)
        
        # Líneas verticales
        for y in range(y1, y2, gap*2):
            cv2.line(img, (x1, y), (x1, min(y+gap, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y+gap, y2)), color, thickness)