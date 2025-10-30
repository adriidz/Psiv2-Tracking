# car.py - Versión mejorada
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
from utilities import compute_hsv_hist, bbox_center
from filterpy.kalman import KalmanFilter

BBox = Tuple[int, int, int, int]

@dataclass
class Car:
    """
    Representa un vehículo rastreado con historial de posiciones,
    características de apariencia y predicciones durante oclusiones.
    """
    track_id: int
    bbox: BBox
    confidence: float = 0.0
    hits: int = 1
    lost: int = 0
    
    # Historial de posiciones
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    
    # Características de apariencia
    hsv_hist: Optional[np.ndarray] = None
    
    # Predicción durante oclusión
    predicted_bbox: Optional[BBox] = None
    
    # Velocidades estimadas (para predicción más suave)
    velocity_history: List[Tuple[float, float]] = field(default_factory=list)
    max_velocity_history: int = 5
    
    # Estado estático/móvil
    is_static: bool = False
    
    # Filtro de Kalman para predicción de estado
    kalman_filter: Optional['KalmanFilter'] = None
    
    def update(self, frame: np.ndarray, bbox: BBox, conf: float):
        """
        Actualiza el track con nueva detección.
        """
        # Calcular velocidad antes de actualizar
        if self.centroids:
            old_center = self.centroids[-1]
            new_center = bbox_center(bbox)
            vx = new_center[0] - old_center[0]
            vy = new_center[1] - old_center[1]
            
            self.velocity_history.append((vx, vy))
            if len(self.velocity_history) > self.max_velocity_history:
                self.velocity_history.pop(0)
        
        # Actualizar estado
        self.bbox = bbox
        self.confidence = conf
        self.hits += 1
        self.lost = 0
        
        # Actualizar centroide
        center = bbox_center(bbox)
        self.centroids.append(center)
        
        # Limitar historial de centroides (últimos 50)
        if len(self.centroids) > 50:
            self.centroids = self.centroids[-50:]
        
        # Actualizar histograma de apariencia (con suavizado)
        try:
            new_hist = compute_hsv_hist(frame, bbox)
            if self.hsv_hist is None:
                self.hsv_hist = new_hist
            else:
                # Suavizado: 80% anterior + 20% nuevo
                self.hsv_hist = 0.8 * self.hsv_hist + 0.2 * new_hist
                # Renormalizar
                s = self.hsv_hist.sum()
                if s > 0:
                    self.hsv_hist /= s
        except:
            pass
        
        # Limpiar predicción al recuperar detección real
        self.predicted_bbox = None
    
    def mark_missed(self):
        """Marca el track como no detectado en este frame."""
        self.lost += 1
    
    def current_direction(self) -> Optional[str]:
        """
        Determina la dirección actual del movimiento.
        Returns: 'up', 'down', 'left', 'right', o None
        """
        if len(self.centroids) < 3:
            return None
        
        # Usar últimas posiciones para determinar dirección
        recent = self.centroids[-min(5, len(self.centroids)):]
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        
        # Umbral mínimo de movimiento
        if abs(dx) < 5 and abs(dy) < 5:
            return None
        
        # Determinar dirección predominante
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def get_average_velocity(self) -> Tuple[float, float]:
        """
        Calcula la velocidad promedio reciente con pesos decrecientes.
        Útil para predicciones más suaves.
        """
        if not self.velocity_history:
            return (0.0, 0.0)
        
        # Pesos lineales: más recientes tienen más peso
        weights = np.linspace(0.5, 1.0, len(self.velocity_history))
        weights /= weights.sum()
        
        vx = sum(v[0] * w for v, w in zip(self.velocity_history, weights))
        vy = sum(v[1] * w for v, w in zip(self.velocity_history, weights))
        
        return (vx, vy)
    
    def is_moving_fast(self, threshold: float = 10.0) -> bool:
        """
        Determina si el vehículo se mueve rápidamente.
        Útil para ajustar parámetros de búsqueda.
        """
        vx, vy = self.get_average_velocity()
        speed = (vx**2 + vy**2)**0.5
        return speed > threshold