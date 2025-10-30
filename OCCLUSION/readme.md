# 🚗 Vehicle Tracking System - Documentación Completa

## 📋 Índice
1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Configuración y Parámetros](#configuración-y-parámetros)
4. [Sistema de Validaciones](#sistema-de-validaciones)
5. [Guía de Uso](#guía-de-uso)
6. [Troubleshooting](#troubleshooting)
7. [Documentación Adicional](#documentación-adicional)

---

## 🎯 Visión General

Sistema de tracking de vehículos que combina **detección YOLO** con un **tracker personalizado ultra-estricto** diseñado para prevenir saltos imposibles y mantener coherencia física en las trayectorias.

### ✨ Características Principales

- **Detección**: YOLO11 (variantes n/s/m) con umbral de confianza 0.42
- **Tracking**: OcclusionTracker con 10 validaciones físicas en cascada
- **Anti-Saltos**: Sistema de validación estricto que previene asociaciones imposibles
- **Oclusión**: Manejo robusto de tracks perdidos temporalmente
- **Conteo**: VehicleCounter integrado con líneas de conteo configurables

### 🏆 Versión Actual: Ultra-Estricto v3.0

**Filosofía**: "Mejor crear un ID nuevo que mezclar objetos diferentes"

**Características**:
- 10 validaciones físicas en cascada
- Radio de búsqueda conservador (100-140px)
- Umbral de matching alto (0.50)
- Prioridad absoluta en coherencia física sobre minimización de IDs

---

## 🏗️ Arquitectura del Sistema

### Estructura de Archivos

```
OCCLUSION/
├── main.py                          # Punto de entrada con configuración
├── improved_tracker.py              # Lógica principal del tracker
├── car.py                           # Clase Car con historial de trayectorias
├── detection_frames.py              # Procesamiento de detecciones YOLO
├── VehicleCounter.py                # Sistema de conteo de vehículos
├── utilities.py                     # Funciones auxiliares (IoU, histogramas, etc.)
├── README.md                        # 👈 Este archivo
├── SOLUCION_SALTOS_IMPOSIBLES.md   # Documentación técnica de validaciones
├── COMPARACION_ENFOQUES.md         # Evolución y comparación de versiones
├── weights/
│   └── yolo11n.pt                  # Modelo YOLO
└── runs/                            # Videos procesados y resultados
```

### Flujo de Datos

```
Video Frame → YOLO Detection → OcclusionTracker → VehicleCounter → Output
                  ↓                    ↓                ↓
            Bounding Boxes      Track Association    Count Events
            + Confidence        + ID Assignment      + Statistics
```

---

## ⚙️ Configuración y Parámetros

### 📍 Ubicación: `main.py`

### Parámetros de Detección

```python
# YOLO
MODEL = "yolo11s.pt"           # Modelo: yolo11n (rápido) / yolo11s (balanceado) / yolo11m (preciso)
IMAGE_SIZE = 960               # Resolución de entrada (640/960/1280)
CONFIDENCE_THRESHOLD = 0.42    # Umbral de confianza para detecciones

# Clases de interés
CLASSES_OF_INTEREST = [2, 3, 5, 7]  # 2=car, 3=motorcycle, 5=bus, 7=truck
```

### Parámetros del Tracker

```python
# Búsqueda y Matching
SEARCH_RADIUS = 100.0          # Radio base de búsqueda (px)
MIN_MATCH_SCORE = 0.50         # Umbral mínimo de score para match
MIN_HITS = 4                   # Detecciones antes de confirmar track
SKIP_FRAMES = 1                # Frames entre procesamiento (1=todos)

# Gestión de Tracks
BUFFER_FRAMES = 20             # Frames que permanece un track perdido
MAX_AGE = 60                   # Edad máxima sin detección antes de eliminar

# Validaciones Físicas
MAX_VELOCITY_RATIO = 3.0       # Cambio máximo de velocidad permitido
MAX_DIRECTION_CHANGE = 100     # Cambio máximo de dirección (grados)
MAX_SIZE_RATIO = 1.8           # Cambio máximo de tamaño
```

### Parámetros de Conteo

```python
# Líneas de conteo (coordenadas en píxeles)
COUNTING_LINES = [
    {"start": (400, 0), "end": (400, 720), "direction": "horizontal"},
    {"start": (800, 0), "end": (800, 720), "direction": "horizontal"}
]
```

---

## 🛡️ Sistema de Validaciones

El tracker implementa **10 validaciones en cascada** para prevenir asociaciones imposibles:

### 1️⃣ Validación de Distancia Base
```python
if distance > search_radius:
    return 0.0  # Fuera de alcance
```
- **Radio dinámico**: 100px base, hasta 140px para tracks perdidos
- **Objetivo**: Limitar búsqueda a vecindad razonable

### 2️⃣ Validación de Velocidad Extrema
```python
if velocity > MAX_VELOCITY * 3:
    return 0.0  # Velocidad imposible
```
- **Límite**: 3x la velocidad máxima observada
- **Objetivo**: Rechazar teleportaciones instantáneas

### 3️⃣ Validación de Cambio de Velocidad
```python
velocity_ratio = new_velocity / avg_velocity
if velocity_ratio > MAX_VELOCITY_RATIO or velocity_ratio < 1/MAX_VELOCITY_RATIO:
    return 0.0  # Aceleración imposible
```
- **Límite**: 3x cambio de velocidad para tracks establecidos
- **Objetivo**: Coherencia en aceleración/desaceleración

### 4️⃣ Validación de Dirección
```python
direction_diff = abs(new_direction - track_direction)
if direction_diff > MAX_DIRECTION_CHANGE:
    return 0.0  # Cambio de dirección imposible
```
- **Límites dinámicos**:
  - Tracks recientes (hits < 6): 100° máximo
  - Tracks establecidos (hits ≥ 6): 75° máximo
- **Objetivo**: Prevenir giros bruscos irreales

### 5️⃣ Validación de Movimiento Perpendicular
```python
dot_product = dx * vx + dy * vy
if abs(dot_product) < distance * avg_velocity * 0.3:
    return 0.0  # Movimiento perpendicular
```
- **Criterio**: El movimiento debe tener componente en dirección histórica
- **Objetivo**: Rechazar movimientos en ángulo recto respecto a trayectoria

### 6️⃣ Validación de Tamaño
```python
size_ratio = new_size / track_size
if size_ratio > MAX_SIZE_RATIO or size_ratio < 1/MAX_SIZE_RATIO:
    return 0.0  # Cambio de tamaño imposible
```
- **Límite**: 1.8x cambio de tamaño
- **Objetivo**: Prevenir asociaciones entre vehículos de tamaños incompatibles

### 7️⃣ Validación de Desaceleración Extrema
```python
if velocity_ratio < 0.25:  # Reducción >75%
    return 0.0  # Desaceleración imposible
```
- **Límite**: Velocidad no puede caer a <25% instantáneamente
- **Objetivo**: Rechazar frenados físicamente imposibles

### 8️⃣ Validación de Apariencia (HSV)
```python
correlation = cv2.compareHist(track_hist, det_hist, cv2.HISTCMP_CORREL)
if correlation < 0.3:  # Apariencia muy diferente
    score *= 0.5  # Penalización fuerte
```
- **Umbral**: Correlación HSV > 0.3
- **Objetivo**: Favorecer vehículos visualmente similares

### 9️⃣ Validación de IoU (Tracks Activos)
```python
iou_value = iou(track_bbox, det_bbox)
if iou_value > 0.3:
    return iou_value  # Match directo
```
- **Umbral**: IoU > 0.3 para match directo
- **Objetivo**: Priorizar solapamiento espacial para tracks recientes

### 🔟 Validación de Score Final
```python
if match_score < MIN_MATCH_SCORE:
    return None  # No match
```
- **Umbral**: 0.50 (50%)
- **Objetivo**: Solo aceptar matches con alta confianza

---

## 📊 Composición del Score de Matching

El score final combina múltiples componentes:

```python
score = (
    0.45 * distance_score +    # 45% - Proximidad espacial
    0.25 * appearance_score +  # 25% - Similitud visual
    0.20 * direction_score +   # 20% - Coherencia direccional
    0.10 * size_score          # 10% - Consistencia de tamaño
)
```

### Normalización de Componentes

```python
# Distancia (normalizada por radio)
distance_score = 1.0 - (distance / search_radius)

# Apariencia (correlación HSV)
appearance_score = max(0.0, hsv_correlation)

# Dirección (penalización por diferencia angular)
direction_score = 1.0 - (direction_diff / 180.0)

# Tamaño (penalización por ratio)
size_score = 1.0 - abs(1.0 - size_ratio)
```

---

## 🚀 Guía de Uso

### Instalación de Dependencias

```bash
pip install ultralytics opencv-python numpy
```

### Ejecución Básica

```bash
cd OCCLUSION
python main.py
```

### Configuración Rápida

1. **Seleccionar Modelo**:
   ```python
   # En main.py
   MODEL = "yolo11n.pt"  # Rápido (30+ FPS)
   MODEL = "yolo11s.pt"  # Balanceado (20-25 FPS) ← RECOMENDADO
   MODEL = "yolo11m.pt"  # Preciso (15-20 FPS)
   ```

2. **Ajustar Sensibilidad**:
   ```python
   # Más estricto (menos IDs, más riesgo de saltos)
   MIN_MATCH_SCORE = 0.55
   SEARCH_RADIUS = 90.0
   
   # Más permisivo (más IDs, menos saltos)
   MIN_MATCH_SCORE = 0.45
   SEARCH_RADIUS = 120.0
   ```

3. **Configurar Conteo**:
   ```python
   # Línea vertical en x=400
   {"start": (400, 0), "end": (400, 720), "direction": "vertical"}
   
   # Línea horizontal en y=500
   {"start": (0, 500), "end": (1280, 500), "direction": "horizontal"}
   ```

### Modos de Salida

```python
# main.py
SAVE_VIDEO = True          # Guardar video con anotaciones
DISPLAY_REALTIME = True    # Mostrar ventana en tiempo real
DEBUG_MODE = True          # Imprimir mensajes de validación
```

---

## 🔧 Troubleshooting

### Problema: Demasiados IDs nuevos

**Síntomas**:
- Video corto genera 300+ IDs
- Tracks se fragmentan frecuentemente
- IDs cambian aunque vehículo esté visible

**Soluciones**:

1. **Reducir MIN_MATCH_SCORE**:
   ```python
   MIN_MATCH_SCORE = 0.45  # Era 0.50
   ```

2. **Aumentar SEARCH_RADIUS**:
   ```python
   SEARCH_RADIUS = 120.0   # Era 100.0
   ```

3. **Aumentar BUFFER_FRAMES**:
   ```python
   BUFFER_FRAMES = 30      # Era 20
   ```

4. **Reducir MIN_HITS**:
   ```python
   MIN_HITS = 3            # Era 4
   ```

### Problema: Saltos imposibles / Mega-saltos

**Síntomas**:
- IDs saltan de un extremo a otro del frame
- Contador se activa falsamente
- Trayectorias cruzan toda la pantalla instantáneamente

**Soluciones**:

1. **Aumentar MIN_MATCH_SCORE**:
   ```python
   MIN_MATCH_SCORE = 0.55  # Era 0.50
   ```

2. **Reducir SEARCH_RADIUS**:
   ```python
   SEARCH_RADIUS = 90.0    # Era 100.0
   ```

3. **Activar DEBUG para identificar validaciones**:
   ```python
   DEBUG_MODE = True
   ```
   Buscar líneas `REJECT` en consola para ver qué validación falla.

4. **Reducir MAX_DIRECTION_CHANGE**:
   ```python
   MAX_DIRECTION_CHANGE = 70  # Era 100
   ```

### Problema: Tracks se pierden en oclusiones

**Síntomas**:
- Vehículos detrás de obstáculos generan nuevo ID
- ID cambia tras pasar detrás de otro vehículo
- Tracks válidos desaparecen prematuramente

**Soluciones**:

1. **Aumentar BUFFER_FRAMES**:
   ```python
   BUFFER_FRAMES = 35      # Era 20
   ```

2. **Aumentar MAX_AGE**:
   ```python
   MAX_AGE = 80            # Era 60
   ```

3. **Reducir MIN_HITS** (para confirmar tracks más rápido):
   ```python
   MIN_HITS = 3            # Era 4
   ```

### Problema: Performance bajo

**Síntomas**:
- FPS < 10
- Procesamiento lento
- Retrasos en visualización

**Soluciones**:

1. **Usar modelo más ligero**:
   ```python
   MODEL = "yolo11n.pt"    # Era yolo11s.pt
   ```

2. **Reducir IMAGE_SIZE**:
   ```python
   IMAGE_SIZE = 640        # Era 960
   ```

3. **Aumentar SKIP_FRAMES**:
   ```python
   SKIP_FRAMES = 2         # Era 1 (procesa 1 de cada 2 frames)
   ```

4. **Desactivar visualización**:
   ```python
   DISPLAY_REALTIME = False
   ```

---

## 📈 Ajustes Recomendados por Escenario

### 🏙️ Tráfico Urbano Denso
```python
SEARCH_RADIUS = 90.0
MIN_MATCH_SCORE = 0.55
MIN_HITS = 4
BUFFER_FRAMES = 25
MAX_DIRECTION_CHANGE = 100
```
**Rationale**: Muchas oclusiones, movimientos complejos, priorizar anti-saltos.

### 🛣️ Autopista/Carretera
```python
SEARCH_RADIUS = 120.0
MIN_MATCH_SCORE = 0.45
MIN_HITS = 3
BUFFER_FRAMES = 35
MAX_DIRECTION_CHANGE = 50
```
**Rationale**: Movimiento predecible y lineal, pocas oclusiones, priorizar continuidad.

### 🚦 Intersección con Giros
```python
SEARCH_RADIUS = 100.0
MIN_MATCH_SCORE = 0.50
MIN_HITS = 4
BUFFER_FRAMES = 20
MAX_DIRECTION_CHANGE = 120
```
**Rationale**: Cambios de dirección frecuentes pero controlados, balance entre flexibilidad y coherencia.

### 🌃 Video Nocturno/Baja Calidad
```python
SEARCH_RADIUS = 110.0
MIN_MATCH_SCORE = 0.48
MIN_HITS = 3
BUFFER_FRAMES = 30
CONFIDENCE_THRESHOLD = 0.35
```
**Rationale**: Detecciones menos confiables, más permisivo pero con validaciones activas.

---

## 🧪 Modo Debug

Activar para diagnóstico detallado:

```python
DEBUG_MODE = True
```

### Salida de Debug

```
Frame 45: 12 detections, 8 active tracks
  Track 5 (hits=15, lost=0) matched to detection 3 (score=0.68)
    ✓ Distance: 35.2px (limit: 100px)
    ✓ Velocity: 12.3px/f → 14.1px/f (ratio: 1.15, limit: 3.0)
    ✓ Direction: 85° → 92° (change: 7°, limit: 75°)
    ✓ Size: 4850px² → 5120px² (ratio: 1.06, limit: 1.8)
    ✓ Appearance: HSV correlation 0.72
    → MATCH ACCEPTED
    
  Track 12 (hits=8, lost=3) vs detection 7:
    ✗ Direction: 45° → 150° (change: 105°, limit: 100°)
    → REJECT: Direction change too large
```

### Interpretar Mensajes REJECT

- `REJECT: Distance`: Detección fuera de radio de búsqueda
- `REJECT: Extreme velocity`: Velocidad instantánea imposible
- `REJECT: Velocity ratio`: Aceleración/desaceleración imposible
- `REJECT: Direction change`: Giro demasiado brusco
- `REJECT: Perpendicular movement`: Movimiento en ángulo recto a trayectoria
- `REJECT: Size ratio`: Cambio de tamaño incompatible
- `REJECT: Extreme deceleration`: Frenado físicamente imposible
- `REJECT: Score too low`: Score final < MIN_MATCH_SCORE

---

## 📚 Documentación Adicional

### Archivos de Referencia

1. **SOLUCION_SALTOS_IMPOSIBLES.md**
   - Explicación técnica de las 10 validaciones
   - Ejemplos de código con cálculos detallados
   - Casos de uso y edge cases

2. **COMPARACION_ENFOQUES.md**
   - Evolución del tracker (4 versiones)
   - Comparación de parámetros
   - Resultados y trade-offs de cada enfoque

### Enlaces Externos

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Kalman Filter Tracking](https://en.wikipedia.org/wiki/Kalman_filter)

---

## 🎯 Resumen de Filosofía del Tracker

### Principios Clave

1. **Coherencia Física > Minimización de IDs**
   - Mejor crear IDs nuevos que asociaciones incorrectas
   - Las validaciones físicas son inamovibles

2. **Validaciones en Cascada**
   - Cada validación es una barrera adicional
   - Fallar cualquiera = rechazo inmediato
   - No hay "compensación" entre validaciones

3. **Radio de Búsqueda Conservador**
   - 100px base es suficiente para movimiento normal
   - Expansión gradual solo para tracks perdidos (max 140px)
   - Nunca superar límites de velocidad física

4. **Score Alto = Confianza Alta**
   - MIN_MATCH_SCORE = 0.50 es deliberadamente alto
   - Solo asociaciones muy probables son aceptadas
   - La duda favorece la creación de nuevo ID

5. **Prioridad: Anti-Saltos**
   - Evitar falsos conteos es crítico
   - Un mega-salto puede invalidar todo el conteo
   - La robustez del conteo depende de la coherencia de tracks

---

## 🔄 Ciclo de Mejora Continua

### Proceso Recomendado

1. **Ejecutar con video de prueba**
2. **Activar DEBUG_MODE**
3. **Analizar mensajes REJECT**
4. **Identificar patrón problemático**
5. **Ajustar UNO o DOS parámetros**
6. **Repetir**

### Métricas de Éxito

- ✅ **0 mega-saltos** (cruces imposibles de pantalla)
- ✅ **Tracks coherentes** (trayectorias suaves y predecibles)
- ✅ **Conteo preciso** (sin falsos positivos por saltos)
- ⚠️ **IDs totales**: Aceptable si < 2x número real de vehículos
- ⚠️ **Fragmentación**: Aceptable si ocurre en oclusiones largas

### Red Flags

- 🚨 **IDs saltando** entre vehículos opuestos
- 🚨 **Cambios de dirección >100°** en 1 frame
- 🚨 **Velocidades >50px/frame** sin justificación
- 🚨 **Conteo incrementándose** sin cruces reales

---

## 📞 Soporte

Para problemas no resueltos con esta documentación:

1. Revisar `SOLUCION_SALTOS_IMPOSIBLES.md` para detalles técnicos
2. Consultar `COMPARACION_ENFOQUES.md` para contexto histórico
3. Activar `DEBUG_MODE` y analizar salida
4. Documentar: parámetros usados, síntomas, mensajes REJECT

---

**Versión del Documento**: 3.0
**Última Actualización**: 2024
**Configuración Actual**: Ultra-Estricto v3.0

---

