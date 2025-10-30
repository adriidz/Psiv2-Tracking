# 🔧 TROUBLESHOOTING - Kalman Tracker No Funciona Bien

## 🎯 Diagnóstico Rápido

### Ejecuta con Debug Activado
```powershell
python main.py --debug
```

Observa en la consola:
- `[Kalman]`: Predicciones del filtro
- `[Mahal]`: Distancias de emparejamiento
- `[Update]`: Actualizaciones de tracks
- `[Delete]`: Tracks eliminados

---

## 🐛 Problemas Comunes y Soluciones

### ❌ PROBLEMA 1: "Sigue creando muchos IDs diferentes"

**Causa**: Kalman no está emparejando bien tras oclusiones

**Solución 1**: Aumentar threshold de Mahalanobis
```python
# En main.py, línea ~17
MAHALANOBIS_THRESHOLD = 5.0  # Más permisivo (era 4.0)
```

**Solución 2**: Aumentar max_lost
```python
MAX_LOST_FRAMES = 30  # Mantener tracks más tiempo
```

**Solución 3**: Reducir min_hits
```python
MIN_HITS = 2  # Confirmar más rápido
```

---

### ❌ PROBLEMA 2: "Los bbox 'saltan' o se ven inestables"

**Causa**: Matrices Q/R mal balanceadas

**Solución**: En `improved_tracker.py`, función `create_kalman_filter()`:

**Para MÁS suavizado (menos jitter):**
```python
kf.Q[4:6, 4:6] *= 20.0   # Menos ruido en velocidad (era 50.0)
kf.R[:2, :2] *= 15.0     # Más confianza en modelo (era 10.0)
```

**Para MÁS respuesta (seguir mejor cambios):**
```python
kf.Q[4:6, 4:6] *= 100.0  # Más ruido en velocidad (era 50.0)
kf.R[:2, :2] *= 5.0      # Más confianza en YOLO (era 10.0)
```

---

### ❌ PROBLEMA 3: "Peor que OcclusionTracker"

**Causa**: Kalman puede no ser adecuado para tu escenario

**Solución**: Volver a OcclusionTracker

En `main.py`, cambiar:
```python
# Línea ~220 (aproximadamente)
from improved_tracker import OcclusionTracker  # Importar

tracker = OcclusionTracker(  # Cambiar clase
    iou_threshold=0.25,
    max_lost=25,
    min_hits=4,
    buffer_frames=20,
    search_radius=100.0,
    min_match_score=0.50,
    debug=args.debug
)
```

**Kalman es mejor para:**
- ✅ Trayectorias largas y suaves
- ✅ Oclusiones predecibles
- ✅ Vehículos con velocidad constante

**OcclusionTracker es mejor para:**
- ✅ Conteo preciso (menos falsos positivos)
- ✅ Escenas complejas con muchas oclusiones
- ✅ Cambios bruscos de velocidad/dirección

---

### ❌ PROBLEMA 4: "Tracks 'driftan' (se desvían) durante oclusión"

**Causa**: Modo híbrido no activo o mal calibrado

**Solución**: En `improved_tracker.py`, línea ~970:

**Aumentar peso de YOLO:**
```python
# Cambiar de 70% Kalman + 30% YOLO a:
final_bbox = (
    int(0.5 * corrected_bbox[0] + 0.5 * bbox[0]),  # 50-50
    int(0.5 * corrected_bbox[1] + 0.5 * bbox[1]),
    int(0.5 * corrected_bbox[2] + 0.5 * bbox[2]),
    int(0.5 * corrected_bbox[3] + 0.5 * bbox[3])
)
```

**O desactivar completamente (usar solo YOLO):**
```python
# Comentar todo el bloque if track.hits >= 5:
# y dejar solo:
pass  # Usar bbox original de YOLO
```

---

### ❌ PROBLEMA 5: "Error: LinAlgError o matriz singular"

**Causa**: Covarianzas mal inicializadas

**Solución**: Ya está solucionado con try-catch, pero si persiste:

En `create_kalman_filter()`:
```python
# Aumentar covarianza inicial
kf.P[:4, :4] *= 100.0    # Era 50.0
kf.P[4:6, 4:6] *= 1000.0 # Era 500.0
```

---

## 📊 Comparación de Parámetros

### Escenario: Tráfico Urbano Denso
```python
IOU_THRESHOLD = 0.30
MAX_LOST_FRAMES = 15
MIN_HITS = 5
MAHALANOBIS_THRESHOLD = 3.5

# En create_kalman_filter():
kf.Q[4:6, 4:6] *= 30.0   # Velocidad cambia mucho
kf.R[:2, :2] *= 8.0      # Confiar más en modelo
```

### Escenario: Autopista (velocidad constante)
```python
IOU_THRESHOLD = 0.25
MAX_LOST_FRAMES = 25
MIN_HITS = 3
MAHALANOBIS_THRESHOLD = 5.0

# En create_kalman_filter():
kf.Q[4:6, 4:6] *= 10.0   # Velocidad más constante
kf.R[:2, :2] *= 15.0     # Confiar mucho en modelo
```

### Escenario: Muchas Oclusiones
```python
IOU_THRESHOLD = 0.35
MAX_LOST_FRAMES = 40
MIN_HITS = 4
MAHALANOBIS_THRESHOLD = 6.0

# En create_kalman_filter():
kf.Q[4:6, 4:6] *= 50.0   # Permitir cambios
kf.R[:2, :2] *= 12.0     # Balance
```

---

## 🧪 Experimentos Recomendados

### Test 1: Visualizar Predicciones
```python
# En main.py
SHOW_PREDICTIONS = True  # Ya existe
DEBUG_MODE = True
```

Observa:
- Bbox verde (YOLO) vs bbox azul (Kalman)
- ¿Están cerca? ✅ Kalman funciona bien
- ¿Muy separados? ❌ Ajustar Q/R

### Test 2: Contar Cambios de ID
```bash
# Ejecutar y anotar:
python main.py > log.txt

# Buscar en log.txt:
grep "\[NEW\]" log.txt | wc -l  # Número de IDs creados
```

**Meta**: < 20% más IDs que número de vehículos reales

### Test 3: Comparación Lado a Lado
```bash
# Ejecutar OcclusionTracker
# Anotar: IDs creados, conteo final

# Ejecutar KalmanTracker
# Anotar: IDs creados, conteo final

# Comparar resultados
```

---

## 💡 Cuándo Usar Cada Tracker

### Usa **KalmanTracker** si:
- ✅ Necesitas trayectorias suaves
- ✅ Tienes oclusiones largas (>5 frames)
- ✅ Los vehículos se mueven de forma predecible
- ✅ Toleras algunos IDs extra a cambio de mejor seguimiento

### Usa **OcclusionTracker** si:
- ✅ El conteo preciso es crítico
- ✅ Hay muchos cambios bruscos de velocidad
- ✅ Las oclusiones son impredecibles
- ✅ Prefieres prevenir falsos positivos

---

## 🚀 Siguiente Nivel: Tracker Híbrido

Si ninguno funciona bien, considera crear un **HybridTracker**:

```python
class HybridTracker(Tracker):
    """Usa Kalman para tracking, OcclusionTracker para validación"""
    
    def update(self, frame, detections):
        # 1. Predicción con Kalman
        kalman_predictions = self._kalman_predict()
        
        # 2. Validaciones físicas de OcclusionTracker
        validated_matches = self._validate_matches(kalman_predictions)
        
        # 3. Actualización solo si pasa validaciones
        for match in validated_matches:
            self._update_with_kalman(match)
```

---

## 📞 Última Opción

Si nada funciona:
1. Revisa que `SKIP_FRAMES = 1` (crítico)
2. Verifica que YOLO detecta bien (`--debug`)
3. Considera usar YOLO11m o YOLO11x (más preciso)
4. Ajusta `CONFIDENCE_THRESHOLD` (probar 0.35-0.50)

---

**¿Necesitas ayuda específica?** Comparte:
- Tipo de escena (urbano/autopista/parking)
- Problemas observados (muchos IDs / saltos / drift)
- Valores actuales de parámetros
