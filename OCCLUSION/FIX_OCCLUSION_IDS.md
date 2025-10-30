# 🔧 SOLUCIÓN: IDs Cambian Tras Oclusión

## ✅ CAMBIOS REALIZADOS

### 1. **Parámetros Más Permisivos**
```python
MAX_LOST_FRAMES = 30        # Era 20 → Mantiene tracks más tiempo
MAHALANOBIS_THRESHOLD = 6.0 # Era 4.0 → Más permisivo para re-ID
```

### 2. **Distancia Física Adaptativa**
```python
# Antes: 200px fijo
# Ahora: 150px + (lost * 15px), máx 400px

lost=1  → 165px
lost=5  → 225px
lost=10 → 300px
lost=15 → 375px
```

### 3. **Threshold Adaptativo**
```python
# Tracks perdidos >15 frames: threshold × 1.5
# Ejemplo: 6.0 × 1.5 = 9.0 (muy permisivo)
```

### 4. **Filtro de Kalman Optimizado**
```python
Q[:2] *= 0.1    # Confía más en predicción de posición
Q[4:6] *= 5.0   # Permite cambios suaves de velocidad
R[:2] *= 5.0    # Balance entre modelo y medición
```

### 5. **Validaciones Relajadas**
```python
# Cambio de tamaño: 2.5x → 3.0x
# Tracks mínimos: 3 → 2 (permite re-ID más temprana)
```

---

## 🧪 CÓMO VERIFICAR QUE FUNCIONA

### Test 1: Ejecutar con Debug
```powershell
python main.py --debug
```

**Busca en consola:**
```
[Match candidate] T5 (lost=3) <-> Det2: dist=4.2
[✅ RE-ID] T5 (lost=3) <-> Det2: dist=4.2
```

✅ **SI VES ESTO**: Re-identificación funciona  
❌ **SI NO VES**: Ajusta parámetros (ver abajo)

### Test 2: Script Específico
```powershell
python test_occlusion.py
```

### Test 3: Observar Visualización
1. Cuando vehículo entra en oclusión → **Bbox naranja** (predicción)
2. Predicción debe moverse en dirección esperada
3. Cuando reaparece → **Mismo ID** si predicción fue buena

---

## 🔧 SI SIGUE SIN FUNCIONAR

### Problema: "Nunca veo [Match candidate]"

**Causa**: Predicción está MUY lejos de donde reaparece

**Solución 1**: Aumentar threshold aún más
```python
# En main.py
MAHALANOBIS_THRESHOLD = 8.0  # Era 6.0
```

**Solución 2**: Aumentar distancia física
```python
# En improved_tracker.py, línea ~880
max_physical_dist = 200.0 + (track.lost * 20.0)  # Era 15.0
max_physical_dist = min(max_physical_dist, 500.0)  # Era 400.0
```

### Problema: "Veo [Match candidate] pero no [RE-ID]"

**Causa**: Otro track tiene mejor score

**Solución**: Activar debug y ver todos los candidatos
```python
# En improved_tracker.py, buscar:
if self.debug and dist < self.mahalanobis_threshold:
# Cambiar a:
if self.debug:  # Mostrar TODOS
```

### Problema: "Predicción va en dirección incorrecta"

**Causa**: Velocidad mal estimada o aceleración fuerte

**Solución 1**: Aumentar ruido de velocidad
```python
# En create_kalman_filter()
kf.Q[4:6, 4:6] *= 10.0  # Era 5.0 (permite más cambios)
```

**Solución 2**: Usar OcclusionTracker en vez de Kalman
```python
# En main.py, cambiar:
tracker = OcclusionTracker(
    iou_threshold=0.25,
    max_lost=30,
    buffer_frames=30,
    search_radius=150.0,  # Aumentado
    min_match_score=0.40,  # Reducido (más permisivo)
    debug=True
)
```

---

## 📊 COMPARACIÓN: Antes vs Ahora

| Parámetro | Antes | Ahora | Efecto |
|-----------|-------|-------|--------|
| Max lost | 20 | 30 | +50% tiempo |
| Mahalanobis | 4.0 | 6.0 | +50% distancia |
| Dist física | 200px | 150-400px | Adaptativo |
| Tamaño ratio | 2.5x | 3.0x | +20% tolerancia |
| Min hits | 3 | 2 | Re-ID más rápida |
| Threshold adapt | No | Sí (1.5x) | Más permisivo |

---

## 🎯 CASOS DE USO

### Oclusiones Cortas (1-5 frames)
✅ **Debería funcionar perfectamente** con parámetros actuales

### Oclusiones Medias (5-15 frames)
✅ **Funciona bien** si velocidad es constante  
⚠️ **Puede fallar** si vehículo cambia velocidad/dirección

### Oclusiones Largas (>15 frames)
⚠️ **Difícil** incluso con parámetros permisivos  
💡 **Considera**: Aceptar crear nuevo ID

---

## 💡 ÚLTIMO RECURSO

Si NADA funciona, el problema puede ser:

1. **YOLO detecta mal** al reaparecer (bbox muy diferente)
   → Ajusta `CONFIDENCE_THRESHOLD`

2. **Velocidad cambia drásticamente** durante oclusión
   → Kalman asume velocidad constante (no puede predecir aceleración)

3. **Oclusión demasiado larga**
   → Ningún tracker puede predecir >20 frames sin detección

4. **Múltiples vehículos reaparecen juntos**
   → Algoritmo húngaro puede asignar mal

**Solución pragmática**: 
```python
# Aumentar MAX_LOST_FRAMES pero aceptar algunos IDs nuevos
MAX_LOST_FRAMES = 40
# Es preferible algunos IDs extra que perder continuidad
```

---

## 🚀 PARA PROBAR AHORA

```powershell
# 1. Con debug
python main.py --debug

# 2. Sin debug (más rápido)
python main.py

# 3. Test específico
python test_occlusion.py
```

**Observa:**
- ¿Se mantiene el ID tras oclusión?
- ¿La predicción (naranja) es razonable?
- ¿Hay mensajes de RE-ID en consola?

---

**¿Necesitas más ayuda?** Dime:
1. Duración típica de oclusiones (frames)
2. ¿Velocidad constante o cambia?
3. ¿Un vehículo o varios reaparecen juntos?
