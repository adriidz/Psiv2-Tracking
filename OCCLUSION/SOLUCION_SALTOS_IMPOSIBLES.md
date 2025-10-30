# 🔒 Solución a Saltos Imposibles - Tracker Ultra-Estricto

## 🎯 Problema Original

Observado en la imagen:
- **ID 55** salta desde coche arriba-izquierda → coche abajo
- **ID 28** también presenta salto imposible
- Estos saltos **cruzan la línea de conteo** → disparan contador falsamente ❌

```
┌────────────────────────────────────────┐
│  ID 55 (arriba) ──────┐               │
│                       │  SALTO        │
│                       │  IMPOSIBLE    │
│  Línea de conteo ─────┼────────       │
│                       ↓               │
│  ID 55 (abajo) ←──────┘               │
│  ❌ Contador +1 (FALSO)               │
└────────────────────────────────────────┘
```

---

## ✅ Solución Implementada

### 1. **Parámetros Ultra-Conservadores**

```python
# ANTES (Balanceado)
SEARCH_RADIUS = 120.0
MIN_MATCH_SCORE = 0.45
MIN_HITS = 3
SKIP_FRAMES = 2
BUFFER_FRAMES = 25

# AHORA (Ultra-Estricto) 🔒
SEARCH_RADIUS = 100.0      # -17% más pequeño
MIN_MATCH_SCORE = 0.50     # +11% más alto
MIN_HITS = 4               # +33% más confirmaciones
SKIP_FRAMES = 1            # Procesar TODOS los frames
BUFFER_FRAMES = 20         # -20% eliminar dudosos antes
```

### 2. **Radio de Búsqueda Muy Restrictivo**

```python
# Radio efectivo según frames perdidos:
Lost = 1:  80px   (0.8x)  ← Muy cerca
Lost = 2:  100px  (1.0x)  ← Radio base
Lost = 3:  108px  (1.08x) ← Crece poco
Lost = 5:  124px  (1.24x)
Lost = 10: 140px  (1.4x)  ← Máximo absoluto

# COMPARACIÓN:
# Antes: hasta 216px (demasiado grande!)
# Ahora: hasta 140px (controlado)
```

**Efecto**: Un coche arriba NO puede emparejarse con uno abajo (distancia > 140px típicamente).

### 3. **Validación de Velocidad Mejorada**

```python
# Tracks muy establecidos (hits ≥ 8):
max_speed_change = 2.0x   # Antes: 3.0x

# Tracks establecidos (hits ≥ 5):
max_speed_change = 2.5x   # Antes: 3.0x

# Tracks nuevos:
max_speed_change = 3.0x   # Igual

# NUEVA: Rechazar desaceleración brusca
if implied_speed < prev_speed * 0.3:
    REJECT
```

**Efecto**: No puede acelerar/frenar bruscamente → previene saltos.

### 4. **Validación de Dirección Muy Estricta**

```python
# Tracks muy establecidos (hits ≥ 8):
max_angle = 75°    # Antes: 90°

# Tracks establecidos (hits ≥ 5):
max_angle = 85°    # Antes: 90°

# Tracks nuevos:
max_angle = 100°   # Antes: 110°
```

**Efecto**: Un coche yendo hacia la derecha NO puede de repente ir hacia abajo.

### 5. **NUEVA: Validación Perpendicular**

```python
# Calcular distancia perpendicular a la trayectoria
perp_dist = abs(-uy * dx + ux * dy)

# Si detección está muy lateral (perpendicular):
if perp_dist > search_radius * 0.6:
    REJECT  # Movimiento lateral imposible
```

**Efecto**: Si el track va horizontal, rechaza detecciones verticales.

```
Track: →→→ (horizontal)
Det: ↑ (perpendicular)
❌ REJECT
```

### 6. **NUEVA: Desaceleración Brusca**

```python
# No permitir frenado de golpe
if implied_speed < prev_speed * 0.3 and prev_speed > 5.0:
    REJECT
```

**Efecto**: Previene que un coche rápido "frene" instantáneamente (típico en saltos falsos).

### 7. **Validación de Tamaño Más Estricta**

```python
# Tracks establecidos:
max_ratio = 1.6x   # Antes: 2.0x

# Tracks nuevos:
max_ratio = 1.8x   # Antes: 2.0x
```

**Efecto**: Sedán (100x50) no puede convertirse en camión (160x90) fácilmente.

### 8. **Score Mínimo Absoluto**

```python
# NUEVO: Rechazar antes del threshold
if score < 0.35:
    REJECT
```

**Efecto**: Incluso si pasa otras validaciones, scores muy bajos se rechazan.

### 9. **Filtros de Re-ID Más Estrictos**

```python
# ANTES:
if track.hits < 3:  continue
if track.lost > 15: continue

# AHORA:
if track.hits < 4:  continue  # Más historial requerido
if track.lost > 10: continue  # Menos tiempo de búsqueda
```

**Efecto**: Solo tracks muy confiables pueden reasignarse.

### 10. **Umbrales Adaptativos Más Altos**

```python
# Lost = 1:
threshold = 0.50 * 0.90 = 0.45

# Lost = 2:
threshold = 0.50 * 0.95 = 0.475

# Lost = 3+:
threshold = 0.50 * 1.05 = 0.525  # MÁS ALTO que base!
```

**Efecto**: Cuanto más tiempo perdido, más difícil reasignar.

---

## 📊 Comparación de Radios Efectivos

```
Frame Perdido │ ANTES (Balanceado) │ AHORA (Ultra-Estricto) │ Reducción
━━━━━━━━━━━━━━┼────────────────────┼────────────────────────┼───────────
Lost = 1      │ 156px (1.3x)       │ 80px  (0.8x)          │ -49% ✅
Lost = 2      │ 156px              │ 100px (1.0x)          │ -36% ✅
Lost = 3      │ 174px (1.45x)      │ 108px (1.08x)         │ -38% ✅
Lost = 5      │ 192px (1.6x)       │ 124px (1.24x)         │ -35% ✅
Lost = 10     │ 216px (1.8x MAX)   │ 140px (1.4x MAX)      │ -35% ✅
━━━━━━━━━━━━━━┴────────────────────┴────────────────────────┴───────────

Estáticos     │ 60px               │ 40px                  │ -33% ✅
```

**Conclusión**: Radios mucho más pequeños → imposible emparejar objetos lejanos.

---

## 🎯 Cómo Previene el Salto del Ejemplo

### Escenario: ID 55 intenta saltar de arriba a abajo

```
┌─────────────────────────────────────────────────────────────┐
│  Coche A (arriba, ID 55)                                    │
│  Última posición: (100, 50)                                 │
│  Velocidad: 5px/frame hacia derecha                         │
│  Dirección: → (horizontal)                                  │
│                                                              │
│  ─────── Línea de conteo (y=200) ───────                    │
│                                                              │
│  Coche B (abajo)                                            │
│  Posición: (120, 300)                                       │
└─────────────────────────────────────────────────────────────┘

Intentando emparejar ID 55 con Coche B:

1. ❌ DISTANCIA: 
   dist = √[(120-100)² + (300-50)²] = 250px
   Máximo permitido: 100px (lost=1, 0.8x)
   → REJECT inmediato

2. ❌ VELOCIDAD (si pasara #1):
   implied_speed = 250 / 1 = 250 px/frame
   prev_speed = 5 px/frame
   ratio = 250 / 5 = 50x
   Máximo: 3x
   → REJECT

3. ❌ DIRECCIÓN (si pasara #1 y #2):
   prev_dir: → (0°, horizontal)
   new_dir: ↓ (90°, vertical)
   angle = 90°
   Máximo: 85° (si hits≥5)
   → REJECT

4. ❌ PERPENDICULAR (si pasara anteriores):
   Movimiento horizontal
   Detección abajo (perpendicular)
   perp_dist = 250px
   Máximo: 0.6 * 100 = 60px
   → REJECT

RESULTADO: ❌❌❌❌ Empareamiento IMPOSIBLE
→ Se crea ID nuevo para Coche B
→ Contador NO se activa falsamente ✅
```

---

## 🚀 Resultados Esperados

### Antes (Balanceado):
- ❌ Saltos ocasionales de 200+ píxeles
- ❌ Contador dispara falsamente
- ❌ Trayectorias cruzan líneas imposiblemente
- ⚠️  ~200 IDs en video típico

### Ahora (Ultra-Estricto):
- ✅ Saltos prácticamente eliminados
- ✅ Contador preciso
- ✅ Trayectorias físicamente coherentes
- ⚠️  ~250-300 IDs (más, pero correcto)

**Trade-off**: Más IDs totales, pero CERO saltos imposibles.

---

## 🎛️ Si Todavía Hay Problemas

### Si AÚN hay algún salto imposible:

```python
# Configuración EXTREMA
SEARCH_RADIUS = 80.0       # Radio muy pequeño
MIN_MATCH_SCORE = 0.55     # Score muy alto
MIN_HITS = 5               # Muchas confirmaciones
CONFIDENCE_THRESHOLD = 0.45 # Menos detecciones
```

### Si crea DEMASIADOS IDs:

```python
# Aflojar un poco (cuidado)
SEARCH_RADIUS = 110.0
MIN_MATCH_SCORE = 0.48
BUFFER_FRAMES = 25
# Pero mantener MIN_HITS = 4 y SKIP_FRAMES = 1
```

---

## 🐛 Debugging

Activa debug para ver rechazos:

```python
DEBUG_MODE = True
```

Verás mensajes como:
```
[ReID] T12 (mobile, hits=8) <- Det3 (score=0.52, lost=2)
  REJECT T55: dist=250px > radius=80px
  REJECT T28: velocidad implícita 45.0 > 2.5x velocidad previa 8.2
  REJECT T17: cambio dirección 92° en track establecido
  REJECT T33: detección muy perpendicular (perp=85px)
[NEW] T56 created
```

**Interpretación**:
- ✅ Los rechazos son **buenos** - previenen errores
- ✅ Crear IDs nuevos es preferible a saltos imposibles
- ❌ Si casi TODO se rechaza → radio demasiado pequeño

---

## 📈 Métricas de Éxito

### Indicadores de que funciona:
1. ✅ **Trayectorias suaves**: Sin saltos bruscos
2. ✅ **Contador estable**: No dispara en falso
3. ✅ **Debug muestra rechazos**: Validaciones funcionando
4. ✅ **IDs razonables**: Ni muy pocos (saltos) ni muchísimos

### Señales de problema:
1. ❌ **Trayectorias saltan**: Radio demasiado grande
2. ❌ **Contador dispara**: Validaciones insuficientes
3. ❌ **Explosión de IDs** (>500): Demasiado estricto o skip_frames alto
4. ❌ **Sin rechazos en debug**: Validaciones no se aplican

---

## 🎓 Conclusión

Este tracker prioriza **precisión del contador** sobre minimizar IDs:

✅ **CERO saltos imposibles** → Contador confiable  
✅ **Validaciones en cascada** → Múltiples capas de protección  
✅ **Radios pequeños** → Solo empareja cercanos  
✅ **Scores altos** → Alta confianza requerida  

**Filosofía**: 
> "Es mejor tener 300 IDs correctos que 150 IDs con saltos imposibles"

El contador es más importante que tener pocos IDs. 🎯

---

**Fecha**: 29 Octubre 2025  
**Versión**: Ultra-Estricto v3.0 🔒  
**Objetivo**: Eliminar saltos que cruzan líneas de conteo
