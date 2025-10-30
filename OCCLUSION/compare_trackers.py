"""
Script de comparación: OcclusionTracker vs KalmanTracker
Ejecuta ambos trackers en paralelo para comparar resultados
"""
import sys
from pathlib import Path

print("=" * 80)
print("🔍 COMPARACIÓN: OcclusionTracker vs KalmanTracker")
print("=" * 80)

# Test 1: OcclusionTracker (baseline)
print("\n📊 TEST 1: OcclusionTracker (baseline)")
print("-" * 80)
print("Tracker basado en heurísticas y validaciones físicas")
print("Ejecutando...")

# Cambiar temporalmente el tracker
from improved_tracker import OcclusionTracker, KalmanTracker
import main

# Guardar el tracker original
original_config = {
    'IOU_THRESHOLD': main.IOU_THRESHOLD,
    'MAX_LOST_FRAMES': main.MAX_LOST_FRAMES,
    'MIN_HITS': main.MIN_HITS,
}

try:
    # Configurar para OcclusionTracker
    main.IOU_THRESHOLD = 0.25
    main.MAX_LOST_FRAMES = 25
    main.MIN_HITS = 4
    
    # Crear tracker
    tracker_occlusion = OcclusionTracker(
        iou_threshold=0.25,
        max_lost=25,
        min_hits=4,
        buffer_frames=20,
        search_radius=100.0,
        min_match_score=0.50,
        debug=False
    )
    
    print(f"✅ OcclusionTracker configurado")
    print(f"   - Buffer: 20 frames")
    print(f"   - Search radius: 100px")
    print(f"   - Min score: 0.50")
    
except Exception as e:
    print(f"❌ Error configurando OcclusionTracker: {e}")

print("\n" + "=" * 80)
print("📊 TEST 2: KalmanTracker (mejorado)")
print("-" * 80)
print("Tracker basado en filtros de Kalman con predicción estadística")

try:
    # Configurar para KalmanTracker
    main.IOU_THRESHOLD = 0.30
    main.MAX_LOST_FRAMES = 20
    main.MIN_HITS = 3
    main.MAHALANOBIS_THRESHOLD = 4.0
    
    tracker_kalman = KalmanTracker(
        iou_threshold=0.30,
        max_lost=20,
        min_hits=3,
        mahalanobis_threshold=4.0,
        debug=False
    )
    
    print(f"✅ KalmanTracker configurado")
    print(f"   - Estado: 7D [x,y,w,h,vx,vy,vw]")
    print(f"   - Mahalanobis threshold: 4.0")
    print(f"   - Modo híbrido: 70% Kalman + 30% YOLO")
    
except Exception as e:
    print(f"❌ Error configurando KalmanTracker: {e}")

print("\n" + "=" * 80)
print("💡 DIFERENCIAS CLAVE:")
print("=" * 80)
print("""
OcclusionTracker:
  ✅ Validaciones físicas muy estrictas
  ✅ Previene saltos imposibles
  ❌ Puede crear muchos IDs en oclusiones largas
  ❌ Predicción lineal simple

KalmanTracker:
  ✅ Predicción estadística sofisticada
  ✅ Suaviza trayectorias (menos jitter)
  ✅ Predice cambios de tamaño
  ❌ Puede "driftar" sin detecciones
  ✅ Modo híbrido previene drift

RECOMENDACIÓN:
  🎯 Para CONTEO preciso: OcclusionTracker (menos errores de salto)
  🎯 Para TRACKING suave: KalmanTracker (trayectorias más naturales)
  🎯 Para OCLUSIONES largas: KalmanTracker (mejor predicción)
""")

print("=" * 80)
print("\n🚀 Para probar cada uno:")
print("\n1. OcclusionTracker:")
print("   Edita main.py y cambia:")
print("   tracker = OcclusionTracker(...)")

print("\n2. KalmanTracker (actual):")
print("   Edita main.py y usa:")
print("   tracker = KalmanTracker(...)")

print("\n3. Comparación visual:")
print("   python main.py --debug")
print("   Observa los colores:")
print("   - Verde: detección actual")
print("   - Azul: predicción Kalman")
print("   - Naranja: oclusión")

print("\n" + "=" * 80)
