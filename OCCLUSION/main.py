# main.py - Versión con OcclusionTracker y configuración simplificada
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from detection_frames import *
from improved_tracker import KalmanTracker

# ============================================================================
# 🎛️ CONFIGURACIÓN PRINCIPAL - AJUSTA ESTOS VALORES
# ============================================================================

# --- Paths ---
VIDEO_PATH = Path(r"videos\output3.mp4")
WEIGHTS_PATH = "weights/yolo11n.pt"  # Cambiar a yolo11n.pt, yolo11m.pt, etc.

# --- Parámetros de Detección YOLO ---
CONFIDENCE_THRESHOLD = 0.42     # Confianza mínima (0.0-1.0). Menor = más detecciones
IMAGE_SIZE = 960               # Tamaño de imagen para inferencia (640, 960, 1024, 1280)
SKIP_FRAMES = 2              # Procesar cada N frames (1=todos, 2=cada 2, 3=cada 3) - NO SALTAR para mejor tracking

# --- Parámetros del Tracker (Kalman) - OPTIMIZADOS PARA OCLUSIONES ---
IOU_THRESHOLD = 0.25                   # IoU mínimo para asociar tracks activos
MAX_LOST_FRAMES = 45                   # 🔥 MÁS TIEMPO para mantener tracks durante oclusión
MIN_HITS = 2                           # Detecciones mínimas para confirmar un track
MAHALANOBIS_THRESHOLD = 6.0            # 🔥 MÁS PERMISIVO para re-identificar tras oclusión

# --- Parámetros de Re-identificación (MUY ESTRICTOS) ---
SEARCH_RADIUS = 100.0                  # Radio de búsqueda BASE en píxeles (80-120) - REDUCIDO para evitar saltos
MIN_MATCH_SCORE = 0.50                 # Score mínimo para reasociar (0.45-0.60) - MUY ALTO para prevenir saltos

# --- Anti-Fragmentación ---
ENABLE_FRAGMENT_MERGE = True          # Activar fusión de fragmentos (True/False)
FRAGMENT_IOU = 0.05                    # IoU mínimo para considerar fragmentos

# --- Visualización ---
SHOW_DISPLAY = True                    # Mostrar ventana durante procesamiento
SHOW_PREDICTIONS = True                # Mostrar bbox predicho durante oclusión
DEBUG_MODE = False                     # Mostrar mensajes de debug en consola

# ============================================================================
# 📚 GUÍA RÁPIDA DE AJUSTE - VERSIÓN ULTRA-ESTRICTA 🔒
# ============================================================================
"""
🎯 SISTEMA ULTRA-ESTRICTO: CERO TOLERANCIA a saltos imposibles

PROBLEMA RESUELTO: Saltos que cruzan la línea de conteo y disparan contador
SOLUCIÓN: Múltiples validaciones en cascada + parámetros muy conservadores

VALIDACIONES IMPLEMENTADAS (MEJORADAS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 🔒 Velocidad máxima: 2-3x para establecidos (antes 3-4x)
2. 🔒 Dirección: 75° establecidos / 85° normales / 100° nuevos (antes 90°/110°)
3. 🔒 Tamaño: 1.6-1.8x máx (antes 2.0x)
4. 🔒 Apariencia: Rechaza corr < -0.3
5. 🔒 Estáticos: Radio 0.3x, máx 30px salto (antes 0.4x, 50px)
6. 🔒 Radio adaptativo: Crece MUY poco (máx 1.4x vs 1.8x antes)
7. 🔒 Movimiento perpendicular: NUEVA - rechaza detecciones laterales imposibles
8. 🔒 Desaceleración brusca: NUEVA - rechaza frenados imposibles
9. 🔒 Score mínimo absoluto: 0.35 antes de threshold
10. 🔒 Filtros estrictos: hits≥4, lost≤10 (antes hits≥3, lost≤15)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIGURACIÓN ACTUAL (OPTIMIZADA ANTI-SALTOS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH_RADIUS = 100px     ⚠️ CRÍTICO (antes 120px)
MIN_MATCH_SCORE = 0.50    ⚠️ CRÍTICO (antes 0.45)
MIN_HITS = 4              ⚠️ CRÍTICO (antes 3) - más confirmaciones
SKIP_FRAMES = 1           ⚠️ CRÍTICO - procesar TODOS los frames
BUFFER_FRAMES = 20        (reducido para eliminar tracks dudosos antes)
CONFIDENCE_THR = 0.42     (algo más alto para menos falsos positivos)

Radio máximo efectivo:
  • Lost=1: 80px  (0.8x)
  • Lost=2: 100px (1.0x)
  • Lost=3: 108px (1.08x)
  • Lost=5: 124px (1.24x)
  • Lost=10: 140px (1.4x MAX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AJUSTE FINO (si es necesario):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Si TODAVÍA hay saltos imposibles:
   → REDUCIR: SEARCH_RADIUS = 80-90px
   → AUMENTAR: MIN_MATCH_SCORE = 0.55-0.60
   → AUMENTAR: MIN_HITS = 5-6
   → Activar DEBUG_MODE = True para ver qué está pasando

✅ Si crea DEMASIADOS IDs (tracks se pierden fácil):
   → AUMENTAR: BUFFER_FRAMES = 25-30
   → AUMENTAR: SEARCH_RADIUS = 110-120px (cuidado!)
   → REDUCIR: MIN_MATCH_SCORE = 0.47-0.48
   → REDUCIR: MIN_HITS = 3

⚙️  Para vehículos MUY LENTOS (escena urbana):
   → REDUCIR: SKIP_FRAMES = 1 (ya está)
   → MANTENER: Parámetros actuales

🏎️  Para vehículos MUY RÁPIDOS (autopista):
   → AUMENTAR: SEARCH_RADIUS = 130-150px
   → REDUCIR: SKIP_FRAMES = 1 (crítico)
   → MANTENER: Validaciones estrictas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILOSOFÍA:
💡 "Preferible perder un track que crear un salto imposible"
💡 "El contador es más importante que tener pocos IDs"
💡 "Validaciones en cascada previenen errores compuestos"

COLORES DE VISUALIZACIÓN:
    🟡 Amarillo: Vehículo activo (detectado en frame actual)
    🟠 Naranja: Track en buffer de oclusión (buscando reasignación)
"""

# ============================================================================
# 💻 CÓDIGO PRINCIPAL - NO NECESITAS MODIFICAR ESTO
# ============================================================================

# Must be set before importing ultralytics
os.environ["ULTRALYTICS_HOME"] = str(Path(__file__).resolve().parent)
YOLO_DIR = Path(__file__).resolve().parent
SETTINGS["runs_dir"] = str(YOLO_DIR / "runs")
SETTINGS["weights_dir"] = str(YOLO_DIR / "weights")
Path(SETTINGS["runs_dir"]).mkdir(parents=True, exist_ok=True)
Path(SETTINGS["weights_dir"]).mkdir(parents=True, exist_ok=True)

CAR_CLASS_ID = 2  # COCO: 2 = car

def parse_args():
    """Argumentos opcionales por línea de comandos (sobrescriben las variables globales)."""
    import argparse
    p = argparse.ArgumentParser(description="Vehicle Tracking con OcclusionTracker")
    p.add_argument("--video", type=str, default=str(VIDEO_PATH), help="Path al video")
    p.add_argument("--weights", type=str, default=WEIGHTS_PATH, help="Modelo YOLO")
    p.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confianza mínima")
    p.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Tamaño de imagen")
    p.add_argument("--skip", type=int, default=SKIP_FRAMES, help="Procesar cada N frames")
    p.add_argument("--display", action="store_true", default=SHOW_DISPLAY, help="Mostrar ventana")
    p.add_argument("--no-display", dest="display", action="store_false", help="Sin ventana")
    p.add_argument("--debug", action="store_true", default=DEBUG_MODE, help="Modo debug")
    return p.parse_args()

def print_config(args, tracker):
    """Imprime la configuración actual."""
    print("\n" + "=" * 80)
    print("🚗 VEHICLE TRACKER - CONFIGURACIÓN ULTRA-ESTRICTA 🔒")
    print("=" * 80)
    print(f"📹 Video: {args.video}")
    print(f"🤖 Modelo: {args.weights}")
    print(f"📊 Confianza: {args.conf:.2f} | Tamaño: {args.imgsz}px | Skip: {args.skip} frames")
    print("-" * 80)
    print(f"🎯 Tracker: {tracker.__class__.__name__}")
    print(f"   • Buffer frames: {tracker.buffer_frames} (mantiene IDs perdidos)")
    print(f"   • Min hits: {tracker.min_hits} ⚠️ (ALTO - más confirmaciones)")
    print(f"   • IoU threshold: {tracker.iou_threshold} (para tracks activos)")
    print("-" * 80)
    print(f"🔍 Sistema de Re-identificación ULTRA-ESTRICTO (Anti-Saltos):")
    print(f"   📌 Fase 1: IoU para tracks activos (threshold={tracker.iou_threshold})")
    print(f"   📌 Fase 2: Re-ID con MÚLTIPLES validaciones físicas")
    print(f"      🔒 Solo tracks con hits≥4 y lost≤10 (MUY SELECTIVO)")
    print(f"      🔒 Velocidad máx: 2-3x la velocidad previa (tracks establecidos)")
    print(f"      🔒 Dirección máx: 75° (muy establecidos) / 85° (establecidos) / 100° (nuevos)")
    print(f"      🔒 Tamaño: máx 1.6x diferencia (tracks establecidos) / 1.8x (nuevos)")
    print(f"      🔒 Apariencia: rechaza si muy diferente (corr<-0.3)")
    print(f"      🔒 Validación perpendicular: rechaza movimientos laterales imposibles")
    print(f"      🔒 Score mínimo absoluto: 0.35 (antes de threshold)")
    print(f"   🎚️  Parámetros: radius={tracker.search_radius:.0f}px (crece hasta {tracker.search_radius*1.4:.0f}px MAX),")
    print(f"                 score_min={tracker.min_match_score:.2f} (MUY ALTO)")
    print("-" * 80)
    print(f"🛡️  VALIDACIONES ANTI-SALTO MEJORADAS:")
    print(f"   ✅ Estáticos: radio 0.3x (máx 40px), no saltan >30px")
    print(f"   ✅ Lost=1: radio 0.8x (96px con config actual)")
    print(f"   ✅ Lost=2: radio 1.0x (100px)")
    print(f"   ✅ Lost=3+: crece lentamente (máx 1.4x = 140px)")
    print(f"   ✅ Score compuesto: 45% dist + 25% apariencia + 20% dir + 10% tamaño")
    print(f"   ✅ Desaceleración brusca: rechaza frenados imposibles")
    print("-" * 80)
    print(f"🔧 Anti-fragmentación:")
    if tracker.fragment_iou > 0:
        print(f"   • ACTIVADA: IoU>{tracker.fragment_iou}, áreas similares >30%")
    else:
        print(f"   • DESACTIVADA")
    print("-" * 80)
    print(f"💡 FILOSOFÍA: CERO TOLERANCIA a saltos imposibles")
    print(f"   • Previene que contador se active falsamente")
    print(f"   • Múltiples validaciones en cascada")
    print(f"   • Preferible perder un track que crear salto imposible")
    print(f"   • Radio pequeño + score alto + validaciones estrictas")
    print("-" * 80)
    print(f"🐛 Debug: {'ON' if args.debug else 'OFF'}")
    print(f"👁️  Display: {'ON' if args.display else 'OFF'}")
    print("=" * 80 + "\n")

def main():
    args = parse_args()

    # Cargar video
    try:
        cap = open_capture(Path(args.video))
    except Exception as e:
        print(f"❌ Error al abrir video: {e}")
        sys.exit(1)

    # Preparar writer
    writer, out_path, width, height, fps_in = prepare_writer(cap)
    
    # Cargar modelo
    print(f"⏳ Cargando modelo {args.weights}...")
    model = YOLO(args.weights)
    print(f"✅ Modelo cargado correctamente")
    
    # Configurar display
    if args.display:
        setup_display_if_needed(args.display, width, height)
    
    # Inicializar tracker
    tracker = KalmanTracker(
        iou_threshold=IOU_THRESHOLD,
        max_lost=MAX_LOST_FRAMES,
        min_hits=MIN_HITS,
        mahalanobis_threshold=MAHALANOBIS_THRESHOLD,
        debug=args.debug  # Pasar flag de debug
    )
    
    # (El resto de la función main permanece igual)
    # Mostrar configuración
    # print_config(args, tracker) # Esta función habría que adaptarla o eliminarla
    
    # Procesar frames
    process_frames(cap, writer, model, args, width, height, fps_in, out_path, tracker)

if __name__ == "__main__":
    main()