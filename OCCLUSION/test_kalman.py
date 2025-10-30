"""
Script de prueba rápida para el KalmanTracker mejorado
Ejecuta el tracker en tu video con configuración optimizada
"""
import sys
from pathlib import Path

# Asegurar que el directorio esté en el path
sys.path.insert(0, str(Path(__file__).parent))

# Configuración para prueba rápida
VIDEO_PATH = Path("videos/output3.mp4")
WEIGHTS_PATH = "weights/yolo11s.pt"

# Parámetros OPTIMIZADOS para Kalman
CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 960
SKIP_FRAMES = 1  # CRÍTICO: procesar todos los frames

# Parámetros del KalmanTracker (OPTIMIZADOS)
IOU_THRESHOLD = 0.25
MAX_LOST_FRAMES = 15
MIN_HITS = 3
MAHALANOBIS_THRESHOLD = 3.5

# Visualización
SHOW_DISPLAY = True
DEBUG_MODE = False

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 PRUEBA DE KALMAN TRACKER MEJORADO")
    print("=" * 80)
    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🤖 Modelo: {WEIGHTS_PATH}")
    print(f"🎯 Confianza: {CONFIDENCE_THRESHOLD}")
    print(f"📏 Imagen: {IMAGE_SIZE}px")
    print(f"⏭️  Skip frames: {SKIP_FRAMES}")
    print("-" * 80)
    print(f"🔵 Tracker: KalmanTracker (MEJORADO)")
    print(f"   • IoU threshold: {IOU_THRESHOLD}")
    print(f"   • Max lost: {MAX_LOST_FRAMES} frames")
    print(f"   • Min hits: {MIN_HITS}")
    print(f"   • Mahalanobis: {MAHALANOBIS_THRESHOLD}")
    print("-" * 80)
    print(f"💡 Mejoras implementadas:")
    print(f"   ✅ Estado 7D: [x, y, w, h, vx, vy, vw]")
    print(f"   ✅ Predicción de tamaño adaptativo")
    print(f"   ✅ Actualización completa [x,y,w,h]")
    print(f"   ✅ Bbox suavizado con Kalman")
    print(f"   ✅ Distancia Mahalanobis 4D")
    print(f"   ✅ Manejo robusto de errores")
    print("-" * 80)
    print(f"🎨 Colores de visualización:")
    print(f"   🟢 Verde: Track activo (con detección)")
    print(f"   🔵 Azul claro: Predicción Kalman (debug)")
    print(f"   🔴 Naranja: Track en oclusión (solo predicción)")
    print("-" * 80)
    print(f"ℹ️  Presiona 'Q' o 'ESC' para detener")
    print("=" * 80)
    print()
    
    # Importar y ejecutar main
    from main import main
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
