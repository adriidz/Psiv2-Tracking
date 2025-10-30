# detection_frames.py - Actualizado
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from improved_tracker import Tracker, yolo_result_to_detections
from VehicleCounter import VehicleCounter

CAR_CLASS_ID = 2  # COCO: 2 = car

# Importar configuración del main si existe
try:
    from __main__ import MIN_HITS
except ImportError:
    # Valores por defecto si no se importan
    MIN_HITS = 1

YOLO_DIR = Path(__file__).resolve().parent
SETTINGS["runs_dir"] = str(YOLO_DIR / "runs")
SETTINGS["weights_dir"] = str(YOLO_DIR / "weights")
Path(SETTINGS["runs_dir"]).mkdir(parents=True, exist_ok=True)
Path(SETTINGS["weights_dir"]).mkdir(parents=True, exist_ok=True)

# --- Constants for playback ---
PLAYBACK_OPTIONS = ["fast", "1x"]
PLAYBACK = PLAYBACK_OPTIONS[0]  # "1x" for real-time, "fast" for as-fast-as-possible

def draw_boxes(frame, result, label_suffix=""):
    """Dibuja bounding boxes de detecciones YOLO."""
    if result is None or result.boxes is None:
        return frame
    names = result.names
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls_id = int(b.cls)
        conf = float(b.conf)
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}{label_suffix}"
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def window_closed(win_name: str) -> bool:
    """Returns True if the window was closed by the user (clicking the X)."""
    try:
        v = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        a = cv2.getWindowProperty(win_name, cv2.WND_PROP_AUTOSIZE)
        return v < 1 or a < 0
    except cv2.error:
        return True

def parse_args():
    """Esta función ya no se usa aquí, se define en main.py."""
    pass

def open_capture(video_path: Path):
    """Abre el video para captura."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {video_path}")
    return cap

def prepare_writer(cap: cv2.VideoCapture):
    """Prepara el VideoWriter para guardar el output."""
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_dir = Path(SETTINGS["runs_dir"]) / "cars_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cars_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
    if not writer.isOpened():
        out_path = out_dir / f"cars_{ts}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
    return writer, out_path, width, height, fps_in

def init_model(weights_path: str):
    """Inicializa el modelo YOLO."""
    return YOLO(weights_path)

def setup_display_if_needed(display: bool, width: int, height: int):
    """Configura la ventana de display."""
    if not display:
        return None
    win_name = "cars"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        from ctypes import windll
        user32 = windll.user32
        user32.SetProcessDPIAware()
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except Exception:
        screen_w, screen_h = 1920, 1080

    scale = min(0.8 * screen_w / width, 0.8 * screen_h / height, 1.0)
    win_w = max(1, int(width * scale))
    win_h = max(1, int(height * scale))
    cv2.resizeWindow(win_name, win_w, win_h)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except cv2.error:
        pass
    return win_name

def process_frames(cap: cv2.VideoCapture, writer: cv2.VideoWriter, model, args, 
                   width: int, height: int, fps_in: float, out_path: Path, tracker: Tracker):
    """Procesa todos los frames del video."""
    
    # Contador horizontal (cuenta ambas direcciones: arriba-abajo)
    counter_horizontal = VehicleCounter(
        line_position=2/3, 
        margin=5, 
        orientation='horizontal'
    )
    counter_horizontal.set_line_position(height)

    # Línea vertical izquierda (solo cuenta der->izq)
    counter_left = VehicleCounter(
        line_position=0.3, 
        margin=3, 
        orientation='vertical', 
        direction='right_to_left'
    )
    counter_left.set_line_position(height, width)

    # Línea vertical derecha (solo cuenta izq->der)
    counter_right = VehicleCounter(
        line_position=0.89, 
        margin=3, 
        orientation='vertical', 
        direction='left_to_right'
    )
    counter_right.set_line_position(height, width)

    frame_period = 1.0 / (fps_in if fps_in > 0 else 30.0)
    next_frame_ts = time.perf_counter() + frame_period

    last_result = None
    frame_idx = 0
    t0 = time.time()
    win_name = "cars" if args.display else None

    print(f"🎬 Iniciando procesamiento...")
    print(f"   Presiona 'Q' o 'ESC' para detener")
    print("-" * 70)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_shape = frame.shape

        # Inferencia cada N frames (según args.skip)
        if args.skip <= 1 or frame_idx % args.skip == 0:
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                classes=[CAR_CLASS_ID],
                verbose=False
            )
            last_result = results[0] if results else None
            if last_result is not None and frame_idx % 30 == 0:  # Log cada 30 frames
                print(f"Frame {frame_idx}: {len(last_result.boxes)} cars detected")

        # Convertir detecciones
        detections = yolo_result_to_detections(last_result) if last_result is not None else []
        
        # Actualizar tracker
        track_ids = tracker.update(frame, detections)

        # Actualizar contadores
        for track_id, car in track_ids.items():
            if car.centroids:
                center_x, center_y = car.centroids[-1]

                # Contador horizontal (línea verde completa)
                counter_horizontal.update(
                    track_id,
                    center_x=int(center_x),
                    center_y=int(center_y),
                    line_start=0.0,
                    line_end=1.0,
                    frame_shape=frame_shape
                )

                # Línea vertical izquierda (roja)
                counter_left.update(
                    track_id,
                    center_x=int(center_x),
                    center_y=int(center_y),
                    line_start=0.2,
                    line_end=0.4,
                    frame_shape=frame_shape
                )

                # Línea vertical derecha (azul)
                counter_right.update(
                    track_id,
                    center_x=int(center_x),
                    center_y=int(center_y),
                    line_start=0.2,
                    line_end=0.4,
                    frame_shape=frame_shape
                )

        # Dibujar tracks (simplificado)
        annotated = tracker.draw_tracks(frame.copy(), min_hits=MIN_HITS)

        # Dibujar líneas de conteo
        counter_horizontal.draw(annotated, color=(0, 255, 0), label_y_start=30, 
                               line_start=0.0, line_end=1.0)
        counter_left.draw(annotated, color=(255, 0, 0), label_y_start=90, 
                         line_start=0.2, line_end=0.4)
        counter_right.draw(annotated, color=(0, 0, 255), label_y_start=120, 
                          line_start=0.2, line_end=0.4)

        # Guardar frame
        writer.write(annotated)

        # Mostrar si display está activo
        if args.display:
            if window_closed(win_name):
                print("\n🛑 Ventana cerrada por el usuario")
                break
            cv2.imshow(win_name, annotated)

            if PLAYBACK == "1x":
                now = time.perf_counter()
                delay_s = max(0.0, next_frame_ts - now)
                delay_ms = max(1, int(delay_s * 1000))
                key = cv2.waitKey(delay_ms) & 0xFF
                next_frame_ts += frame_period
            else:
                key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("\n🛑 Detenido por el usuario")
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"📊 Estadísticas:")
    print(f"   • Frames procesados: {frame_idx}")
    print(f"   • Tiempo total: {elapsed:.1f}s")
    print(f"   • FPS promedio: {frame_idx/elapsed:.1f}")
    print(f"   • Video guardado: {out_path}")
    print("-" * 70)
    print(f"🚦 Conteo de Vehículos:")
    print(f"   Horizontal (Verde):")
    print(f"      ⬇️  Arriba → Abajo: {counter_horizontal.count_forward}")
    print(f"      ⬆️  Abajo → Arriba: {counter_horizontal.count_backward}")
    print(f"   Vertical Izquierda (Roja):")
    print(f"      ⬅️  Derecha → Izquierda: {counter_left.count_backward}")
    print(f"   Vertical Derecha (Azul):")
    print(f"      ➡️  Izquierda → Derecha: {counter_right.count_forward}")
    print("=" * 70)
    
    return frame_idx, elapsed, out_path