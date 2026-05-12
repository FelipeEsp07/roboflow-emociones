"""
ui_app.py — Dashboard web moderno para roboflow_emociones.

Este módulo NO modifica el script original (roboflow_emociones.py).
Reutiliza sus clases (`RoboflowClient`, `InferenceWorker`, `DetectionSmoother`,
`FaceTracker`, `SessionLogger`, etc.) y construye encima una interfaz NiceGUI.

Para arrancar:
    python ui_app.py

Se abre automáticamente en el navegador en http://localhost:8080.
También se puede acceder desde otro dispositivo en la misma red
con la IP local del equipo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

# Reutilizamos toda la lógica del script principal sin modificarlo.
from roboflow_emociones import (
    DEFAULT_API_URL,
    DEFAULT_MODEL_ID,
    SESSIONS_DIR,
    Detection,
    DetectionSmoother,
    FaceTracker,
    InferenceWorker,
    RoboflowClient,
    SessionLogger,
    color_for,
    draw_detections,
    open_camera,
)

from nicegui import app, ui
from fastapi import Response
from fastapi.responses import StreamingResponse

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass


# ====================================================================
#  Configuración global
# ====================================================================

logger = logging.getLogger("ui_app")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ui_config.json"
ASSETS_DIR = ROOT / "assets"

UDEC_GREEN = "#1A8A3E"
UDEC_GREEN_DARK = "#0F5A28"

EMOJIS: dict[str, str] = {
    "happy": "😊", "happiness": "😊",
    "sad": "😢",  "sadness": "😢",
    "angry": "😠", "anger": "😠",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲", "surprised": "😲",
    "neutral": "😐",
    "contempt": "😒",
}


def emoji_for(cls: str) -> str:
    return EMOJIS.get(cls.strip().lower(), "🙂")


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# ====================================================================
#  Persistencia de configuración
# ====================================================================

DEFAULT_CONFIG: dict = {
    "api_key": "",
    "model_id": DEFAULT_MODEL_ID,
    "api_url": DEFAULT_API_URL,
    "camera_index": 0,
    "width": 640,
    "height": 480,
    "conf_threshold": 0.35,
    "mirror": True,
    "smoothing_enabled": True,
    "smooth_window": 7,
    "smooth_iou": 0.4,
    "tracking_enabled": True,
    "track_iou": 0.3,
    "track_max_missed": 15,
    "pdf_title": "Sistema de Análisis de Emociones Faciales en Tiempo Real",
    "pdf_authors": ("Cristian Camilo Posada García;"
                    "Luis Felipe Espinel Botina;"
                    "Michael Steven Naranjo Bautista"),
    "pdf_institution": "Universidad de Cundinamarca",
    "pdf_faculty": "Facultad de Ingeniería",
    "pdf_year": "",
    "dark_mode": True,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
        except Exception as exc:
            logger.warning("No se pudo leer ui_config.json: %s", exc)
    # La env var sobrescribe la key guardada si está disponible.
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key and not cfg["api_key"]:
        cfg["api_key"] = env_key
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                           encoding="utf-8")


# ====================================================================
#  Estado de la sesión activa
# ====================================================================

EMPTY_JPEG_PLACEHOLDER: bytes = b""


def _make_placeholder() -> bytes:
    # 4:3 (640x480) para coincidir con la mayoría de webcams por defecto
    # y evitar un "salto" visual cuando se inicia la captura real.
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (15, 15, 15)
    cv2.putText(img, "Sin video", (220, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (180, 180, 180), 2,
                cv2.LINE_AA)
    cv2.putText(img, "Pulsa 'Iniciar sesion' para comenzar",
                (110, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (140, 140, 140), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


EMPTY_JPEG_PLACEHOLDER = _make_placeholder()


class AppState:
    """Estado mutable de la aplicación. Se accede desde threads y desde UI."""

    def __init__(self) -> None:
        self.config: dict = load_config()

        # Sesión activa
        self.client: Optional[RoboflowClient] = None
        self.worker: Optional[InferenceWorker] = None
        self.smoother: Optional[DetectionSmoother] = None
        self.tracker: Optional[FaceTracker] = None
        self.session_logger: Optional[SessionLogger] = None

        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.paused = False

        # Métricas
        self.last_jpeg: bytes = EMPTY_JPEG_PLACEHOLDER
        self.last_jpeg_lock = threading.Lock()
        self.last_detections: list[Detection] = []
        self.last_seen_result_ts = 0.0
        self.api_count = 0
        self.started_at: Optional[float] = None
        self.last_latency_ms: Optional[float] = None
        self.dominant_class: Optional[str] = None
        self.dominant_conf: float = 0.0
        self.active_tracks: int = 0
        self.snapshots_taken: int = 0
        self.error_message: Optional[str] = None

    def reset_metrics(self) -> None:
        self.last_detections = []
        self.last_seen_result_ts = 0.0
        self.api_count = 0
        self.started_at = time.time()
        self.last_latency_ms = None
        self.dominant_class = None
        self.dominant_conf = 0.0
        self.active_tracks = 0
        self.snapshots_taken = 0
        self.error_message = None
        with self.last_jpeg_lock:
            self.last_jpeg = EMPTY_JPEG_PLACEHOLDER


state = AppState()


# ====================================================================
#  Bucle de captura
# ====================================================================

def capture_loop() -> None:
    """Thread principal de captura + render. Corre mientras state.running."""
    cfg = state.config
    assert state.cap is not None and state.worker is not None

    # Diagnóstico inicial de la cámara
    actual_w = int(state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = state.cap.get(cv2.CAP_PROP_FPS) or 0.0
    logger.info("Capture loop iniciado — cámara abierta a %dx%d, FPS reportado %.0f",
                actual_w, actual_h, actual_fps)

    frames_read = 0
    frames_failed = 0
    last_log = time.time()

    while state.running:
        ok, frame = state.cap.read()
        if not ok or frame is None:
            frames_failed += 1
            time.sleep(0.03)
            continue
        frames_read += 1

        # Log periódico de actividad (cada 2 segundos)
        now = time.time()
        if now - last_log > 2.0:
            logger.info("Capture: %d frames OK, %d fallidos en últimos %.1fs",
                        frames_read, frames_failed, now - last_log)
            frames_read = frames_failed = 0
            last_log = now
        if cfg.get("mirror", True):
            frame = cv2.flip(frame, 1)

        # Detección de auth_failed
        if state.worker.auth_failed.is_set():
            state.error_message = state.worker.auth_message or "Fallo de autenticación"
            state.running = False
            break

        # Submit al worker (no bloquea, política latest-frame)
        if not state.paused:
            state.worker.submit(frame.copy())

        result = state.worker.latest()
        if result is not None and result.received_at > state.last_seen_result_ts:
            state.last_seen_result_ts = result.received_at
            state.api_count += 1
            state.last_latency_ms = result.latency_s * 1000

            processed = result.detections
            if cfg.get("smoothing_enabled", True) and state.smoother:
                processed = state.smoother.update(processed)
            if cfg.get("tracking_enabled", True) and state.tracker:
                processed = state.tracker.update(processed)
            state.last_detections = processed
            state.active_tracks = state.tracker.active_count() if state.tracker else 0

            # Logging (frame limpio antes de dibujar)
            if state.session_logger:
                state.session_logger.log(
                    [d for d in processed
                     if d.confidence >= cfg.get("conf_threshold", 0.35)],
                    frame=frame,
                )
                state.snapshots_taken = len(state.session_logger._best_per_class)

            # Calcular dominante
            visible = [d for d in processed
                       if d.confidence >= cfg.get("conf_threshold", 0.35)]
            if visible:
                # Tomamos la detección con mayor confianza como "dominante"
                top = max(visible, key=lambda d: d.confidence)
                state.dominant_class = top.cls
                state.dominant_conf = top.confidence
            else:
                state.dominant_class = None
                state.dominant_conf = 0.0

        # Render del frame con anotaciones
        display = frame.copy()
        if state.last_detections:
            draw_detections(display, state.last_detections,
                            cfg.get("conf_threshold", 0.35))

        ok2, buf = cv2.imencode(".jpg", display,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok2:
            with state.last_jpeg_lock:
                state.last_jpeg = buf.tobytes()

    logger.info("Capture loop detenido")


# ====================================================================
#  Acciones del ciclo de vida de la sesión
# ====================================================================

def start_session() -> tuple[bool, str]:
    """Arranca cámara + worker + logger + capture thread. Retorna (ok, msg)."""
    if state.running:
        return False, "Ya hay una sesión activa."
    cfg = state.config
    if not cfg.get("api_key"):
        return False, ("Falta la API Key de Roboflow. Ve a Configuración y "
                       "pégala antes de iniciar.")

    try:
        state.client = RoboflowClient(
            api_key=cfg["api_key"],
            model_id=cfg["model_id"],
            api_url=cfg["api_url"],
        )
        state.cap = open_camera(
            index=cfg["camera_index"],
            width=cfg["width"],
            height=cfg["height"],
        )
    except SystemExit as exc:
        return False, f"No se pudo abrir la cámara: {exc}"
    except Exception as exc:
        return False, f"Error al iniciar: {exc}"

    state.smoother = DetectionSmoother(
        window_size=cfg["smooth_window"],
        iou_threshold=cfg["smooth_iou"],
    )
    state.tracker = FaceTracker(
        iou_threshold=cfg["track_iou"],
        max_missed=cfg["track_max_missed"],
    )
    state.session_logger = SessionLogger(log_dir=SESSIONS_DIR)
    state.worker = InferenceWorker(state.client)
    state.worker.start()

    state.reset_metrics()
    state.running = True
    state.paused = False

    state.capture_thread = threading.Thread(
        target=capture_loop, daemon=True, name="CaptureLoop")
    state.capture_thread.start()

    return True, f"Sesión iniciada: {state.session_logger.session_id}"


def pause_session(value: bool) -> None:
    state.paused = value


def stop_session(generate_report: bool = True) -> dict:
    """Detiene la sesión y genera los informes. Retorna dict con rutas."""
    if not state.running and state.session_logger is None:
        return {}

    state.running = False
    state.paused = False

    if state.capture_thread:
        state.capture_thread.join(timeout=2.0)
    if state.worker:
        state.worker.stop()
        state.worker.join(timeout=2.0)
    if state.cap:
        try:
            state.cap.release()
        except Exception:
            pass
    if state.client:
        state.client.close()

    out: dict = {}
    sl = state.session_logger
    if sl is not None:
        summary = sl.summary()
        out["summary"] = summary

        try:
            chart_path = sl.close(generate_chart=generate_report)
            out["chart"] = chart_path
        except Exception as exc:
            logger.warning("Error al cerrar el logger: %s", exc)
            chart_path = None

        if generate_report:
            try:
                md_path = sl.generate_report(
                    chart_path=chart_path, summary=summary,
                    model_id=state.config["model_id"],
                )
                out["md"] = md_path
            except Exception as exc:
                logger.warning("Error al generar MD: %s", exc)
            try:
                cfg = state.config
                authors = [a.strip() for a in cfg["pdf_authors"].split(";")
                           if a.strip()]
                pdf_path = sl.generate_pdf_report(
                    chart_path=chart_path, summary=summary,
                    model_id=cfg["model_id"],
                    title=cfg["pdf_title"],
                    authors=authors,
                    institution=cfg["pdf_institution"],
                    faculty=cfg["pdf_faculty"],
                    year=cfg["pdf_year"],
                )
                out["pdf"] = pdf_path
            except Exception as exc:
                logger.warning("Error al generar PDF: %s", exc)

        out["csv"] = sl.csv_path
        out["snapshots_dir"] = sl.snapshots_dir

    # Limpieza
    state.session_logger = None
    state.worker = None
    state.cap = None
    state.client = None
    state.smoother = None
    state.tracker = None
    state.capture_thread = None
    state.last_detections = []
    state.active_tracks = 0
    state.dominant_class = None
    with state.last_jpeg_lock:
        state.last_jpeg = EMPTY_JPEG_PLACEHOLDER

    return out


def take_snapshot() -> Optional[Path]:
    """Guarda el frame actual con anotaciones en snapshots/."""
    if not state.running:
        return None
    with state.last_jpeg_lock:
        data = state.last_jpeg
    if not data:
        return None
    snap_dir = ROOT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    path.write_bytes(data)
    return path


# ====================================================================
#  Lectura de sesiones pasadas
# ====================================================================

def list_past_sessions() -> list[dict]:
    """Escanea SESSIONS_DIR y construye una lista de sesiones disponibles."""
    items: list[dict] = []
    if not SESSIONS_DIR.exists():
        return items
    for csv_file in sorted(SESSIONS_DIR.glob("session_*.csv"), reverse=True):
        sid = csv_file.stem  # ej. session_20260508_193645
        base = csv_file.parent / sid
        entry = {
            "id": sid,
            "csv": csv_file,
            "chart": base.with_suffix(".png"),
            "md": Path(str(base) + "_informe.md"),
            "pdf": Path(str(base) + "_informe.pdf"),
            "snapshots_dir": Path(str(base) + "_snapshots"),
            "mtime": csv_file.stat().st_mtime,
            "size_kb": csv_file.stat().st_size / 1024,
        }
        # Parsear fecha del nombre
        try:
            ts = sid.replace("session_", "")
            entry["display_date"] = (
                f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
            )
        except Exception:
            entry["display_date"] = sid
        items.append(entry)
    return items


def open_in_explorer(path: Path) -> None:
    """Abre un archivo o carpeta con la app por defecto del sistema."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        logger.warning("No se pudo abrir %s: %s", path, exc)


# ====================================================================
#  Endpoints HTTP
# ====================================================================

@app.get("/video_frame")
async def video_frame_endpoint() -> Response:
    """Devuelve el último JPEG del frame (fallback sin streaming)."""
    with state.last_jpeg_lock:
        data = state.last_jpeg or EMPTY_JPEG_PLACEHOLDER
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/video_stream")
async def video_stream_endpoint() -> StreamingResponse:
    """
    Stream MJPEG (multipart/x-mixed-replace). El navegador mantiene la
    conexión abierta y renderiza cada frame conforme llega — sin flicker
    ni recargas. Es el patrón estándar para video web.
    """
    boundary = b"--frame"

    async def gen():
        try:
            while True:
                with state.last_jpeg_lock:
                    data = state.last_jpeg or EMPTY_JPEG_PLACEHOLDER
                if data:
                    yield (boundary + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(data)).encode()
                           + b"\r\n\r\n" + data + b"\r\n")
                # ~30 fps de envío. La captura suele ir más lenta, así
                # que repetir el último frame mantiene la vista fluida.
                await asyncio.sleep(0.033)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


# Servir carpeta sessions/ como estática para mostrar imágenes y descargar PDFs
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
app.add_static_files("/sessions_static", str(SESSIONS_DIR))


# ====================================================================
#  UI — utilidades
# ====================================================================

def fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "0:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def compute_distribution() -> list[tuple[str, float]]:
    """Devuelve [(clase, porcentaje)] del logger activo, top primeras."""
    sl = state.session_logger
    if sl is None:
        return []
    from collections import Counter
    counts: Counter = Counter()
    for events in sl._timeline.values():
        for _, cls, _ in events:
            counts[cls] += 1
    total = sum(counts.values()) or 1
    return [(cls, n / total * 100) for cls, n in counts.most_common()]


# ====================================================================
#  UI — Tab Live
# ====================================================================

def build_live_tab() -> None:
    """Vista principal: video + métricas + controles."""
    with ui.row().classes("w-full no-wrap items-stretch gap-4"):

        # ---- Columna izquierda: video y controles ----
        with ui.column().classes("flex-grow items-stretch gap-3"):
            # Card del video — usa MJPEG streaming para evitar flicker.
            # El wrapper tiene aspect-ratio fijo 4:3 (coincide con la
            # webcam por defecto a 640x480). La imagen lo llena al
            # 100% en ambas dimensiones y `object-fit: contain` se
            # encarga de centrarla si la fuente real tiene otro ratio
            # (ej. 16:9 dejaría una banda fina arriba/abajo, sin
            # romper la composición ni dejar bandas desiguales).
            with ui.card().classes("p-0 overflow-hidden") \
                    .style(
                        "border-radius: 12px; background: #0a0a0a; "
                        "width: 100%; max-width: 1100px; "
                        "margin-left: auto; margin-right: auto;"
                    ):
                ui.html(
                    '<div style="position: relative; width: 100%; '
                    'aspect-ratio: 4/3; max-height: 78vh; '
                    'background: #0a0a0a; overflow: hidden;">'
                    '<img id="cam-stream" src="/video_stream" '
                    'style="position: absolute; inset: 0; '
                    'width: 100%; height: 100%; '
                    'display: block; object-fit: contain;" alt="video">'
                    '</div>'
                )

            # Controles
            with ui.row().classes("w-full justify-center gap-2 q-mt-sm"):
                btn_start = ui.button("Iniciar sesión", icon="play_arrow") \
                    .props(f"color=primary unelevated")
                btn_pause = ui.button("Pausar", icon="pause") \
                    .props("flat color=primary").bind_visibility_from(
                        state, "running")
                btn_stop = ui.button("Detener y generar PDF", icon="stop") \
                    .props("color=negative unelevated").bind_visibility_from(
                        state, "running")
                btn_snap = ui.button("Snapshot", icon="camera_alt") \
                    .props("flat color=primary").bind_visibility_from(
                        state, "running")

            footer = ui.label().classes("text-caption text-grey-7")

        # ---- Columna derecha: métricas ----
        with ui.column().classes("w-96 gap-3"):

            # Card emoción dominante
            with ui.card().classes("w-full") \
                    .style(f"border-left: 4px solid {UDEC_GREEN};"):
                ui.label("EMOCIÓN DOMINANTE").classes(
                    "text-caption text-grey")
                with ui.row().classes("items-baseline gap-3"):
                    emoji_label = ui.label("—").classes("text-h2")
                    with ui.column().classes("gap-0"):
                        dom_name = ui.label("—").classes(
                            "text-h6 text-weight-medium")
                        dom_conf = ui.label("Sin datos").classes(
                            "text-caption text-grey")

            # Card sesión
            with ui.card().classes("w-full"):
                ui.label("SESIÓN").classes("text-caption text-grey")
                with ui.row().classes("items-center gap-4 q-mt-xs"):
                    with ui.column().classes("gap-0"):
                        ui.label("Tiempo").classes("text-caption text-grey")
                        duration_label = ui.label("0:00").classes(
                            "text-h6 text-weight-medium")
                    with ui.column().classes("gap-0"):
                        ui.label("Personas").classes(
                            "text-caption text-grey")
                        persons_label = ui.label("0").classes(
                            "text-h6 text-weight-medium")
                    with ui.column().classes("gap-0"):
                        ui.label("Inferencias").classes(
                            "text-caption text-grey")
                        api_label = ui.label("0").classes(
                            "text-h6 text-weight-medium")

            # Card distribución
            with ui.card().classes("w-full"):
                ui.label("DISTRIBUCIÓN EN VIVO").classes(
                    "text-caption text-grey")
                dist_container = ui.column().classes("w-full gap-1 q-mt-xs")
                with dist_container:
                    ui.label("Sin datos todavía.").classes(
                        "text-caption text-grey")

            # Card estado API
            with ui.card().classes("w-full"):
                ui.label("ESTADO").classes("text-caption text-grey")
                status_label = ui.label("Detenida").classes(
                    "text-body1 text-weight-medium")
                latency_label = ui.label("").classes(
                    "text-caption text-grey")

    # ---- Comportamientos ----

    async def do_start() -> None:
        btn_start.props("loading")
        ok, msg = await asyncio.to_thread(start_session)
        btn_start.props(remove="loading")
        if ok:
            ui.notify(msg, type="positive")
        else:
            ui.notify(msg, type="negative", multi_line=True)

    async def do_stop() -> None:
        btn_stop.props("loading")
        result = await asyncio.to_thread(stop_session, True)
        btn_stop.props(remove="loading")
        if not result:
            return
        # Notificación con acción para abrir el PDF
        summary = result.get("summary", {})
        pdf = result.get("pdf")
        msg = (f"Sesión cerrada. {summary.get('total_detections', 0)} "
               f"detecciones registradas.")
        ui.notify(msg, type="positive", timeout=4000)
        if pdf is not None and Path(pdf).exists():
            ui.notify("PDF generado. Puedes verlo en la pestaña Sesiones.",
                      type="info", timeout=4000)

    def do_pause() -> None:
        new_val = not state.paused
        pause_session(new_val)
        btn_pause.text = "Reanudar" if new_val else "Pausar"
        btn_pause.icon = "play_arrow" if new_val else "pause"
        ui.notify("Pausada" if new_val else "Reanudada", type="info",
                  timeout=1000)

    def do_snap() -> None:
        path = take_snapshot()
        if path:
            ui.notify(f"Snapshot guardado: {path.name}", type="positive",
                      timeout=2000)
        else:
            ui.notify("No se pudo capturar.", type="warning")

    btn_start.on("click", do_start)
    btn_stop.on("click", do_stop)
    btn_pause.on("click", do_pause)
    btn_snap.on("click", do_snap)

    # ---- Actualizadores en tiempo real ----

    last_ts_cache = [0.0]

    def refresh_metrics() -> None:
        # Botón principal
        btn_start.visible = not state.running

        # Estado y latencia
        if state.error_message:
            status_label.text = "⚠ " + state.error_message
            status_label.classes(replace="text-body1 text-weight-medium "
                                         "text-negative")
            latency_label.text = ""
        elif state.running:
            mode = "En pausa" if state.paused else "En vivo"
            status_label.text = "● " + mode
            status_label.classes(replace="text-body1 text-weight-medium "
                                         "text-positive")
            if state.last_latency_ms:
                latency_label.text = (f"latencia API: "
                                      f"{state.last_latency_ms:.0f} ms")
            else:
                latency_label.text = "esperando primera inferencia..."
        else:
            status_label.text = "Detenida"
            status_label.classes(replace="text-body1 text-weight-medium "
                                         "text-grey")
            latency_label.text = ""

        # Duración
        if state.started_at and state.running:
            duration_label.text = fmt_duration(time.time() - state.started_at)
        elif state.session_logger is None:
            duration_label.text = "0:00"

        # Personas / API
        persons_label.text = str(state.active_tracks)
        api_label.text = str(state.api_count)

        # Emoción dominante
        if state.dominant_class:
            emoji_label.text = emoji_for(state.dominant_class)
            dom_name.text = state.dominant_class.capitalize()
            dom_conf.text = f"Confianza {state.dominant_conf * 100:.0f}%"
        elif state.running:
            emoji_label.text = "👀"
            dom_name.text = "Esperando rostro..."
            dom_conf.text = ""
        else:
            emoji_label.text = "—"
            dom_name.text = "—"
            dom_conf.text = "Sin datos"

        # Footer técnico
        if state.running:
            footer.text = (f"FPS de inferencia ≈ "
                           f"{state.api_count / max(1, time.time() - (state.started_at or time.time())):.1f}/s · "
                           f"snapshots: {state.snapshots_taken}")
        else:
            footer.text = "Lista para iniciar."

        # Distribución (cada ~1s para no parpadear)
        if state.running and state.session_logger:
            now = time.time()
            if now - last_ts_cache[0] > 1.0:
                last_ts_cache[0] = now
                _render_distribution(dist_container)

    def _render_distribution(container: ui.column) -> None:
        dist = compute_distribution()
        container.clear()
        if not dist:
            with container:
                ui.label("Sin detecciones todavía.").classes(
                    "text-caption text-grey")
            return
        with container:
            for cls, pct in dist[:5]:
                color = bgr_to_hex(color_for(cls))
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(emoji_for(cls)).classes("text-h6").style(
                        "min-width: 28px;")
                    ui.label(cls).classes("text-body2").style(
                        "min-width: 90px;")
                    bar = ui.linear_progress(value=pct / 100, show_value=False) \
                        .props(f"color={color[1:]}").classes("flex-grow")
                    bar.style(f"background-color: rgba(0,0,0,0.05);")
                    ui.label(f"{pct:.1f}%").classes(
                        "text-caption").style("min-width: 50px; "
                                              "text-align: right;")

    # Timer único: actualiza métricas. El video va por MJPEG streaming
    # (no requiere timer, el navegador lo refresca solo).
    ui.timer(0.3, refresh_metrics)


# ====================================================================
#  UI — Tab Sesiones
# ====================================================================

def build_sessions_tab() -> None:
    """Lista de sesiones pasadas con acciones."""
    header = ui.row().classes("w-full items-center q-mb-md")
    with header:
        ui.label("Historial de sesiones").classes("text-h6")
        ui.space()
        refresh_btn = ui.button(icon="refresh").props("flat round")

    sessions_container = ui.column().classes("w-full gap-2")

    def render():
        sessions_container.clear()
        items = list_past_sessions()
        if not items:
            with sessions_container:
                with ui.card().classes("w-full bg-grey-2"):
                    ui.label("Aún no hay sesiones registradas.").classes(
                        "text-grey-7")
                    ui.label("Inicia una sesión desde la pestaña Cámara para "
                             "generar tu primer informe.").classes(
                        "text-caption text-grey")
            return
        for it in items:
            with sessions_container:
                with ui.card().classes("w-full") \
                        .style(f"border-left: 4px solid {UDEC_GREEN};"):
                    with ui.row().classes("w-full items-center no-wrap"):
                        with ui.column().classes("gap-0 flex-grow"):
                            ui.label(it["display_date"]).classes(
                                "text-body1 text-weight-medium")
                            ui.label(f"ID: {it['id']}").classes(
                                "text-caption text-grey")
                        with ui.row().classes("gap-1"):
                            if it["pdf"].exists():
                                ui.button("PDF", icon="picture_as_pdf") \
                                    .props("flat color=primary").on(
                                        "click",
                                        lambda p=it["pdf"]: open_in_explorer(p))
                            if it["md"].exists():
                                ui.button("MD", icon="description") \
                                    .props("flat color=primary").on(
                                        "click",
                                        lambda p=it["md"]: open_in_explorer(p))
                            if it["chart"].exists():
                                ui.button("Gráfico", icon="bar_chart") \
                                    .props("flat color=primary").on(
                                        "click",
                                        lambda p=it["chart"]: open_in_explorer(p))
                            ui.button("Carpeta", icon="folder_open") \
                                .props("flat color=primary").on(
                                    "click",
                                    lambda p=it["csv"].parent:
                                        open_in_explorer(p))
                            ui.button("Análisis", icon="analytics") \
                                .props(f"color=primary unelevated").on(
                                    "click",
                                    lambda sid=it["id"]:
                                        show_analysis_for(sid))

    refresh_btn.on("click", render)
    render()


# ====================================================================
#  UI — Tab Análisis
# ====================================================================

_analysis_state = {"current_id": None}


def show_analysis_for(session_id: str) -> None:
    """Navega a la pestaña Análisis y muestra esa sesión."""
    _analysis_state["current_id"] = session_id
    tabs.set_value("analysis")  # type: ignore
    analysis_container.refresh()


@ui.refreshable
def analysis_container() -> None:
    """Vista detallada de una sesión específica."""
    sid = _analysis_state.get("current_id")
    if not sid:
        with ui.card().classes("w-full bg-grey-2"):
            ui.label("Selecciona una sesión en la pestaña 'Sesiones' "
                     "para ver su análisis aquí.").classes("text-grey-7")
        return

    items = list_past_sessions()
    item = next((i for i in items if i["id"] == sid), None)
    if item is None:
        ui.label(f"No se encontró la sesión {sid}").classes("text-negative")
        return

    # Cabecera
    with ui.row().classes("w-full items-center q-mb-md"):
        ui.icon("analytics", size="lg").style(f"color: {UDEC_GREEN};")
        with ui.column().classes("gap-0"):
            ui.label(f"Análisis · {item['display_date']}").classes(
                "text-h6")
            ui.label(item["id"]).classes("text-caption text-grey")
        ui.space()
        if item["pdf"].exists():
            ui.button("Abrir PDF", icon="picture_as_pdf").props(
                "color=primary unelevated").on(
                "click", lambda p=item["pdf"]: open_in_explorer(p))
        ui.button("Carpeta", icon="folder_open").props(
            "flat color=primary").on(
            "click", lambda p=item["csv"].parent: open_in_explorer(p))

    # Resumen de la sesión (parseando el CSV)
    import csv
    detections = []
    try:
        with open(item["csv"], encoding="utf-8") as f:
            for row in csv.DictReader(f):
                detections.append(row)
    except Exception as exc:
        ui.label(f"No se pudo leer el CSV: {exc}").classes("text-negative")
        return

    from collections import Counter
    cls_counter = Counter(d["class"] for d in detections)
    total = sum(cls_counter.values()) or 1
    track_ids = {d["track_id"] for d in detections}
    duration = float(detections[-1]["elapsed_s"]) if detections else 0.0

    # Cards de KPIs
    with ui.row().classes("w-full gap-3 q-mb-md no-wrap"):
        _kpi_card("Duración", fmt_duration(duration), "schedule")
        _kpi_card("Detecciones", str(len(detections)), "fact_check")
        _kpi_card("Personas únicas", str(len(track_ids)), "groups")
        _kpi_card("Emociones distintas", str(len(cls_counter)), "psychology")

    # Distribución como barras
    with ui.card().classes("w-full q-mb-md"):
        ui.label("Distribución de emociones").classes(
            "text-subtitle1 text-weight-medium")
        for cls, n in cls_counter.most_common():
            pct = n / total * 100
            with ui.row().classes("items-center gap-2 w-full"):
                ui.label(emoji_for(cls)).classes("text-h6").style(
                    "min-width: 30px;")
                ui.label(cls).classes("text-body2").style(
                    "min-width: 100px;")
                ui.linear_progress(value=pct / 100, show_value=False).props(
                    f"color=green").classes("flex-grow")
                ui.label(f"{pct:.1f}% ({n})").classes(
                    "text-caption").style("min-width: 90px; "
                                          "text-align: right;")

    # Gráfico embebido
    if item["chart"].exists():
        rel = f"/sessions_static/{item['chart'].name}"
        with ui.card().classes("w-full q-mb-md"):
            ui.label("Línea de tiempo emocional").classes(
                "text-subtitle1 text-weight-medium q-mb-sm")
            ui.image(rel).classes("w-full")

    # Snapshots
    if item["snapshots_dir"].exists():
        snaps = sorted(item["snapshots_dir"].glob("*.jpg"))
        if snaps:
            with ui.card().classes("w-full"):
                ui.label("Snapshots representativos").classes(
                    "text-subtitle1 text-weight-medium q-mb-sm")
                with ui.row().classes("w-full gap-3 wrap"):
                    for snap in snaps:
                        rel = (f"/sessions_static/"
                               f"{item['snapshots_dir'].name}/{snap.name}")
                        with ui.column().classes("items-center gap-1"):
                            ui.image(rel).style(
                                "width: 160px; height: 160px; "
                                "object-fit: cover; border-radius: 8px;")
                            ui.label(snap.stem).classes(
                                "text-caption text-weight-medium")


def _kpi_card(label: str, value: str, icon: str) -> None:
    with ui.card().classes("flex-grow") \
            .style(f"border-top: 3px solid {UDEC_GREEN};"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon, size="md").style(f"color: {UDEC_GREEN};")
            with ui.column().classes("gap-0"):
                ui.label(label).classes("text-caption text-grey")
                ui.label(value).classes("text-h6 text-weight-medium")


# ====================================================================
#  UI — Tab Configuración
# ====================================================================

def build_settings_tab() -> None:
    cfg = state.config

    with ui.row().classes("w-full items-center q-mb-md"):
        ui.label("Configuración").classes("text-h6")

    # Acordeón con secciones
    with ui.expansion("Conexión a Roboflow", icon="cloud", value=True) \
            .classes("w-full"):
        api_input = ui.input("API Key",
                             password=True, password_toggle_button=True,
                             value=cfg["api_key"]).classes("w-full")
        model_input = ui.input("Modelo",
                               value=cfg["model_id"]).classes("w-full")
        url_input = ui.input("URL del endpoint",
                             value=cfg["api_url"]).classes("w-full")
        ui.label("Tu Private API Key está en "
                 "https://app.roboflow.com/settings/api").classes(
            "text-caption text-grey")

    with ui.expansion("Cámara y video", icon="videocam").classes("w-full"):
        cam_input = ui.number("Índice de cámara",
                              value=cfg["camera_index"], min=0, max=9,
                              format="%.0f").classes("w-full")
        w_input = ui.number("Ancho (px)", value=cfg["width"], min=160,
                            max=1920, format="%.0f").classes("w-full")
        h_input = ui.number("Alto (px)", value=cfg["height"], min=120,
                            max=1080, format="%.0f").classes("w-full")
        mirror_input = ui.switch("Espejo (selfie)", value=cfg["mirror"])

    with ui.expansion("Detección y procesamiento", icon="tune").classes(
            "w-full"):
        conf_slider = ui.slider(min=0.05, max=0.95, step=0.05,
                                value=cfg["conf_threshold"]).props(
            "label-always color=green")
        ui.label("Umbral de confianza").classes("text-caption text-grey")
        smooth_switch = ui.switch("Suavizado temporal",
                                  value=cfg["smoothing_enabled"])
        sw_input = ui.number("Ventana del smoothing (frames)",
                             value=cfg["smooth_window"], min=1, max=30,
                             format="%.0f").classes("w-full")
        track_switch = ui.switch("Tracking con IDs persistentes",
                                 value=cfg["tracking_enabled"])
        ti_input = ui.number("IoU mínimo para mantener track",
                             value=cfg["track_iou"], min=0.1, max=0.9,
                             step=0.05, format="%.2f").classes("w-full")
        tmm_input = ui.number("Frames sin detección antes de descartar",
                              value=cfg["track_max_missed"], min=1, max=120,
                              format="%.0f").classes("w-full")

    with ui.expansion("Metadatos del informe PDF", icon="picture_as_pdf") \
            .classes("w-full"):
        title_input = ui.input("Título",
                               value=cfg["pdf_title"]).classes("w-full")
        authors_input = ui.input("Autores (separados por ';')",
                                 value=cfg["pdf_authors"]).classes("w-full")
        inst_input = ui.input("Institución",
                              value=cfg["pdf_institution"]).classes("w-full")
        fac_input = ui.input("Facultad",
                             value=cfg["pdf_faculty"]).classes("w-full")
        year_input = ui.input("Año (vacío = año actual)",
                              value=cfg["pdf_year"]).classes("w-full")

    ui.separator().classes("q-my-md")

    def do_save() -> None:
        state.config.update({
            "api_key": api_input.value or "",
            "model_id": model_input.value or DEFAULT_MODEL_ID,
            "api_url": url_input.value or DEFAULT_API_URL,
            "camera_index": int(cam_input.value or 0),
            "width": int(w_input.value or 640),
            "height": int(h_input.value or 480),
            "mirror": bool(mirror_input.value),
            "conf_threshold": float(conf_slider.value),
            "smoothing_enabled": bool(smooth_switch.value),
            "smooth_window": int(sw_input.value or 7),
            "tracking_enabled": bool(track_switch.value),
            "track_iou": float(ti_input.value or 0.3),
            "track_max_missed": int(tmm_input.value or 15),
            "pdf_title": title_input.value or "",
            "pdf_authors": authors_input.value or "",
            "pdf_institution": inst_input.value or "",
            "pdf_faculty": fac_input.value or "",
            "pdf_year": year_input.value or "",
        })
        save_config(state.config)
        ui.notify("Configuración guardada", type="positive")

    with ui.row().classes("w-full justify-end gap-2"):
        ui.button("Restaurar valores por defecto", icon="restart_alt") \
            .props("flat").on("click", lambda: _reset_defaults())
        ui.button("Guardar configuración", icon="save") \
            .props("color=primary unelevated").on("click", do_save)


def _reset_defaults() -> None:
    state.config.update({k: v for k, v in DEFAULT_CONFIG.items()})
    save_config(state.config)
    ui.notify("Valores por defecto restaurados. Refrescá la página.",
              type="info")


# ====================================================================
#  Página principal
# ====================================================================

tabs: ui.tabs  # se define en build_main()


def build_main() -> None:
    global tabs

    ui.colors(primary=UDEC_GREEN, secondary=UDEC_GREEN_DARK)

    # Header
    with ui.header(elevated=False).style(
            f"background-color: white; border-bottom: 3px solid {UDEC_GREEN};"
            "color: #1F2937;"):
        with ui.row().classes("items-center w-full no-wrap"):
            ui.icon("psychology", size="lg").style(f"color: {UDEC_GREEN};")
            ui.label("Análisis de Emociones Faciales").classes(
                "text-h6 text-weight-medium").style("color: #1F2937;")
            ui.label("· Universidad de Cundinamarca").classes(
                "text-caption text-grey")
            ui.space()
            status_pill = ui.label("● Inactiva").classes(
                "q-px-md q-py-xs").style(
                "background: #f3f4f6; border-radius: 999px; "
                "font-size: 12px; color: #6b7280;")
            dark_btn = ui.button(icon="dark_mode").props(
                "flat round").tooltip("Tema claro/oscuro")

    # Estado del header (refrescado periódicamente)
    def refresh_status_pill() -> None:
        if state.error_message:
            status_pill.text = "⚠ Error"
            status_pill.style(replace="background: #fee2e2; "
                                      "border-radius: 999px; font-size: 12px; "
                                      "color: #b91c1c; padding: 4px 12px;")
        elif state.running and state.paused:
            status_pill.text = "⏸ En pausa"
            status_pill.style(replace="background: #fef3c7; "
                                      "border-radius: 999px; font-size: 12px; "
                                      "color: #92400e; padding: 4px 12px;")
        elif state.running:
            status_pill.text = "● En vivo"
            status_pill.style(replace="background: #dcfce7; "
                                      "border-radius: 999px; font-size: 12px; "
                                      "color: #166534; padding: 4px 12px;")
        else:
            status_pill.text = "● Inactiva"
            status_pill.style(replace="background: #f3f4f6; "
                                      "border-radius: 999px; font-size: 12px; "
                                      "color: #6b7280; padding: 4px 12px;")

    ui.timer(0.5, refresh_status_pill)

    dark = ui.dark_mode(value=state.config.get("dark_mode", False))

    def toggle_dark() -> None:
        dark.value = not dark.value
        state.config["dark_mode"] = dark.value
        save_config(state.config)
    dark_btn.on("click", toggle_dark)

    # Tabs
    with ui.tabs().classes("w-full").props(
            f"active-color=primary indicator-color=primary") as t:
        ui.tab("live", label="Cámara", icon="videocam")
        ui.tab("sessions", label="Sesiones", icon="folder_special")
        ui.tab("analysis", label="Análisis", icon="analytics")
        ui.tab("settings", label="Configuración", icon="settings")
    tabs = t

    with ui.tab_panels(tabs, value="live").classes("w-full"):
        with ui.tab_panel("live"):
            build_live_tab()
        with ui.tab_panel("sessions"):
            build_sessions_tab()
        with ui.tab_panel("analysis"):
            analysis_container()
        with ui.tab_panel("settings"):
            build_settings_tab()


# ====================================================================
#  Entrypoint
# ====================================================================

@ui.page("/")
def index_page() -> None:
    build_main()


def shutdown_hook() -> None:
    """Limpieza al cerrar la app."""
    logger.info("Cerrando aplicación...")
    if state.running:
        try:
            stop_session(generate_report=False)
        except Exception:
            pass


app.on_shutdown(shutdown_hook)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Devuelve True si ya hay un servidor escuchando en (host, port)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _parse_ui_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dashboard web para detección de emociones (NiceGUI).")
    p.add_argument("--port", type=int, default=8080,
                   help="Puerto donde escuchar (default: 8080).")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="Host donde escuchar. '0.0.0.0' acepta conexiones "
                        "desde la red local (default).")
    p.add_argument("--no-browser", action="store_true",
                   help="No abrir el navegador automáticamente al iniciar.")
    return p.parse_args()


if __name__ in {"__main__", "__mp_main__"}:
    _args = _parse_ui_args()

    # Diagnóstico previo al bind: si el puerto ya está ocupado, casi seguro
    # quedó una instancia anterior corriendo en segundo plano. En vez de
    # arrancar otra que falle silenciosamente, avisamos al usuario.
    if is_port_in_use(_args.port):
        print()
        print("=" * 64)
        print(f"  El puerto {_args.port} ya está en uso.")
        print("=" * 64)
        print()
        print("  Probablemente hay otra instancia del dashboard")
        print(f"  corriendo en http://localhost:{_args.port}")
        print()
        print("  Tres opciones para resolverlo:")
        print()
        print("    1) Cierra la instancia vieja:")
        print("       taskkill /IM python.exe /F")
        print()
        print(f"    2) Abre la instancia que ya está corriendo:")
        print(f"       http://localhost:{_args.port}")
        print()
        print("    3) Usa otro puerto:")
        print(f"       python ui_app.py --port {_args.port + 1}")
        print()
        print("=" * 64)
        sys.exit(1)

    ui.run(
        title="Análisis de Emociones · UdeC",
        favicon="🎭",
        host=_args.host,
        port=_args.port,
        reload=False,
        show=not _args.no_browser,
        dark=None,
    )
