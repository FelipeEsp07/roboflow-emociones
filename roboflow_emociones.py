"""
Detección de emociones en tiempo real con Roboflow (versión local, robusta).

Rediseño orientado a robustez y calidad:
- Sin `inference_sdk` ni `supervision`. Llama directamente al endpoint REST de
  Roboflow Serverless con `requests`, eliminando el conflicto Pillow/`_Ink` que
  rompía el notebook original.
- Inferencia en hilo de fondo: la previsualización de la cámara nunca se bloquea
  esperando a la API.
- Política "latest frame": cuando llega un frame nuevo y la inferencia anterior
  no terminó, se descarta el viejo. La API siempre trabaja sobre lo más reciente.
- Reintentos automáticos con backoff exponencial sobre errores HTTP transitorios
  (429 / 5xx).
- Soporte opcional de archivo .env (si python-dotenv está instalado).
- Paleta de colores por clase y dibujo de etiquetas con medida real del texto.
- Controles en caliente:
    q / ESC  salir
    s        guardar snapshot anotado
    espacio  pausar / reanudar inferencia
    + / -    ajustar umbral de confianza
    m        alternar espejo (selfie view)
- Logging estructurado (sin `print` disperso).

Dependencias mínimas:
    pip install opencv-python numpy requests reportlab matplotlib
    pip install python-dotenv      # opcional

Uso:
    python roboflow_emociones.py
    python roboflow_emociones.py --camera 0 --width 1280 --height 720
    python roboflow_emociones.py --image foto.jpg
"""

from __future__ import annotations

import argparse
import base64
import csv
import logging
import os
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("roboflow_emociones")

DEFAULT_MODEL_ID = "human-face-emotions/28"
DEFAULT_API_URL = "https://serverless.roboflow.com"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"

EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    # BGR para OpenCV
    "happy": (0, 220, 0),
    "happiness": (0, 220, 0),
    "sad": (220, 130, 0),
    "sadness": (220, 130, 0),
    "angry": (0, 0, 220),
    "anger": (0, 0, 220),
    "fear": (180, 0, 220),
    "disgust": (0, 180, 180),
    "surprise": (0, 220, 220),
    "surprised": (0, 220, 220),
    "neutral": (200, 200, 200),
    "contempt": (130, 130, 0),
}


def color_for(label: str) -> tuple[int, int, int]:
    key = label.strip().lower()
    if key in EMOTION_COLORS:
        return EMOTION_COLORS[key]
    h = abs(hash(key)) % 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


# --------------------------------- modelos ----------------------------------


@dataclass
class Detection:
    cls: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: Optional[int] = None  # asignado por FaceTracker, None si no hay tracking


@dataclass
class InferenceResult:
    detections: list[Detection]
    raw: dict
    latency_s: float
    received_at: float


def parse_predictions(result: dict, img_shape: tuple[int, int]) -> list[Detection]:
    """Convierte la respuesta cruda de Roboflow en `Detection` con bbox en píxeles."""
    if not isinstance(result, dict):
        return []

    raw_preds = result.get("predictions", [])
    if isinstance(raw_preds, dict):
        flat = []
        for value in raw_preds.values():
            if isinstance(value, list):
                flat.extend(value)
        raw_preds = flat
    if not isinstance(raw_preds, list):
        return []

    h_img, w_img = img_shape
    out: list[Detection] = []

    for pred in raw_preds:
        try:
            confidence = float(pred.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue

        cls = str(pred.get("class", pred.get("class_name", "emotion")))

        if all(k in pred for k in ("x", "y", "width", "height")):
            cx = float(pred["x"]); cy = float(pred["y"])
            w = float(pred["width"]); h = float(pred["height"])
            x1 = int(round(cx - w / 2))
            y1 = int(round(cy - h / 2))
            x2 = int(round(cx + w / 2))
            y2 = int(round(cy + h / 2))
        elif all(k in pred for k in ("x_min", "y_min", "x_max", "y_max")):
            x1 = int(pred["x_min"]); y1 = int(pred["y_min"])
            x2 = int(pred["x_max"]); y2 = int(pred["y_max"])
        else:
            continue

        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        out.append(Detection(cls=cls, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2))

    return out


# ------------------------------ smoothing -----------------------------------


def _bbox_iou(a: Detection, b: Detection) -> float:
    """Intersection-over-Union entre dos detecciones."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionSmoother:
    """
    Suavizado temporal sin tracking explícito.

    Mantiene una ventana deslizante con las detecciones de los últimos
    `window_size` frames. Para cada detección nueva busca, en cada frame
    pasado, la detección con mayor IoU; si supera `iou_threshold`, se
    considera "la misma cara" y aporta su voto a la clase final.

    La clase resultante es la de mayor suma de confianzas (voto ponderado),
    y la confianza mostrada es el promedio de las que apoyan esa clase.

    Esto reduce el parpadeo entre frames (happy → neutral → happy → ...)
    sin necesidad de IDs persistentes (eso vendrá con tracking real).
    """

    def __init__(self, window_size: int = 7, iou_threshold: float = 0.4) -> None:
        self.window_size = max(1, window_size)
        self.iou_threshold = iou_threshold
        self.history: deque[list[Detection]] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.history.clear()

    def update(self, detections: list[Detection]) -> list[Detection]:
        smoothed: list[Detection] = []

        for det in detections:
            # Votos: empezamos con la propia detección.
            supporters: list[Detection] = [det]

            # Buscar mejor match en cada frame pasado.
            for past_frame in self.history:
                best_iou = 0.0
                best_match: Optional[Detection] = None
                for past_det in past_frame:
                    iou = _bbox_iou(det, past_det)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = past_det
                if best_match is not None and best_iou >= self.iou_threshold:
                    supporters.append(best_match)

            # Voto ponderado por confianza.
            class_scores: dict[str, float] = {}
            for s in supporters:
                class_scores[s.cls] = class_scores.get(s.cls, 0.0) + s.confidence

            best_class = max(class_scores, key=class_scores.get)
            agreeing = [s for s in supporters if s.cls == best_class]
            mean_conf = sum(s.confidence for s in agreeing) / len(agreeing)

            smoothed.append(Detection(
                cls=best_class,
                confidence=mean_conf,
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                track_id=det.track_id,
            ))

        # Guardamos las detecciones CRUDAS (no las suavizadas) en el histórico,
        # para que el voto siempre se base en evidencia original del modelo.
        self.history.append(detections)
        return smoothed


# -------------------------------- tracking ----------------------------------


@dataclass
class _Track:
    """Estado interno de una cara seguida entre frames."""
    id: int
    bbox: Detection         # última detección asociada
    missed: int = 0         # frames consecutivos sin match
    seen: int = 1           # total de frames en los que apareció
    first_seen_at: float = 0.0


class FaceTracker:
    """
    Tracker simple por IoU (greedy assignment).

    Para cada nuevo conjunto de detecciones:
      1. Calcula IoU contra todos los tracks vivos.
      2. Empareja en orden descendente de IoU (greedy 1-a-1).
      3. Detecciones sin emparejar → nuevos tracks con IDs nuevos.
      4. Tracks sin emparejar → incrementan `missed`; se descartan si superan
         `max_missed` (la persona salió del frame).

    Cada `Detection` recibe su `track_id` antes de devolverse, lo que permite
    mostrar etiquetas tipo `#3 happy 87%` y, más adelante, exportar la timeline
    por persona.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def active_count(self) -> int:
        return len(self._tracks)

    def update(self, detections: list[Detection]) -> list[Detection]:
        now = time.time()

        # Caso trivial: no hay tracks → todo es nuevo.
        if not self._tracks:
            for det in detections:
                det.track_id = self._next_id
                self._tracks[self._next_id] = _Track(
                    id=self._next_id, bbox=det, first_seen_at=now,
                )
                self._next_id += 1
            return detections

        # Construir todas las parejas (det_idx, track_id, iou) por encima del umbral.
        candidates: list[tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            for tid, track in self._tracks.items():
                iou = _bbox_iou(det, track.bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, di, tid))
        candidates.sort(reverse=True)  # mayor IoU primero

        used_dets: set[int] = set()
        used_tracks: set[int] = set()
        for iou, di, tid in candidates:
            if di in used_dets or tid in used_tracks:
                continue
            used_dets.add(di)
            used_tracks.add(tid)
            detections[di].track_id = tid
            track = self._tracks[tid]
            track.bbox = detections[di]
            track.missed = 0
            track.seen += 1

        # Detecciones sin track → tracks nuevos.
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            det.track_id = self._next_id
            self._tracks[self._next_id] = _Track(
                id=self._next_id, bbox=det, first_seen_at=now,
            )
            self._next_id += 1

        # Tracks sin detección → envejecen; los muy viejos se eliminan.
        for tid in list(self._tracks.keys()):
            if tid not in used_tracks:
                self._tracks[tid].missed += 1
                if self._tracks[tid].missed > self.max_missed:
                    del self._tracks[tid]

        return detections


# ------------------------------ session logger ------------------------------


class SessionLogger:
    """
    Registra cada detección a un CSV y, al cerrar la sesión, genera un gráfico
    Gantt de emociones por persona + un resumen.

    CSV: una fila por detección por inferencia. Columnas:
        timestamp_iso, elapsed_s, track_id, class, confidence, x1, y1, x2, y2

    El gráfico requiere matplotlib (opcional). Si no está instalado, se omite
    sin romper la sesión.
    """

    HEADERS = [
        "timestamp_iso", "elapsed_s", "track_id",
        "class", "confidence", "x1", "y1", "x2", "y2",
    ]

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{stamp}"
        self.csv_path = self.log_dir / f"{self.session_id}.csv"
        self.chart_path = self.log_dir / f"{self.session_id}.png"
        self.report_path = self.log_dir / f"{self.session_id}_informe.md"
        self.pdf_path = self.log_dir / f"{self.session_id}_informe.pdf"
        self.snapshots_dir = self.log_dir / f"{self.session_id}_snapshots"

        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)

        self.start_time = time.time()
        self.start_iso = datetime.fromtimestamp(self.start_time).isoformat(timespec="seconds")
        self.row_count = 0
        # Por track_id: lista de (elapsed_s, class, confidence)
        self._timeline: dict[int, list[tuple[float, str, float]]] = {}
        # Por clase: dict con la detección de máxima confianza vista
        # {confidence, elapsed_s, ts_iso, track_id, snapshot_path}
        self._best_per_class: dict[str, dict] = {}

    def log(self, detections: list[Detection], frame: Optional[np.ndarray] = None) -> None:
        if not detections:
            return
        now = time.time()
        elapsed = now - self.start_time
        ts_iso = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
        for det in detections:
            tid = det.track_id if det.track_id is not None else -1
            self._writer.writerow([
                ts_iso, f"{elapsed:.3f}", tid, det.cls,
                f"{det.confidence:.4f}", det.x1, det.y1, det.x2, det.y2,
            ])
            self._timeline.setdefault(tid, []).append((elapsed, det.cls, det.confidence))
            self.row_count += 1

            # Mejor detección vista para esta clase → snapshot representativo.
            current_best = self._best_per_class.get(det.cls)
            if current_best is None or det.confidence > current_best["confidence"]:
                snap_path = None
                if frame is not None:
                    snap_path = self._save_face_crop(frame, det)
                self._best_per_class[det.cls] = {
                    "confidence": det.confidence,
                    "elapsed_s": elapsed,
                    "ts_iso": ts_iso,
                    "track_id": tid,
                    "snapshot_path": snap_path,
                }
        self._file.flush()

    def _save_face_crop(self, frame: np.ndarray, det: Detection) -> Optional[Path]:
        """Guarda un recorte del rostro con un poco de padding."""
        h, w = frame.shape[:2]
        pad = 24
        x1 = max(0, det.x1 - pad)
        y1 = max(0, det.y1 - pad)
        x2 = min(w, det.x2 + pad)
        y2 = min(h, det.y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        safe_cls = "".join(c if c.isalnum() else "_" for c in det.cls.lower())
        path = self.snapshots_dir / f"{safe_cls}.jpg"
        cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return path

    def summary(self) -> dict:
        """Devuelve un resumen estadístico de la sesión."""
        duration = time.time() - self.start_time
        all_classes = [
            cls for events in self._timeline.values() for _, cls, _ in events
        ]
        counts = Counter(all_classes)
        total = sum(counts.values())
        top = [
            (cls, n / total) for cls, n in counts.most_common()
        ] if total else []
        return {
            "duration_s": duration,
            "total_detections": self.row_count,
            "unique_tracks": len([t for t in self._timeline.keys() if t != -1]),
            "top_emotions": top,
        }

    def close(self, generate_chart: bool = True) -> Optional[Path]:
        """Cierra el CSV y genera el gráfico si es posible. Retorna la ruta del PNG."""
        try:
            self._file.close()
        except Exception:
            pass

        if not generate_chart or self.row_count == 0:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")  # backend sin GUI, no abre ventanas
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            logger.warning(
                "matplotlib no está instalado, se omite el gráfico. "
                "Instálalo con: pip install matplotlib"
            )
            return None

        self._render_chart(plt, Rectangle)
        return self.chart_path

    # --------------------------- report (markdown) --------------------------

    def generate_report(
        self,
        chart_path: Optional[Path] = None,
        summary: Optional[dict] = None,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> Optional[Path]:
        """
        Genera un informe Markdown con: duración, distribución por emoción,
        momento de máxima confianza por clase y el snapshot representativo.
        Devuelve la ruta del .md, o None si no había nada que reportar.
        """
        if self.row_count == 0:
            return None
        if summary is None:
            summary = self.summary()

        # Conteo por clase para la tabla.
        class_counts: Counter = Counter()
        for events in self._timeline.values():
            for _, cls, _ in events:
                class_counts[cls] += 1
        total = sum(class_counts.values()) or 1

        lines: list[str] = []
        lines.append(f"# Informe de sesión — análisis emocional")
        lines.append("")
        lines.append(f"- **Sesión:** `{self.session_id}`")
        lines.append(f"- **Inicio:** {self.start_iso}")
        lines.append(f"- **Duración:** {summary['duration_s']:.1f} s")
        lines.append(f"- **Modelo:** `{model_id}`")
        lines.append(f"- **Detecciones registradas:** {summary['total_detections']}")
        lines.append(f"- **Personas únicas detectadas:** {summary['unique_tracks']}")
        lines.append("")

        # ---- Tabla de distribución ----
        lines.append("## Distribución global de emociones")
        lines.append("")
        lines.append("| Emoción | Detecciones | % del total | Barra |")
        lines.append("|---|---:|---:|:---|")
        for cls, count in class_counts.most_common():
            pct = count / total * 100
            bar = "█" * max(1, int(pct / 4))
            lines.append(f"| {cls} | {count} | {pct:5.1f}% | `{bar}` |")
        lines.append("")

        # ---- Visualización temporal (gráfico Gantt + barras) ----
        if chart_path is not None and chart_path.exists():
            lines.append("## Visualización temporal")
            lines.append("")
            lines.append(
                "El siguiente gráfico muestra la línea de tiempo de emociones "
                "detectadas por persona durante la sesión, junto con la distribución "
                "global agregada."
            )
            lines.append("")
            lines.append(f"![Timeline de emociones]({self._relative_path(chart_path)})")
            lines.append("")

        # ---- Detalle por emoción con máximo y snapshot ----
        lines.append("## Momento de máxima confianza por emoción")
        lines.append("")
        if not self._best_per_class:
            lines.append("*No se registraron picos por clase.*")
        else:
            # Ordenar por confianza descendente.
            ordered = sorted(
                self._best_per_class.items(),
                key=lambda kv: kv[1]["confidence"],
                reverse=True,
            )
            for cls, info in ordered:
                lines.append(f"### {cls}")
                lines.append("")
                lines.append(f"- **Confianza máxima:** {info['confidence']*100:.1f}%")
                lines.append(f"- **Momento:** segundo {info['elapsed_s']:.1f} ({info['ts_iso']})")
                if info["track_id"] not in (None, -1):
                    lines.append(f"- **Persona:** #{info['track_id']}")
                snap = info.get("snapshot_path")
                if snap is not None:
                    rel = self._relative_path(snap)
                    lines.append("")
                    lines.append(f"![{cls}]({rel})")
                lines.append("")

        # ---- Archivos generados ----
        lines.append("## Archivos generados")
        lines.append("")
        lines.append(f"- CSV con todas las detecciones: `{self.csv_path.name}`")
        if chart_path is not None and chart_path.exists():
            lines.append(f"- Gráfico (timeline + distribución): `{chart_path.name}`")
        if self.snapshots_dir.exists():
            lines.append(f"- Snapshots representativos: `{self.snapshots_dir.name}/`")
        lines.append("")

        # ---- Notas ----
        lines.append("## Notas metodológicas")
        lines.append("")
        lines.append(
            "- Las emociones provienen del modelo de Roboflow consumido vía endpoint Serverless."
        )
        lines.append(
            "- El % de cada emoción se calcula sobre el total de detecciones registradas, "
            "no sobre el tiempo total de sesión (puede haber frames sin caras)."
        )
        lines.append(
            "- Los snapshots son recortes del rostro en el frame con máxima confianza para esa clase."
        )
        lines.append(
            "- Se aplican smoothing temporal (votación IoU) y tracking por IoU greedy "
            "antes de registrar las detecciones."
        )
        lines.append(
            "- La detección de emociones por imagen estática refleja **expresión facial**, "
            "no estado emocional interno; usar con cautela y contexto."
        )
        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.report_path

    def _relative_path(self, target: Path) -> str:
        """Ruta del target relativa a la del informe, usando '/' (compatible con MD)."""
        try:
            rel = os.path.relpath(target, self.report_path.parent)
        except ValueError:
            rel = str(target)
        return rel.replace(os.sep, "/")

    # ----------------------------- report (PDF) -----------------------------

    # Logo por defecto si existe; puede sobrescribirse con --logo en CLI.
    DEFAULT_LOGO_CANDIDATES = [
        Path(__file__).resolve().parent / "assets" / "logo_udec.png",
        Path(__file__).resolve().parent / "assets" / "logo.png",
    ]

    # Paleta institucional: verde UdeC + neutros para el cuerpo.
    PDF_PRIMARY_HEX = "#1A8A3E"      # verde UdeC
    PDF_PRIMARY_DARK_HEX = "#0F5A28"  # verde más oscuro para títulos
    PDF_TEXT_HEX = "#1F2937"          # gris muy oscuro para cuerpo
    PDF_MUTED_HEX = "#6B7280"         # gris medio para notas

    def generate_pdf_report(
        self,
        chart_path: Optional[Path] = None,
        summary: Optional[dict] = None,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        title: str = "Sistema de Análisis de Emociones Faciales en Tiempo Real",
        authors: Optional[list[str]] = None,
        institution: str = "Universidad de Cundinamarca",
        faculty: str = "Facultad de Ingeniería",
        year: str = "",
        logo_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Genera un informe PDF claro y visual, pensado para lectura por
        profesionales de psicología (no técnica). Incluye:
        - Portada con logo institucional, título, autores y datos.
        - Resumen ejecutivo.
        - Guía de interpretación.
        - Línea de tiempo de emociones (gráfico embebido).
        - Análisis por emoción con snapshot facial representativo.
        - Notas y limitaciones.
        """
        if self.row_count == 0:
            return None
        if summary is None:
            summary = self.summary()
        if authors is None or not authors:
            authors = ["Autor"]
        if not year:
            year = self.start_iso[:4]

        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Image,
                PageBreak, KeepTogether, HRFlowable, Table, TableStyle,
            )
        except ImportError:
            logger.warning(
                "reportlab no está instalado, se omite el PDF. "
                "Instálalo con: pip install reportlab"
            )
            return None

        # Resolver logo (parámetro explícito > candidatos por defecto > None).
        resolved_logo: Optional[Path] = None
        if logo_path is not None and Path(logo_path).exists():
            resolved_logo = Path(logo_path)
        else:
            for candidate in self.DEFAULT_LOGO_CANDIDATES:
                if candidate.exists():
                    resolved_logo = candidate
                    break

        styles = self._pdf_styles(ParagraphStyle, TA_CENTER, TA_LEFT, TA_JUSTIFY, inch, colors)
        story: list = []

        # ============================== PORTADA ==============================
        if resolved_logo is not None:
            story.append(Spacer(1, 0.5 * inch))
            story.append(self._fit_image(resolved_logo, Image, max_width=4.5 * inch, max_height=2.6 * inch, h_align="CENTER"))
            story.append(Spacer(1, 0.9 * inch))
        else:
            story.append(Spacer(1, 2.0 * inch))

        story.append(Paragraph(title, styles["CoverTitle"]))
        story.append(Spacer(1, 1.4 * inch))

        for name in authors:
            story.append(Paragraph(name.strip(), styles["CoverAuthor"]))

        story.append(Spacer(1, 1.6 * inch))
        story.append(Paragraph(institution, styles["CoverInstitutionBold"]))
        story.append(Paragraph(faculty, styles["CoverInstitution"]))
        story.append(Paragraph(year, styles["CoverInstitution"]))

        story.append(PageBreak())

        # ========================= ENCABEZADO DE CUERPO ======================
        story.append(Paragraph(title, styles["DocTitle"]))
        story.append(self._color_rule(HRFlowable, colors, width="100%", thickness=2.0))
        story.append(Spacer(1, 14))
        story.append(Paragraph(
            f"Sesión <b>{self.session_id}</b>  ·  Inicio: {self.start_iso.replace('T', ' ')}  ·  "
            f"Duración: {summary['duration_s']:.1f} s  ·  "
            f"Detecciones: {summary['total_detections']}  ·  "
            f"Personas: {summary['unique_tracks']}",
            styles["MetaLine"],
        ))
        story.append(Spacer(1, 18))

        # ============================== RESUMEN ==============================
        story.append(self._section_heading("Resumen ejecutivo", styles, HRFlowable, colors))
        class_counts: Counter = Counter()
        for events in self._timeline.values():
            for _, cls, _ in events:
                class_counts[cls] += 1
        total = sum(class_counts.values()) or 1
        top3 = class_counts.most_common(3)
        if top3:
            top_str = ", ".join(
                f"<b>{cls}</b> ({n / total * 100:.1f}%)"
                for cls, n in top3
            )
        else:
            top_str = "sin emociones registradas"

        intro = (
            f"Durante una sesión de <b>{summary['duration_s']:.1f} segundos</b> se "
            f"analizó la expresión facial de <b>{summary['unique_tracks']} persona(s)</b> "
            f"a partir de imágenes de cámara web procesadas en tiempo real. El sistema "
            f"registró un total de <b>{summary['total_detections']} detecciones</b> "
            f"clasificadas en distintas categorías de emoción facial. "
            f"Las emociones predominantes fueron: {top_str}. "
            f"Este informe presenta una visualización temporal de la sesión y un "
            f"análisis cualitativo de cada emoción detectada, acompañado del "
            f"momento de máxima confianza observado para cada una."
        )
        story.append(Paragraph(intro, styles["Body"]))

        # ====================== CÓMO INTERPRETAR EL INFORME ==================
        story.append(self._section_heading(
            "Cómo interpretar este informe",
            styles,
            HRFlowable,
            colors
        ))

        intro_guide = (
            "El presente informe fue generado a partir de un sistema de "
            "<b>clasificación automática de expresiones faciales en tiempo real</b>, "
            "implementado mediante visión por computador e inteligencia artificial. "
            "El programa captura imágenes desde la cámara web del equipo y las envía "
            "a un modelo público de detección de emociones faciales alojado en la "
            "plataforma <i>Roboflow</i>, que retorna por cada fotograma las regiones "
            "donde se identifica un rostro junto con la categoría de expresión "
            "predominante. Para que la previsualización en pantalla no se interrumpa, "
            "la inferencia se ejecuta de forma asíncrona en un hilo de fondo. "
            "Las detecciones obtenidas atraviesan dos etapas adicionales antes de "
            "registrarse: un <b>suavizado temporal</b> que vota la clase ganadora "
            "sobre los últimos fotogramas (reduciendo fluctuaciones rápidas), y un "
            "<b>seguimiento por solapamiento espacial</b> que asigna un identificador "
            "persistente a cada rostro mientras permanezca en escena. Toda la "
            "información generada se registra continuamente y se utiliza al cerrar "
            "la sesión para construir el gráfico, los recortes representativos y "
            "este documento."
        )

        story.append(Paragraph(intro_guide, styles["Body"]))

        guide = [
            (
                "<b>Línea de tiempo:</b> el gráfico temporal muestra la evolución "
                "de las expresiones faciales detectadas durante la sesión. Cada "
                "franja horizontal representa una persona identificada temporalmente "
                "por el sistema (#1, #2, …), mientras que los cambios de color "
                "indican variaciones en la categoría de expresión facial detectada."
            ),
            (
                "<b>Confianza del modelo:</b> cada predicción incluye un valor de "
                "confianza entre 0% y 100%, el cual representa el nivel de seguridad "
                "asignado por el modelo para la categoría detectada. Valores altos "
                "indican una mayor similitud entre la expresión observada y los "
                "patrones aprendidos durante el entrenamiento del modelo."
            ),
            (
                "<b>Snapshots representativos:</b> para cada categoría detectada "
                "se conserva el fotograma con la mayor confianza registrada durante "
                "la sesión. Estas imágenes funcionan como evidencia visual del "
                "momento más representativo identificado por el sistema."
            ),
        ]

        for para in guide:
            story.append(Paragraph(para, styles["Body"]))

        # ============== ANÁLISIS TEMPORAL (gráfico Gantt + barras) ===========
        if chart_path is not None and chart_path.exists():
            story.append(PageBreak())
            story.append(self._section_heading("Análisis temporal de la sesión", styles, HRFlowable, colors))
            story.append(Paragraph(
                "El siguiente gráfico muestra, en su panel superior, la evolución "
                "de la emoción dominante por cada persona detectada a lo largo del "
                "tiempo. El panel inferior agrega la distribución global de "
                "emociones durante la sesión.",
                styles["Body"],
            ))
            story.append(Spacer(1, 10))
            story.append(self._fit_image(
                chart_path, Image, max_width=6.5 * inch, max_height=7.5 * inch, h_align="CENTER",
            ))

        # ===================== ANÁLISIS POR EMOCIÓN ==========================
        if self._best_per_class:
            story.append(PageBreak())
            story.append(self._section_heading("Análisis por emoción", styles, HRFlowable, colors))
            story.append(Paragraph(
                "Para cada emoción detectada se presenta el momento de máxima "
                "confianza observado durante la sesión, acompañado del recorte "
                "facial correspondiente como evidencia visual.",
                styles["Body"],
            ))
            story.append(Spacer(1, 10))

            ordered = sorted(
                self._best_per_class.items(),
                key=lambda kv: kv[1]["confidence"], reverse=True,
            )
            for cls, info in ordered:
                story.append(self._emotion_card(
                    cls, info, class_counts.get(cls, 0), total,
                    styles, Paragraph, Spacer, Image, KeepTogether, Table, TableStyle, colors,
                ))

        # ===================== NOTAS METODOLÓGICAS ===========================
        story.append(PageBreak())
        story.append(self._section_heading("Notas metodológicas y limitaciones", styles, HRFlowable, colors))
        notes = [
            ("Las predicciones provienen del modelo público <b>" + model_id + "</b> "
             "alojado en Roboflow Universe, consumido a través de un endpoint "
             "Serverless. Cada fotograma se envía codificado en JPEG y el modelo "
             "retorna las cajas delimitadoras junto con la clase asignada y un "
             "valor de confianza."),
            ("Las detecciones crudas atraviesan dos etapas adicionales antes de "
             "registrarse: un <b>suavizado temporal</b> que vota por mayoría "
             "ponderada sobre los últimos siete fotogramas (reduciendo el "
             "parpadeo entre clases), y un <b>seguimiento por solapamiento</b> "
             "que asigna identificadores persistentes a cada cara mientras "
             "permanezca en escena."),
            ("Los porcentajes presentados se calculan sobre el <b>total de "
             "detecciones registradas</b>, no sobre el tiempo total de sesión. "
             "Períodos sin caras detectadas no influyen en el denominador."),
            ("El reconocimiento automático de emociones por imagen estática es "
             "un campo en discusión activa. La interpretación de los resultados "
             "debe considerar limitaciones conocidas: sesgos del conjunto de "
             "entrenamiento, sensibilidad a iluminación y ángulo, expresiones "
             "mixtas o sutiles, y diferencias culturales en la manifestación de "
             "las emociones. Este informe es un <b>insumo complementario</b> y "
             "no constituye una herramienta diagnóstica."),
        ]
        for para in notes:
            story.append(Paragraph(para, styles["Body"]))

        # ========================== CONSTRUIR DOC ============================
        doc = SimpleDocTemplate(
            str(self.pdf_path),
            pagesize=LETTER,
            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
            topMargin=0.9 * inch, bottomMargin=0.9 * inch,
            title=f"Informe — {self.session_id}",
            author=" / ".join(authors),
            subject="Análisis de detección de emociones faciales",
        )
        doc.build(
            story,
            onFirstPage=self._cover_page_decoration,
            onLaterPages=self._body_page_decoration,
        )
        return self.pdf_path

    # ----------------------------- PDF helpers ------------------------------

    def _pdf_styles(self, ParagraphStyle, TA_CENTER, TA_LEFT, TA_JUSTIFY, inch, colors) -> dict:
        """
        Estilos del PDF basados en Times Roman 12pt para todo el cuerpo.
        Los títulos usan Times Bold con tamaño escalado para jerarquía visual.
        """
        primary_dark = colors.HexColor(self.PDF_PRIMARY_DARK_HEX)
        text = colors.HexColor(self.PDF_TEXT_HEX)
        muted = colors.HexColor(self.PDF_MUTED_HEX)
        BODY_SIZE = 12
        BODY_LEAD = 17  # ~1.4 line height para legibilidad
        return {
            # ---- Portada ----
            "CoverTitle": ParagraphStyle(
                "CoverTitle", fontName="Times-Bold", fontSize=18,
                leading=22, alignment=TA_CENTER, textColor=text,
            ),
            "CoverAuthor": ParagraphStyle(
                "CoverAuthor", fontName="Times-Roman", fontSize=BODY_SIZE,
                leading=18, alignment=TA_CENTER, textColor=text,
            ),
            "CoverInstitutionBold": ParagraphStyle(
                "CoverInstitutionBold", fontName="Times-Bold", fontSize=BODY_SIZE,
                leading=18, alignment=TA_CENTER, textColor=text,
            ),
            "CoverInstitution": ParagraphStyle(
                "CoverInstitution", fontName="Times-Roman", fontSize=BODY_SIZE,
                leading=18, alignment=TA_CENTER, textColor=text,
            ),
            # ---- Cuerpo ----
            "DocTitle": ParagraphStyle(
                "DocTitle", fontName="Times-Bold", fontSize=18,
                leading=22, alignment=TA_LEFT, textColor=primary_dark, spaceAfter=2,
            ),
            "MetaLine": ParagraphStyle(
                "MetaLine", fontName="Times-Roman", fontSize=10,
                leading=14, alignment=TA_LEFT, textColor=muted,
            ),
            "SectionHeading": ParagraphStyle(
                "SectionHeading", fontName="Times-Bold", fontSize=14,
                leading=18, alignment=TA_LEFT, textColor=primary_dark,
                spaceBefore=14, spaceAfter=2,
            ),
            "SubHeading": ParagraphStyle(
                "SubHeading", fontName="Times-Bold", fontSize=BODY_SIZE,
                leading=16, alignment=TA_LEFT, textColor=text,
                spaceBefore=8, spaceAfter=4,
            ),
            "Body": ParagraphStyle(
                "Body", fontName="Times-Roman", fontSize=BODY_SIZE,
                leading=BODY_LEAD, alignment=TA_JUSTIFY, textColor=text,
                spaceAfter=10,
            ),
            "Caption": ParagraphStyle(
                "Caption", fontName="Times-Italic", fontSize=10,
                leading=13, alignment=TA_CENTER, textColor=muted,
                spaceBefore=4,
            ),
            "EmotionLabel": ParagraphStyle(
                "EmotionLabel", fontName="Times-Bold", fontSize=14,
                leading=18, alignment=TA_LEFT, textColor=text,
            ),
            "EmotionStat": ParagraphStyle(
                "EmotionStat", fontName="Times-Roman", fontSize=BODY_SIZE,
                leading=BODY_LEAD, alignment=TA_LEFT, textColor=text,
                spaceAfter=0,
            ),
        }

    def _section_heading(self, text: str, styles: dict, HRFlowable, colors):
        from reportlab.platypus import KeepTogether, Paragraph
        primary = colors.HexColor(self.PDF_PRIMARY_HEX)
        return KeepTogether([
            Paragraph(text, styles["SectionHeading"]),
            HRFlowable(width="100%", thickness=1.2, color=primary,
                       spaceBefore=2, spaceAfter=10),
        ])

    def _color_rule(self, HRFlowable, colors, *, width="100%", thickness: float = 1.0):
        return HRFlowable(width=width, thickness=thickness,
                          color=colors.HexColor(self.PDF_PRIMARY_HEX),
                          spaceBefore=0, spaceAfter=0)

    def _fit_image(self, image_path: Path, Image, *, max_width: float, max_height: float,
                   h_align: str = "LEFT"):
        """Carga una imagen escalada para caber en max_width × max_height, conservando aspecto."""
        img = cv2.imread(str(image_path))
        if img is None:
            from reportlab.platypus import Spacer
            return Spacer(1, 0)
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)
        target_w = w * scale
        target_h = h * scale
        flow = Image(str(image_path), width=target_w, height=target_h)
        flow.hAlign = h_align
        return flow

    def _emotion_card(
        self, cls: str, info: dict, count: int, total: int, styles: dict,
        Paragraph, Spacer, Image, KeepTogether, Table, TableStyle, colors,
    ):
        """
        Tarjeta visual por emoción, centrada y con más información:
        - Encabezado: chip de color con el nombre, centrado.
        - Línea fina divisoria del color de la emoción.
        - Imagen del rostro más grande a la izquierda + estadísticas a la derecha.
        - Pie con caption en cursiva centrada.
        """
        from reportlab.lib.units import inch
        from reportlab.platypus import HRFlowable

        pct = count / total * 100 if total else 0.0
        b, g, r = color_for(cls)
        chip_color = colors.Color(r / 255.0, g / 255.0, b / 255.0)

        # ---- Estadísticas extra calculadas desde el timeline ----
        all_events: list[tuple[float, float]] = []
        for events in self._timeline.values():
            for elapsed, c, conf in events:
                if c == cls:
                    all_events.append((elapsed, conf))
        if all_events:
            confidences = [c for _, c in all_events]
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            first_t = min(t for t, _ in all_events)
            last_t = max(t for t, _ in all_events)
            approx_duration = max(0.0, last_t - first_t)
        else:
            avg_conf = min_conf = first_t = last_t = approx_duration = 0.0

        # ---- 1) Cabecera: chip + nombre, centrado ----
        chip_header = Table(
            [["", Paragraph(cls.capitalize(), styles["EmotionLabel"])]],
            colWidths=[0.40 * inch, 2.6 * inch],
        )
        chip_header.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), chip_color),
            ("BOX", (0, 0), (0, 0), 0.6, colors.HexColor(self.PDF_TEXT_HEX)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (1, 0), (1, 0), 10),
        ]))
        chip_header.hAlign = "CENTER"

        # ---- 2) Línea separadora con el color de la emoción ----
        accent_line = HRFlowable(
            width=3.2 * inch, thickness=1.0, color=chip_color,
            hAlign="CENTER", spaceBefore=2, spaceAfter=8,
        )

        # ---- 3) Estadísticas (más información) ----
        stats_html = (
            f"<b>Confianza máxima:</b> {info['confidence'] * 100:.1f}%<br/>"
            f"<b>Confianza promedio:</b> {avg_conf * 100:.1f}%<br/>"
            f"<b>Confianza mínima:</b> {min_conf * 100:.1f}%<br/>"
            f"<b>Momento del pico:</b> segundo {info['elapsed_s']:.1f}<br/>"
            f"<b>Primera aparición:</b> segundo {first_t:.1f}<br/>"
            f"<b>Última aparición:</b> segundo {last_t:.1f}<br/>"
            f"<b>Duración aproximada:</b> {approx_duration:.1f} s<br/>"
            f"<b>Frecuencia:</b> {count} detecciones ({pct:.1f}% del total)"
        )
        if info["track_id"] not in (None, -1):
            stats_html += f"<br/><b>Persona identificada:</b> #{info['track_id']}"

        stats_para = Paragraph(stats_html, styles["EmotionStat"])

        # ---- 4) Layout principal: imagen + stats, centrado ----
        snap = info.get("snapshot_path")
        if snap is not None and Path(snap).exists():
            img_flow = self._fit_image(
                Path(snap), Image,
                max_width=2.5 * inch, max_height=3.0 * inch,
                h_align="CENTER",
            )
            content_table = Table(
                [[img_flow, stats_para]],
                colWidths=[2.7 * inch, 3.6 * inch],
            )
            content_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 14),
            ]))
            content_table.hAlign = "CENTER"

            caption = Paragraph(
                f"<i>Recorte facial del momento de máxima confianza para "
                f"«{cls}» (t = {info['elapsed_s']:.1f} s, "
                f"{info['confidence'] * 100:.1f}%).</i>",
                styles["Caption"],
            )

            inner = [
                chip_header, accent_line,
                content_table, Spacer(1, 6),
                caption, Spacer(1, 22),
            ]
        else:
            inner = [
                chip_header, accent_line,
                stats_para, Spacer(1, 22),
            ]

        return KeepTogether(inner)

    def _cover_page_decoration(self, canvas_obj, doc) -> None:
        """Sutil barra verde inferior en la portada."""
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib import colors
        canvas_obj.saveState()
        canvas_obj.setFillColor(colors.HexColor(self.PDF_PRIMARY_HEX))
        canvas_obj.rect(0, 0, LETTER[0], 18, stroke=0, fill=1)
        canvas_obj.restoreState()

    def _body_page_decoration(self, canvas_obj, doc) -> None:
        """Pie de página con número y nombre de sesión, más banda superior fina."""
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        canvas_obj.saveState()
        # Banda superior fina con color institucional.
        canvas_obj.setFillColor(colors.HexColor(self.PDF_PRIMARY_HEX))
        canvas_obj.rect(0, LETTER[1] - 8, LETTER[0], 8, stroke=0, fill=1)
        # Pie: izquierda = id de sesión, derecha = página.
        canvas_obj.setFillColor(colors.HexColor(self.PDF_MUTED_HEX))
        canvas_obj.setFont("Times-Roman", 10)
        canvas_obj.drawString(0.9 * inch, 0.5 * inch, f"Sesión {self.session_id}")
        canvas_obj.drawRightString(
            LETTER[0] - 0.9 * inch, 0.5 * inch,
            f"Página {canvas_obj.getPageNumber()}",
        )
        canvas_obj.restoreState()

    # --------------------------- chart rendering ----------------------------

    def _render_chart(self, plt, Rectangle) -> None:
        """Genera dos paneles: Gantt por persona + barras totales por emoción."""
        timeline = {tid: events for tid, events in self._timeline.items() if events}
        if not timeline:
            return

        # Orden por aparición.
        track_ids = sorted(timeline.keys(), key=lambda t: timeline[t][0][0])
        total_duration = max(events[-1][0] for events in timeline.values())
        if total_duration <= 0:
            total_duration = 1.0

        fig = plt.figure(figsize=(14, max(4, 1.0 + len(track_ids) * 0.7) + 3))
        gs = fig.add_gridspec(2, 1, height_ratios=[len(track_ids) + 1, 2.5], hspace=0.45)
        ax_gantt = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        # ---- Gantt ----
        seen_classes: set[str] = set()
        for y_idx, tid in enumerate(track_ids):
            events = timeline[tid]
            cur_start = events[0][0]
            cur_class = events[0][1]
            for t, cls, _ in events[1:]:
                if cls != cur_class:
                    self._draw_segment(ax_gantt, Rectangle, cur_start, t, y_idx, cur_class)
                    seen_classes.add(cur_class)
                    cur_start = t
                    cur_class = cls
            # Último segmento (le damos un mínimo de ancho para que se vea).
            end = max(events[-1][0], cur_start + 0.3)
            self._draw_segment(ax_gantt, Rectangle, cur_start, end, y_idx, cur_class)
            seen_classes.add(cur_class)

        ax_gantt.set_xlim(0, total_duration)
        ax_gantt.set_ylim(-0.5, len(track_ids) - 0.5)
        ax_gantt.set_yticks(range(len(track_ids)))
        ax_gantt.set_yticklabels(
            [f"#{tid}" if tid != -1 else "sin ID" for tid in track_ids]
        )
        ax_gantt.invert_yaxis()
        ax_gantt.set_xlabel("Tiempo (s)")
        ax_gantt.set_ylabel("Persona")
        ax_gantt.set_title("Timeline de emociones por persona")
        ax_gantt.grid(axis="x", linestyle=":", alpha=0.4)

        # Leyenda compartida.
        legend_handles = []
        for cls in sorted(seen_classes):
            legend_handles.append(
                Rectangle((0, 0), 1, 1, facecolor=self._matplotlib_color(cls),
                          edgecolor="black", label=cls)
            )
        ax_gantt.legend(
            handles=legend_handles, loc="upper right",
            title="Emoción", fontsize=9, ncol=min(4, len(legend_handles)),
        )

        # ---- Barras de % de tiempo por emoción (global) ----
        all_classes = [cls for events in timeline.values() for _, cls, _ in events]
        counts = Counter(all_classes)
        total = sum(counts.values()) or 1
        ordered = counts.most_common()
        labels = [c for c, _ in ordered]
        pcts = [n / total * 100 for _, n in ordered]
        colors = [self._matplotlib_color(c) for c in labels]

        bars = ax_bar.bar(labels, pcts, color=colors, edgecolor="black")
        for bar, pct in zip(bars, pcts):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9,
            )
        ax_bar.set_ylabel("% de detecciones")
        ax_bar.set_title("Distribución global de emociones en la sesión")
        ax_bar.set_ylim(0, max(pcts) * 1.15 if pcts else 100)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.4)

        fig.suptitle(
            f"Análisis emocional — {self.csv_path.stem}",
            fontsize=12, fontweight="bold", y=0.995,
        )
        fig.savefig(self.chart_path, dpi=120, bbox_inches="tight")
        # plt.close vendría de plt importado, pero ya tenemos plt en el closure.
        plt.close(fig)

    @staticmethod
    def _draw_segment(ax, Rectangle, start: float, end: float, y_idx: int, cls: str) -> None:
        ax.add_patch(Rectangle(
            (start, y_idx - 0.4), max(end - start, 0.05), 0.8,
            facecolor=SessionLogger._matplotlib_color(cls),
            edgecolor="black", linewidth=0.4,
        ))

    @staticmethod
    def _matplotlib_color(cls: str) -> tuple[float, float, float]:
        """Convierte el color BGR de OpenCV a RGB normalizado para matplotlib."""
        b, g, r = color_for(cls)
        return (r / 255.0, g / 255.0, b / 255.0)


# --------------------------------- dibujo -----------------------------------


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[Detection],
    conf_threshold: float,
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    pad = 4

    for det in detections:
        if det.confidence < conf_threshold:
            continue

        color = color_for(det.cls)
        cv2.rectangle(image_bgr, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        prefix = f"#{det.track_id} " if det.track_id is not None else ""
        label = f"{prefix}{det.cls} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

        # Si la etiqueta cabe encima del bbox, va arriba; si no, dentro del bbox.
        if det.y1 - th - 2 * pad >= 0:
            bg_y1 = det.y1 - th - 2 * pad
            bg_y2 = det.y1
        else:
            bg_y1 = det.y1
            bg_y2 = det.y1 + th + 2 * pad
        bg_x1 = det.x1
        bg_x2 = min(image_bgr.shape[1] - 1, det.x1 + tw + 2 * pad)

        cv2.rectangle(image_bgr, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            image_bgr,
            label,
            (bg_x1 + pad, bg_y2 - pad),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return image_bgr


def _wrap_text_to_width(
    text: str,
    max_width: int,
    *,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.55,
    thickness: int = 1,
    sep: str = " | ",
) -> list[str]:
    """
    Divide un texto unido por `sep` en varias líneas para que cada una
    quepa dentro de `max_width` píxeles. Mide cada parte con getTextSize.
    """
    parts = text.split(sep)
    sep_w = cv2.getTextSize(sep, font, scale, thickness)[0][0]
    lines: list[str] = []
    current: list[str] = []
    current_w = 0
    for part in parts:
        part_w = cv2.getTextSize(part, font, scale, thickness)[0][0]
        added_w = part_w + (sep_w if current else 0)
        if current and current_w + added_w > max_width:
            lines.append(sep.join(current))
            current = [part]
            current_w = part_w
        else:
            current.append(part)
            current_w += added_w
    if current:
        lines.append(sep.join(current))
    return lines


def draw_hud(
    image_bgr: np.ndarray,
    fps: float,
    api_fps: float,
    conf_threshold: float,
    paused: bool,
    last_latency_ms: Optional[float],
    smoothing_enabled: bool = False,
    smooth_window: int = 0,
    tracking_enabled: bool = False,
    active_tracks: int = 0,
    show_hud: bool = True,
) -> np.ndarray:
    if not show_hud:
        return image_bgr

    lat_str = f"  |  lat: {last_latency_ms:.0f}ms" if last_latency_ms is not None else ""
    smooth_str = (
        f"smooth: ON ({smooth_window}f)" if smoothing_enabled else "smooth: OFF"
    )
    track_str = (
        f"track: ON ({active_tracks} caras)" if tracking_enabled else "track: OFF"
    )
    controls = "q/ESC salir | s snapshot | space pausa | +/- umbral | m espejo | t smooth | i track | h ocultar HUD"

    # Margen lateral igual al de la posición de inicio del texto (10 px).
    max_w = max(50, image_bgr.shape[1] - 20)

    lines: list[str] = [
        f"FPS: {fps:.1f}  |  API/s: {api_fps:.1f}{lat_str}",
        f"umbral: {conf_threshold:.2f}  |  {smooth_str}  |  {track_str}  |  {'PAUSA' if paused else 'EN VIVO'}",
    ]
    # Solo la línea de controles necesita wrap: las anteriores casi siempre caben.
    lines.extend(_wrap_text_to_width(controls, max_w))

    x, y = 10, 22
    for line in lines:
        cv2.putText(image_bgr, line, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image_bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return image_bgr


# -------------------------------- API client --------------------------------


class RoboflowClient:
    """Cliente REST mínimo para Roboflow Serverless, con reintentos y timeout."""

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_MODEL_ID,
        api_url: str = DEFAULT_API_URL,
        jpeg_quality: int = 75,
        timeout_s: float = 8.0,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.endpoint = f"{api_url.rstrip('/')}/{model_id}"
        self.jpeg_quality = max(1, min(100, jpeg_quality))
        self.timeout_s = timeout_s

        retry = Retry(
            total=3,
            connect=3,
            read=2,
            backoff_factor=0.4,
            status_forcelist=(408, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def infer(self, frame_bgr: np.ndarray) -> tuple[dict, float]:
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("No se pudo codificar el frame a JPEG.")
        payload = base64.b64encode(buf.tobytes())

        t0 = time.time()
        resp = self.session.post(
            self.endpoint,
            params={"api_key": self.api_key},
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout_s,
        )
        latency = time.time() - t0
        resp.raise_for_status()
        return resp.json(), latency

    def close(self) -> None:
        self.session.close()


# ------------------------------ inference worker ----------------------------


class InferenceWorker(threading.Thread):
    """
    Hilo de inferencia con política 'latest frame': siempre trabaja sobre el
    último frame disponible y descarta los anteriores que aún no procesó.
    """

    def __init__(self, client: RoboflowClient) -> None:
        super().__init__(daemon=True, name="InferenceWorker")
        self.client = client
        self._frame_lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._frame_event = threading.Event()
        self._result_lock = threading.Lock()
        self._latest_result: Optional[InferenceResult] = None
        self._stop_event = threading.Event()
        self._paused = False
        # Señal de fallo de autenticación. Si la API responde 401/403,
        # no tiene sentido reintentar: el worker se detiene y el bucle
        # principal lo detecta para salir con un mensaje claro.
        self.auth_failed = threading.Event()
        self.auth_status_code: Optional[int] = None
        self.auth_message: Optional[str] = None

    def submit(self, frame_bgr: np.ndarray) -> None:
        if self._paused:
            return
        with self._frame_lock:
            self._pending_frame = frame_bgr
        self._frame_event.set()

    def latest(self) -> Optional[InferenceResult]:
        with self._result_lock:
            return self._latest_result

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def stop(self) -> None:
        self._stop_event.set()
        self._frame_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._frame_event.wait(timeout=0.5)
            if self._stop_event.is_set():
                break
            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._frame_event.clear()
            if frame is None:
                continue

            try:
                raw, latency = self.client.infer(frame)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else None
                # 401 / 403 son fallos de credenciales: no se arreglan reintentando.
                # Señalamos el error y salimos del hilo para que el bucle principal cierre la app.
                if code in (401, 403):
                    self.auth_status_code = code
                    self.auth_message = (
                        "API Key inválida o sin permisos para este modelo "
                        f"(HTTP {code}). Verifica ROBOFLOW_API_KEY."
                    )
                    logger.error(self.auth_message)
                    self.auth_failed.set()
                    return
                logger.warning("HTTP %s al inferir: %s", code if code is not None else "?", exc)
                time.sleep(0.5)
                continue
            except requests.RequestException as exc:
                logger.warning("Error de red al inferir: %s", exc)
                time.sleep(0.5)
                continue
            except Exception as exc:
                logger.exception("Error inesperado en inferencia: %s", exc)
                time.sleep(0.5)
                continue

            detections = parse_predictions(raw, frame.shape[:2])
            with self._result_lock:
                self._latest_result = InferenceResult(
                    detections=detections,
                    raw=raw,
                    latency_s=latency,
                    received_at=time.time(),
                )


# --------------------------------- helpers ----------------------------------


def get_api_key(arg_key: Optional[str]) -> str:
    if arg_key:
        return arg_key.strip()
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        return env_key.strip()
    print("Pega tu Private API Key de Roboflow (https://app.roboflow.com/settings/api):")
    key = input("API Key: ").strip()
    if not key:
        raise SystemExit("No se proporcionó API Key.")
    return key


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara con índice {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def save_snapshot(frame_bgr: np.ndarray) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path


def _print_session_summary(
    session_logger: "SessionLogger",
    summary: dict,
    chart_path: Optional[Path],
    report_path: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
) -> None:
    """Imprime un bloque legible al cerrar la sesión."""
    print("\n" + "=" * 60)
    print(" RESUMEN DE SESIÓN")
    print("=" * 60)
    print(f"  Duración:            {summary['duration_s']:.1f} s")
    print(f"  Detecciones (filas): {summary['total_detections']}")
    print(f"  Personas únicas:     {summary['unique_tracks']}")
    if summary["top_emotions"]:
        print("  Distribución:")
        for cls, frac in summary["top_emotions"][:6]:
            bar = "█" * int(frac * 30)
            print(f"    {cls:<12} {frac * 100:5.1f}%  {bar}")
    print(f"  CSV:       {session_logger.csv_path}")
    if chart_path is not None:
        print(f"  Gráfico:   {chart_path}")
    elif summary["total_detections"] > 0:
        print("  Gráfico:   (omitido — instala matplotlib para activarlo)")
    if report_path is not None:
        print(f"  Informe MD:  {report_path}")
    if pdf_path is not None:
        print(f"  Informe PDF: {pdf_path}")
    if session_logger.snapshots_dir.exists():
        n_snaps = len(list(session_logger.snapshots_dir.glob("*.jpg")))
        print(f"  Snapshots:   {session_logger.snapshots_dir}  ({n_snaps} imágenes)")
    print("=" * 60)


# ---------------------------------- modos -----------------------------------


def infer_image(client: RoboflowClient, image_path: str, conf_threshold: float) -> None:
    if not os.path.isfile(image_path):
        raise SystemExit(f"No se encontró la imagen: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise SystemExit(f"OpenCV no pudo leer la imagen: {image_path}")

    try:
        raw, latency = client.infer(image_bgr)
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else None
        if code in (401, 403):
            raise SystemExit(
                f"API Key inválida o sin permisos (HTTP {code}). "
                "Revisa ROBOFLOW_API_KEY en https://app.roboflow.com/settings/api"
            )
        raise
    detections = parse_predictions(raw, image_bgr.shape[:2])
    logger.info("Inferencia OK en %.2fs — %d detecciones", latency, len(detections))

    visible = [d for d in detections if d.confidence >= conf_threshold]
    if visible:
        for i, d in enumerate(visible, 1):
            print(f"{i}. {d.cls}: {d.confidence:.1%}")
    else:
        print("No hay predicciones por encima del umbral.")

    annotated = draw_detections(image_bgr.copy(), detections, conf_threshold)
    out_path = Path(image_path).with_name(Path(image_path).stem + "_anotada.jpg")
    cv2.imwrite(str(out_path), annotated)
    logger.info("Imagen anotada guardada: %s", out_path)

    cv2.imshow("Emociones - imagen (q para cerrar)", annotated)
    while True:
        if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def run_realtime(
    client: RoboflowClient,
    camera_index: int,
    conf_threshold: float,
    width: int,
    height: int,
    mirror: bool,
    smoothing_enabled: bool = True,
    smooth_window: int = 7,
    smooth_iou: float = 0.4,
    tracking_enabled: bool = True,
    track_iou: float = 0.3,
    track_max_missed: int = 15,
    logging_enabled: bool = True,
    log_dir: Optional[Path] = None,
    generate_chart: bool = True,
    generate_report: bool = True,
    generate_pdf: bool = True,
    model_id: str = DEFAULT_MODEL_ID,
    pdf_title: str = "Sistema de Análisis de Emociones Faciales en Tiempo Real",
    pdf_authors: Optional[list[str]] = None,
    pdf_institution: str = "Universidad de Cundinamarca",
    pdf_faculty: str = "Facultad de Ingeniería",
    pdf_year: str = "",
    pdf_logo: Optional[Path] = None,
) -> None:
    cap = open_camera(camera_index, width, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Cámara %d abierta a %dx%d", camera_index, actual_w, actual_h)

    worker = InferenceWorker(client)
    worker.start()

    smoother = DetectionSmoother(window_size=smooth_window, iou_threshold=smooth_iou)
    tracker = FaceTracker(iou_threshold=track_iou, max_missed=track_max_missed)
    last_detections: list[Detection] = []  # se redibujan cada frame, se actualizan solo con resultados nuevos
    logger.info(
        "Smoothing %s (ventana=%d, IoU=%.2f)",
        "ACTIVO" if smoothing_enabled else "DESACTIVADO",
        smooth_window, smooth_iou,
    )
    logger.info(
        "Tracking %s (IoU=%.2f, max_missed=%d)",
        "ACTIVO" if tracking_enabled else "DESACTIVADO",
        track_iou, track_max_missed,
    )

    session_logger: Optional[SessionLogger] = None
    if logging_enabled:
        session_logger = SessionLogger(log_dir=log_dir or SESSIONS_DIR)
        logger.info("Logging activo → %s", session_logger.csv_path)

    paused = False
    show_hud = True
    fps_samples: list[float] = []
    api_count = 0
    last_seen_result_ts = 0.0
    started = time.time()
    last_tick = started

    window = "Roboflow Emociones - q/ESC salir"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    auth_aborted = False
    try:
        while True:
            # Si el worker detectó credenciales inválidas, no tiene sentido seguir.
            if worker.auth_failed.is_set():
                auth_aborted = True
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                logger.warning("Frame inválido de la cámara, reintentando...")
                time.sleep(0.05)
                continue

            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # Pasar copia al worker para que el dibujo posterior no altere su frame.
            worker.submit(frame_bgr.copy())

            result = worker.latest()
            if result is not None and result.received_at > last_seen_result_ts:
                api_count += 1
                last_seen_result_ts = result.received_at
                # Solo procesamos cuando llega un resultado nuevo del worker;
                # entre frames intermedios reusamos las detecciones suavizadas
                # anteriores para que no haya parpadeo.
                if smoothing_enabled:
                    processed = smoother.update(result.detections)
                else:
                    processed = result.detections
                if tracking_enabled:
                    processed = tracker.update(processed)
                last_detections = processed
                # Registrar solo las detecciones que superan el umbral visible:
                # eso mantiene el CSV libre de ruido de baja confianza.
                # Pasamos el frame ANTES de dibujar sobre él, para que los
                # snapshots representativos queden limpios (sin overlays).
                if session_logger is not None:
                    session_logger.log(
                        [d for d in last_detections if d.confidence >= conf_threshold],
                        frame=frame_bgr,
                    )

            display = frame_bgr  # se dibuja sobre el frame que se mostrará
            if last_detections:
                draw_detections(display, last_detections, conf_threshold)

            now = time.time()
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                fps_samples.append(1.0 / dt)
                if len(fps_samples) > 30:
                    fps_samples.pop(0)
            fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
            elapsed = max(now - started, 1e-6)
            api_fps = api_count / elapsed
            last_lat_ms = result.latency_s * 1000 if result else None

            draw_hud(
                display, fps, api_fps, conf_threshold, paused, last_lat_ms,
                smoothing_enabled=smoothing_enabled,
                smooth_window=smooth_window,
                tracking_enabled=tracking_enabled,
                active_tracks=tracker.active_count(),
                show_hud=show_hud,
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                path = save_snapshot(display)
                logger.info("Snapshot guardado: %s", path)
            elif key == ord(" "):
                paused = not paused
                worker.set_paused(paused)
                logger.info("Pausa: %s", paused)
            elif key in (ord("+"), ord("=")):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                logger.info("Umbral: %.2f", conf_threshold)
            elif key in (ord("-"), ord("_")):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                logger.info("Umbral: %.2f", conf_threshold)
            elif key == ord("m"):
                mirror = not mirror
                logger.info("Espejo: %s", mirror)
            elif key == ord("t"):
                smoothing_enabled = not smoothing_enabled
                if smoothing_enabled:
                    smoother.reset()  # arrancar limpio para no mezclar con histórico viejo
                logger.info("Smoothing: %s", "ON" if smoothing_enabled else "OFF")
            elif key == ord("i"):
                tracking_enabled = not tracking_enabled
                if tracking_enabled:
                    tracker.reset()  # IDs vuelven a empezar desde 1
                logger.info("Tracking: %s", "ON" if tracking_enabled else "OFF")
            elif key == ord("h"):
                show_hud = not show_hud
                logger.info("HUD: %s (pulsa 'h' para volver a mostrar)",
                            "ON" if show_hud else "OFF")

    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario.")
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        client.close()
        logger.info("Sesión finalizada. Inferencias completadas: %d", api_count)

        if session_logger is not None:
            summary = session_logger.summary()
            chart_path = session_logger.close(generate_chart=generate_chart)
            report_path = None
            pdf_path = None
            if generate_report:
                try:
                    report_path = session_logger.generate_report(
                        chart_path=chart_path, summary=summary, model_id=model_id,
                    )
                except Exception as exc:
                    logger.warning("No se pudo generar el informe Markdown: %s", exc)
            if generate_pdf:
                try:
                    pdf_path = session_logger.generate_pdf_report(
                        chart_path=chart_path, summary=summary, model_id=model_id,
                        title=pdf_title,
                        authors=pdf_authors,
                        institution=pdf_institution,
                        faculty=pdf_faculty,
                        year=pdf_year,
                        logo_path=pdf_logo,
                    )
                except Exception as exc:
                    logger.warning("No se pudo generar el informe PDF: %s", exc)
            _print_session_summary(
                session_logger, summary, chart_path, report_path, pdf_path,
            )
        if auth_aborted:
            msg = worker.auth_message or "Fallo de autenticación con Roboflow."
            logger.error("Salida forzada: %s", msg)
            print(
                "\n"
                "============================================================\n"
                f" {msg}\n"
                "  Soluciones:\n"
                "    1) Genera/copia tu Private API Key en:\n"
                "         https://app.roboflow.com/settings/api\n"
                "    2) Defínela antes de ejecutar:\n"
                "         $env:ROBOFLOW_API_KEY = \"rf_tu_key_real\"\n"
                "    3) O pásala con --api-key rf_tu_key_real\n"
                "============================================================"
            )
            raise SystemExit(2)


# ----------------------------------- CLI ------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detección de emociones con Roboflow (local, robusto).")
    p.add_argument("--image", type=str, default=None, help="Ruta a una imagen para inferencia única.")
    p.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default 0).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza inicial (0-1).")
    p.add_argument("--model", type=str, default=os.environ.get("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID))
    p.add_argument("--api-url", type=str, default=os.environ.get("ROBOFLOW_API_URL", DEFAULT_API_URL))
    p.add_argument("--api-key", type=str, default=None, help="API Key (sino usa env ROBOFLOW_API_KEY).")
    p.add_argument("--jpeg-quality", type=int, default=75, help="Calidad JPEG enviada al modelo (1-100).")
    p.add_argument("--timeout", type=float, default=8.0, help="Timeout HTTP por request (s).")
    p.add_argument("--no-mirror", action="store_true", help="Desactiva el modo espejo en la webcam.")
    p.add_argument("--no-smooth", action="store_true",
                   help="Desactiva el suavizado temporal (más reactivo pero parpadea).")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="Tamaño de la ventana del suavizado en frames (default: 7).")
    p.add_argument("--smooth-iou", type=float, default=0.4,
                   help="IoU mínimo para considerar dos detecciones la misma cara (default: 0.4).")
    p.add_argument("--no-tracking", action="store_true",
                   help="Desactiva el tracking con IDs persistentes.")
    p.add_argument("--track-iou", type=float, default=0.3,
                   help="IoU mínimo para considerar match con un track existente (default: 0.3).")
    p.add_argument("--track-max-missed", type=int, default=15,
                   help="Frames consecutivos sin detección antes de descartar un track (default: 15).")
    p.add_argument("--no-log", action="store_true",
                   help="Desactiva el registro CSV y el gráfico final de la sesión.")
    p.add_argument("--log-dir", type=str, default=None,
                   help=f"Directorio para CSV y gráficos (default: {SESSIONS_DIR}).")
    p.add_argument("--no-chart", action="store_true",
                   help="Registra el CSV pero no genera el gráfico final.")
    p.add_argument("--no-report", action="store_true",
                   help="No genera el informe Markdown ni los snapshots representativos.")
    p.add_argument("--no-pdf", action="store_true",
                   help="No genera el informe en PDF.")
    p.add_argument("--title", type=str,
                   default=os.environ.get(
                       "REPORT_TITLE",
                       "Sistema de Análisis de Emociones Faciales en Tiempo Real",
                   ),
                   help="Título mostrado en la portada del PDF (env: REPORT_TITLE).")
    p.add_argument("--authors", type=str,
                   default=os.environ.get(
                       "REPORT_AUTHORS",
                       "Cristian Camilo Posada García;"
                       "Luis Felipe Espinel Botina;"
                       "Michael Steven Naranjo Bautista",
                   ),
                   help="Lista de autores separados por ';' (env: REPORT_AUTHORS).")
    p.add_argument("--institution", type=str,
                   default=os.environ.get("REPORT_INSTITUTION", "Universidad de Cundinamarca"),
                   help="Institución en la portada (env: REPORT_INSTITUTION).")
    p.add_argument("--faculty", type=str,
                   default=os.environ.get("REPORT_FACULTY", "Facultad de Ingeniería"),
                   help="Facultad en la portada (env: REPORT_FACULTY).")
    p.add_argument("--year", type=str,
                   default=os.environ.get("REPORT_YEAR", ""),
                   help="Año en la portada (default: año de la sesión).")
    p.add_argument("--logo", type=str,
                   default=os.environ.get("REPORT_LOGO", ""),
                   help="Ruta al logo institucional. Si se omite, busca "
                        "Roboflow/assets/logo_udec.png o Roboflow/assets/logo.png.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    api_key = get_api_key(args.api_key)
    client = RoboflowClient(
        api_key=api_key,
        model_id=args.model,
        api_url=args.api_url,
        jpeg_quality=args.jpeg_quality,
        timeout_s=args.timeout,
    )

    if args.image:
        infer_image(client, args.image, conf_threshold=args.conf)
    else:
        run_realtime(
            client=client,
            camera_index=args.camera,
            conf_threshold=args.conf,
            width=args.width,
            height=args.height,
            mirror=not args.no_mirror,
            smoothing_enabled=not args.no_smooth,
            smooth_window=max(1, args.smooth_window),
            smooth_iou=max(0.0, min(1.0, args.smooth_iou)),
            tracking_enabled=not args.no_tracking,
            track_iou=max(0.0, min(1.0, args.track_iou)),
            track_max_missed=max(1, args.track_max_missed),
            logging_enabled=not args.no_log,
            log_dir=Path(args.log_dir) if args.log_dir else None,
            generate_chart=not args.no_chart,
            generate_report=not args.no_report,
            generate_pdf=not args.no_pdf,
            model_id=args.model,
            pdf_title=args.title,
            pdf_authors=[a.strip() for a in args.authors.split(";") if a.strip()],
            pdf_institution=args.institution,
            pdf_faculty=args.faculty,
            pdf_year=args.year,
            pdf_logo=Path(args.logo) if args.logo else None,
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
