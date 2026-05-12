# Detección de Emociones Faciales en Tiempo Real

Aplicación de visión por computador que detecta y clasifica **expresiones faciales en tiempo real** desde la webcam, usando un modelo público de [Roboflow Universe](https://universe.roboflow.com/). El sistema procesa video en vivo, suaviza las predicciones, hace seguimiento persistente de cada rostro y al finalizar la sesión genera automáticamente un **informe en PDF y Markdown** con gráfico, estadísticas y recortes representativos de cada emoción.

Pensado para uso académico y exploratorio. Implementado en Python puro con OpenCV.

---

## Tabla de contenidos

- [Características](#características)
- [Demo rápida](#demo-rápida)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Cómo obtener tu API Key de Roboflow](#cómo-obtener-tu-api-key-de-roboflow)
- [Uso](#uso)
  - [Modo cámara (tiempo real)](#modo-cámara-tiempo-real)
  - [Modo imagen única](#modo-imagen-única)
  - [Controles en caliente](#controles-en-caliente)
  - [Referencia de CLI completa](#referencia-de-cli-completa)
- [Cómo funciona](#cómo-funciona)
  - [Arquitectura general](#arquitectura-general)
  - [Pipeline de procesamiento](#pipeline-de-procesamiento)
  - [Componentes principales del código](#componentes-principales-del-código)
- [Salidas que genera el programa](#salidas-que-genera-el-programa)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Configuración avanzada](#configuración-avanzada)
- [Solución de problemas](#solución-de-problemas)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Mejoras futuras](#mejoras-futuras)
- [Tecnologías](#tecnologías)
- [Autores](#autores)
- [Licencia](#licencia)

---

## Características

- **Inferencia en tiempo real** desde la webcam local, sin congelar la previsualización: el cliente HTTP corre en un hilo de fondo.
- **Modelo Roboflow Universe** consumido vía REST (sin `inference-sdk`, lo que evita conflictos de dependencias con Pillow/supervision).
- **Suavizado temporal** por votación ponderada en IoU para eliminar el parpadeo entre clases.
- **Tracking por IoU** que asigna identificadores persistentes (`#1`, `#2`, …) a cada rostro mientras esté en escena.
- **Reintentos automáticos** con backoff sobre errores HTTP transitorios (429 / 5xx) y **bailout inmediato** sobre fallos de autenticación (401 / 403).
- **HUD** configurable con métricas en vivo (FPS, latencia, llamadas/segundo), umbral ajustable en caliente y ocultable.
- **CSV** con cada detección registrada (timestamp, persona, clase, confianza, bbox).
- **Gráfico Gantt** del estado emocional por persona a lo largo del tiempo + barras de distribución global.
- **Snapshots representativos** del rostro en el momento de máxima confianza por emoción.
- **Informe automático en Markdown** (visible directo en VS Code / GitHub) y **PDF profesional** con portada institucional, secciones claras y tipografía Times New Roman.

---

## Demo rápida

```powershell
# 1. Instala dependencias
pip install -r requirements.txt

# 2. Define tu API Key (una vez por sesión de terminal)
$env:ROBOFLOW_API_KEY = "rf_tu_key_real"

# 3. Ejecuta
python roboflow_emociones.py
```

Se abre una ventana de OpenCV con la webcam. Cuando cierras la ventana (`q` o `ESC`), se genera automáticamente:

```
sessions/
├─ session_20260508_193645.csv                ← datos crudos
├─ session_20260508_193645.png                ← gráfico Gantt + barras
├─ session_20260508_193645_informe.md         ← informe Markdown
├─ session_20260508_193645_informe.pdf        ← informe PDF
└─ session_20260508_193645_snapshots/
   ├─ happy.jpg
   ├─ neutral.jpg
   └─ ...
```

---

## Requisitos

| Componente | Versión recomendada |
|---|---|
| Python | 3.10 o superior |
| Webcam | cualquiera reconocida por OpenCV |
| Conexión a internet | sí (para llamar a la API de Roboflow) |
| Sistema operativo | Windows, macOS o Linux |

**Dependencias Python** (ver `requirements.txt`):

- `opencv-python` — captura de cámara, renderizado, ventana
- `numpy` — manejo de buffers de imagen
- `requests` — cliente HTTP con reintentos
- `matplotlib` *(opcional)* — generación del gráfico Gantt
- `reportlab` *(opcional)* — generación del PDF
- `python-dotenv` *(opcional)* — carga de credenciales desde `.env`

Si `matplotlib` o `reportlab` no están instalados, el script funciona igual pero omite la salida correspondiente con un warning.

---

## Instalación

```powershell
# Clona el repositorio
git clone https://github.com/FelipeEsp07/roboflow-emociones.git
cd roboflow-emociones

# (Opcional pero recomendado) crea un entorno virtual
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate   # macOS / Linux

# Instala dependencias
pip install -r requirements.txt
```

---

## Cómo obtener tu API Key de Roboflow

1. Entra a **https://app.roboflow.com/** e inicia sesión (o crea una cuenta gratuita).
2. Ve a **Settings → API Keys** del workspace (o directamente a `https://app.roboflow.com/settings/api`).
3. Copia tu **Private API Key** (formato `rf_xxxxxxxxxxxxxxxxxxxxxxx`).

Tres formas de pasársela al script:

```powershell
# Opción A: variable de entorno (recomendado)
$env:ROBOFLOW_API_KEY = "rf_tu_key_real"
python roboflow_emociones.py

# Opción B: archivo .env (requiere python-dotenv)
echo "ROBOFLOW_API_KEY=rf_tu_key_real" > .env
python roboflow_emociones.py

# Opción C: pasarla en cada ejecución
python roboflow_emociones.py --api-key rf_tu_key_real
```

> ⚠️ **Nunca** subas tu API Key a un repositorio público. El `.gitignore` incluido excluye `.env` por defecto.

El modelo por defecto es `human-face-emotions/28`, público en Roboflow Universe. Tu cuenta gratuita tiene cuota mensual suficiente para uso exploratorio.

---

## Uso

### Modo cámara (tiempo real)

```powershell
python roboflow_emociones.py
```

Ejemplos con parámetros:

```powershell
# Mayor resolución
python roboflow_emociones.py --width 1280 --height 720

# Más conservador con detecciones (menos falsos positivos)
python roboflow_emociones.py --conf 0.55

# Sin tracking ni smoothing (más reactivo, más ruidoso)
python roboflow_emociones.py --no-smooth --no-tracking

# Otra cámara (por ejemplo una USB externa)
python roboflow_emociones.py --camera 1
```

### Modo imagen única

Procesa una foto fija en vez de la webcam:

```powershell
python roboflow_emociones.py --image cara.jpg
```

Genera un `cara_anotada.jpg` con las cajas y etiquetas dibujadas, y muestra los resultados por consola.

### Controles en caliente

Mientras la ventana de la webcam está abierta:

| Tecla | Acción |
|---|---|
| `q` / `ESC` | Salir (cierra ventana y genera informes) |
| `s` | Guardar snapshot anotado en `snapshots/` |
| `espacio` | Pausar / reanudar inferencia (la cámara sigue visible) |
| `+` o `=` | Subir umbral de confianza en 0.05 |
| `-` o `_` | Bajar umbral de confianza en 0.05 |
| `m` | Alternar espejo (vista tipo selfie) |
| `t` | Activar / desactivar smoothing temporal |
| `i` | Activar / desactivar tracking con IDs |
| `h` | Mostrar / ocultar el HUD |

### Referencia de CLI completa

```
usage: roboflow_emociones.py [-h] [--image IMAGE] [--camera CAMERA]
                             [--width WIDTH] [--height HEIGHT] [--conf CONF]
                             [--model MODEL] [--api-url API_URL]
                             [--api-key API_KEY] [--jpeg-quality JPEG_QUALITY]
                             [--timeout TIMEOUT] [--no-mirror] [--no-smooth]
                             [--smooth-window SMOOTH_WINDOW]
                             [--smooth-iou SMOOTH_IOU] [--no-tracking]
                             [--track-iou TRACK_IOU]
                             [--track-max-missed TRACK_MAX_MISSED] [--no-log]
                             [--log-dir LOG_DIR] [--no-chart] [--no-report]
                             [--no-pdf] [--title TITLE] [--authors AUTHORS]
                             [--institution INSTITUTION] [--faculty FACULTY]
                             [--year YEAR] [--logo LOGO] [--verbose]
```

**Flags más importantes:**

| Flag | Default | Descripción |
|---|---|---|
| `--image PATH` | — | Procesa una imagen en vez de la cámara |
| `--camera N` | 0 | Índice de la cámara |
| `--width / --height` | 640 / 480 | Resolución solicitada |
| `--conf FLOAT` | 0.35 | Umbral de confianza inicial |
| `--model ID` | `human-face-emotions/28` | Modelo de Roboflow Universe |
| `--api-key KEY` | (env) | Sobrescribe `ROBOFLOW_API_KEY` |
| `--smooth-window N` | 7 | Ventana del suavizado temporal |
| `--smooth-iou T` | 0.4 | IoU mínimo para votar |
| `--track-iou T` | 0.3 | IoU mínimo para mantener track |
| `--track-max-missed N` | 15 | Frames antes de descartar un track |
| `--no-smooth / --no-tracking / --no-log / --no-pdf / --no-chart / --no-report` | — | Desactiva el componente correspondiente |
| `--title / --authors / --institution / --faculty / --year / --logo` | (defaults) | Personalización de la portada del PDF |
| `--verbose` | — | Logging detallado (DEBUG) |

---

## Cómo funciona

### Arquitectura general

```
   ┌──────────┐   frame    ┌──────────────────┐   frame.copy()   ┌─────────────────────┐
   │  Webcam  │──────────▶│  Hilo principal   │─────────────────▶│  InferenceWorker    │
   │ (OpenCV) │            │  (captura + UI)  │                  │  (hilo daemon)      │
   └──────────┘            │                  │◀─────────────────│  Roboflow Serverless│
                           └──────────────────┘   últimas dets   │  vía REST           │
                                    │                            └─────────────────────┘
                                    ▼
                  ┌──────────────────────────────────────┐
                  │  Smoother → Tracker → CSV + Render   │
                  └──────────────────────────────────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────────┐
                  │  Al cerrar: chart, snapshots, MD, PDF│
                  └──────────────────────────────────────┘
```

**Idea clave:** la captura nunca espera a la red. El `InferenceWorker` corre en un hilo aparte con política **"latest frame wins"**: si llega un frame nuevo mientras todavía se está procesando uno anterior, el viejo se descarta. Así la previsualización en pantalla nunca se congela aunque la API tarde 300 ms en responder.

### Pipeline de procesamiento

Cada vez que el worker entrega un resultado, las detecciones pasan por:

1. **`parse_predictions()`** — convierte la respuesta JSON de Roboflow en objetos `Detection` con bbox en píxeles (`x1, y1, x2, y2`).
2. **`DetectionSmoother`** — vota la clase más probable por solapamiento espacial con los últimos N frames (default 7). Reduce el parpadeo entre `happy` y `neutral`, por ejemplo.
3. **`FaceTracker`** — empareja cada detección con un track existente por IoU (greedy, ordenado por IoU descendente). Los tracks no emparejados envejecen y se descartan tras `max_missed` frames.
4. **`SessionLogger`** — escribe una fila al CSV con timestamp, persona, clase, confianza y bbox. Si la confianza es la máxima vista para esa clase, guarda un recorte del rostro.
5. **`draw_detections` + `draw_hud`** — renderizan bboxes con etiqueta `#ID emocion XX%` (color según emoción) y el panel de información encima.

### Componentes principales del código

Todo el código está en un único archivo: [`roboflow_emociones.py`](roboflow_emociones.py).

| Componente | Líneas aprox. | Responsabilidad |
|---|---|---|
| `Detection` (dataclass) | ~10 | Estructura inmutable de una detección |
| `InferenceResult` (dataclass) | ~5 | Resultado completo del worker (detecciones + raw + latencia) |
| `parse_predictions()` | ~50 | Normaliza la respuesta JSON al modelo interno |
| `color_for()` + paleta | ~20 | Color BGR estable por clase (con fallback determinista) |
| `_bbox_iou()` | ~15 | Intersección sobre unión entre dos bboxes |
| `DetectionSmoother` | ~50 | Voto ponderado por IoU en ventana deslizante |
| `FaceTracker` | ~70 | Greedy IoU matching con purga de tracks expirados |
| `SessionLogger` | ~600 | CSV + chart (matplotlib) + MD + PDF (reportlab) + snapshots |
| `RoboflowClient` | ~60 | Cliente HTTP con `Retry` y backoff exponencial |
| `InferenceWorker` (Thread) | ~75 | Hilo de inferencia con "latest frame" lock y detección de 401 |
| `draw_detections` / `draw_hud` | ~120 | Renderizado de bboxes y HUD con auto-wrap |
| `run_realtime()` | ~180 | Bucle principal: captura, renderizado, manejo de teclas |
| `infer_image()` | ~30 | Modo imagen única (sin cámara) |
| `parse_args()` + `main()` | ~80 | CLI con argparse |

---

## Salidas que genera el programa

Cada sesión genera un grupo de archivos identificados por timestamp en la carpeta `sessions/`:

### 1. CSV — `session_AAAAMMDD_HHMMSS.csv`

Una fila por detección, columnas:

```
timestamp_iso, elapsed_s, track_id, class, confidence, x1, y1, x2, y2
```

Útil para análisis estadístico posterior con pandas, R o Excel.

### 2. Gráfico PNG — `session_AAAAMMDD_HHMMSS.png`

Dos paneles:

- **Panel superior (Gantt)**: una franja horizontal por persona detectada. El color cambia conforme cambia la emoción dominante.
- **Panel inferior (barras)**: porcentaje de cada emoción sobre el total de detecciones.

### 3. Snapshots — `session_AAAAMMDD_HHMMSS_snapshots/`

Una imagen JPG por cada emoción detectada, capturada en el momento de **máxima confianza** del modelo, recortada al rostro con padding.

### 4. Informe Markdown — `session_AAAAMMDD_HHMMSS_informe.md`

Renderiza en VS Code, GitHub, GitLab, Obsidian, etc. Incluye:

- Metadatos de la sesión
- Tabla de distribución por emoción con barras unicode
- Gráfico embebido
- Sección por emoción con confianza pico, momento exacto y snapshot
- Notas metodológicas

### 5. Informe PDF — `session_AAAAMMDD_HHMMSS_informe.pdf`

PDF profesional generado con reportlab:

- **Portada** con logo institucional, título configurable, autores, institución, facultad y año
- Tipografía **Times New Roman 12pt** en todo el documento
- Banda superior verde (color institucional) en cada página
- Pie con `Sesión <id>` y número de página
- Secciones: Resumen ejecutivo · Cómo interpretar este informe · Análisis temporal (gráfico) · Análisis por emoción (con snapshots ampliados y estadísticas) · Notas metodológicas

---

## Estructura del proyecto

```
roboflow-emociones/
├─ roboflow_emociones.py          ← script principal (~2000 líneas, todo en un archivo)
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ assets/
│  └─ logo_udec.png               ← logo institucional para la portada del PDF
├─ notebooks/
│  └─ Roboflow emociones.ipynb    ← notebook original de Colab (referencia histórica)
├─ sessions/                       ← (auto-generada) outputs por sesión
└─ snapshots/                      ← (auto-generada) snapshots manuales con tecla 's'
```

---

## Configuración avanzada

### Variables de entorno reconocidas

| Variable | Equivalente CLI | Descripción |
|---|---|---|
| `ROBOFLOW_API_KEY` | `--api-key` | API Key de Roboflow (requerida) |
| `ROBOFLOW_MODEL_ID` | `--model` | ID del modelo a usar |
| `ROBOFLOW_API_URL` | `--api-url` | URL base del endpoint |
| `REPORT_TITLE` | `--title` | Título de la portada del PDF |
| `REPORT_AUTHORS` | `--authors` | Autores (separados por `;`) |
| `REPORT_INSTITUTION` | `--institution` | Institución |
| `REPORT_FACULTY` | `--faculty` | Facultad |
| `REPORT_YEAR` | `--year` | Año |
| `REPORT_LOGO` | `--logo` | Ruta al logo de la portada |

### Cambiar el modelo

Cualquier modelo público de Roboflow Universe que devuelva object detection funciona:

```powershell
python roboflow_emociones.py --model "tu-modelo/version"
```

Por ejemplo: `mask-wearing/4`, `face-detection-mik1i/22`, etc.

### Personalizar el informe

Defaults pensados para los autores de este proyecto (UdeC). Cambiables vía CLI o env:

```powershell
$env:REPORT_TITLE = "Análisis emocional - Caso de estudio 1"
$env:REPORT_INSTITUTION = "Mi Universidad"
$env:REPORT_AUTHORS = "Nombre Uno;Nombre Dos"
python roboflow_emociones.py
```

---

## Solución de problemas

| Síntoma | Diagnóstico / solución |
|---|---|
| `HTTP 401` repetido en consola | API Key inválida. El worker se detiene automáticamente y muestra instrucciones para corregirla. |
| `HTTP 429` repetido | Cuota mensual agotada. Pausa con `espacio` o sube `--smooth-window` para reducir llamadas. |
| Cámara negra o no se abre | Otra app la está usando, o el índice es otro. Prueba `--camera 1`. |
| FPS muy bajo | Reduce `--width/--height` o sube `--smooth-window`. |
| Cajas desplazadas respecto al rostro | Pulsa `m` para alternar el modo espejo. |
| Caras detectadas pero no aparecen cajas | Baja el umbral con `-` hasta ver detecciones. Algunas expresiones tienen baja confianza. |
| `ImportError: matplotlib` | El gráfico se omite, pero el resto funciona. Instala con `pip install matplotlib` si lo quieres. |
| `ImportError: reportlab` | Igual, el PDF se omite. Instala con `pip install reportlab`. |
| Error de UTF-8 en consola Windows | Ejecuta con `$env:PYTHONIOENCODING="utf-8"` antes del script. |

---

## Limitaciones conocidas

- El modelo detecta **expresión facial**, no estado emocional interno. Los resultados son un insumo complementario, no diagnóstico clínico.
- Los modelos de emociones tienen **sesgos** del dataset de entrenamiento: pueden fallar más en grupos sub-representados o con condiciones de iluminación pobres.
- **Diferencias culturales** en cómo se expresan las emociones no están capturadas.
- La inferencia depende de la red. Si Roboflow Serverless está caído o tienes mala conexión, la latencia sube y el FPS de inferencia baja.
- El uso de cuota de la API es **proporcional a los frames analizados**: una sesión de 30 segundos puede consumir 50–150 llamadas.

---

## Mejoras futuras

Funcionalidades identificadas pero no implementadas todavía:

- **Inferencia local con ONNX**: exportar el modelo de Roboflow a `.onnx` para correr offline a ~30 fps sin coste de API.
- **Workflows de Roboflow**: encadenar varios modelos (detección de cara → clasificación de emoción → blur de fondo).
- **GUI web con Streamlit o Gradio**: alternativa a la ventana de OpenCV.
- **Cámara IP / RTSP**: soportar `cv2.VideoCapture("rtsp://...")`.
- **Procesamiento de video grabado**: pasarle un `.mp4` y exportar el `.mp4` anotado.
- **Modo batch de imágenes**: procesar una carpeta entera.
- **Filtros tipo overlay emoji** sobre las caras según emoción detectada.

---

## Tecnologías

| Tecnología | Rol |
|---|---|
| [Python 3.10+](https://www.python.org/) | Lenguaje base |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Captura de cámara, dibujo, ventanas |
| [NumPy](https://numpy.org/) | Buffers de imagen |
| [Requests](https://requests.readthedocs.io/) + `urllib3.Retry` | Cliente HTTP con reintentos |
| [Roboflow Universe](https://universe.roboflow.com/) | Modelo público de detección de emociones |
| [Matplotlib](https://matplotlib.org/) | Gráfico Gantt + barras |
| [ReportLab](https://www.reportlab.com/) | Generación de PDF |
| [python-dotenv](https://github.com/theskumar/python-dotenv) *(opcional)* | Carga de credenciales desde `.env` |

---

## Autores

- **Cristian Camilo Posada García**
- **Luis Felipe Espinel Botina**
- **Michael Steven Naranjo Bautista**

Universidad de Cundinamarca — Facultad de Ingeniería · 2026

---

## Licencia

Este proyecto se distribuye bajo licencia **MIT**. Consulta [LICENSE](LICENSE) para más detalles.
