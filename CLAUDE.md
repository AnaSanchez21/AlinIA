# AlinIA

App de IA que detecta postura corporal en ejercicios (sentadillas, flexiones, peso muerto) y da feedback automático sobre forma física, usando MediaPipe Pose + análisis biomecánico.

> Contexto completo del proyecto, decisiones de arquitectura y deuda técnica: ver `README.md`.

## Stack
- Python 3.8+ · Streamlit (app web)
- MediaPipe Pose (33 puntos corporales) · OpenCV (cv2)
- NumPy (geometría/ángulos) · Plotly (gráficos) · Pillow (UTF-8 para tildes)
- scipy, scikit-learn
- Sin base de datos (análisis frame-by-frame)

## Estructura
- `pose_analyzer.py` — CORE: detección, cálculo de ángulos, validación. Clases: `PoseAnalyzer`, `SquatAnalyzer`, `PushupAnalyzer`, `DeadliftAnalyzer`
- `process_video.py` — CLI: `process_exercise_video()` (frame-by-frame), `process_image()` (estático)
- `streamlit_fitness_app.py` — App web principal (tabs Video/Imagen/Ejemplos)
- `test_pose_image.py`, `test_angles.py`, `test_squat_analyzer.py` — tests manuales
- `examples/sample_videos/` — videos de demo

## Comandos
```bash
# Setup (una vez)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# App web (con venv activo)
streamlit run streamlit_fitness_app.py   # http://localhost:8501

# Procesar video
python process_video.py input_video.mp4 -o output_video.mp4 -e squat   # -e squat|pushup|deadlift

# Tests manuales (output: terminal + imagen resultado)
python test_pose_image.py        # detección de pose
python test_angles.py            # cálculo de ángulos
python test_squat_analyzer.py    # validación + feedback + score

# Verificar instalación
python -c "import mediapipe; import cv2; import streamlit; print('OK')"
```

## Convenciones
- **Estilo:** PEP 8, líneas máx. 100 caracteres, UTF-8
- **Comentarios:** español; **código:** inglés. Docstrings: qué / parámetros / retorno
- **Nombres:** funciones y variables `snake_case`, clases `PascalCase`, constantes `UPPER_SNAKE_CASE`
- **Feedback:** formato `[✅/⚠️/❌] Descripción (valor°) [Acción]`
- **Scores:** empiezan en 100, restan por errores, nunca negativos
- **Commits:** mensajes descriptivos en español (`Feat:`, `Fix:`, `Docs:`)

## Ranges biomecánicos (fuente de verdad)
```python
# SENTADILLA
KNEE_ANGLE_RANGE = (75, 120)    # flexión correcta
BACK_ANGLE_RANGE = (-45, 45)    # 0-35°=OK, 35-50°=warning, >50°=peligro

# FLEXIÓN
ELBOW_ANGLE_RANGE = (70, 120)

# PESO MUERTO
BACK_ANGLE_RANGE = (-10, 10)    # CRÍTICO: espalda neutra
```
> Pendiente de definir como constante: rango de espalda en flexión y rango de rodilla en peso muerto (aparecían en la tabla del README pero sin valor en código). No inventar valores: confirmar antes de usarlos.

## Reglas importantes
- **NUNCA modificar `mediapipe.Pose()`** — ya está optimizado (lite/full/heavy)
- **Landmarks:** verificar siempre `.visibility > 0.5` antes de usar (0.3-0.5 con cuidado, <0.3 ignorar)
- **Ángulos:** usar `np.clip(cos_angle, -1, 1)` antes de `arccos` para evitar errores numéricos
- **Puntos:** convertir a `(x, y)` para `calculate_angle()`; ignorar `z` si no se necesita
- **Rutas:** usar `pathlib.Path()` para compatibilidad cross-platform
- **Caracteres especiales:** usar UTF-8 con Pillow para tildes
- **Antes de dar por terminada una tarea:** correr el test correspondiente y verificar que el feedback tiene sentido y el score es justo (90+ con buena forma, 50- con mala forma)

## Flujo del análisis
```
INPUT (video/imagen)
  → [MediaPipe] detecta 33 puntos (x, y, z, visibilidad)
  → [Extracción] puntos clave (rodilla, cadera, hombro…)
  → [Cálculo] calculate_angle(p1, p2, p3) → grados
  → [Validación] compara vs ANGLE_RANGE → feedback
  → [Scoring] acumula penalizaciones → score/100
  → [Visualización] OpenCV dibuja skeleton + ángulos + feedback
  → OUTPUT (imagen/video procesado + métricas)
```

## Error común: producto punto
```python
# CORRECTO — clip obligatorio
cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
cos_angle = np.clip(cos_angle, -1, 1)   # evita cos>1 o cos<-1 por error numérico
angle_rad = acos(cos_angle)
```
