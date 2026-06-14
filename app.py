"""
app.py
App Streamlit para analisis de forma en ejercicios - AlinIA
"""

import streamlit as st
import cv2
import tempfile
import os
import subprocess
from collections import Counter
import re
import numpy as np
import plotly.graph_objects as go
from src.pose_analyzer import PoseAnalyzer, SquatAnalyzer, PushupAnalyzer, DeadliftAnalyzer, FrontViewAnalyzer

# ====== CONFIG ======
st.set_page_config(
    page_title="AlinIA - Fitness Form Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ===== BASE / FONDO ===== */
.stApp { background-color: #050C22 !important; }
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1A3F 0%, #111933 100%) !important;
    border-right: 1px solid #00A3CC44;
}
[data-testid="stSidebar"] * { color: #FAFAFA; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ===== TEXTO GLOBAL ===== */
p, li, span, label, .stMarkdown { color: #FAFAFA; }
h1 { color: #00FFFF !important; letter-spacing: 2px; }
h2 { color: #00FFFF !important; }
h3 { color: #00A3CC !important; }

/* ===== CABECERA LOGO ===== */
.alinia-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 1rem 0 0.5rem 0;
    margin-bottom: 0.5rem;
}
.alinia-logo {
    animation: logo-pulse 3s ease-in-out infinite;
}
@keyframes logo-pulse {
    0%, 100% { filter: drop-shadow(0 0 6px #00FFFF88); }
    50% { filter: drop-shadow(0 0 18px #00FFFFcc); }
}
.alinia-title {
    font-size: 3.2em;
    font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #00FFFF, #00A3CC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.alinia-subtitle {
    font-size: 0.95em;
    color: #B0B0B0;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: #111933;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #00A3CC33;
}
.stTabs [data-baseweb="tab"] {
    color: #B0B0B0 !important;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 8px 20px;
    transition: all 0.3s;
}
.stTabs [data-baseweb="tab"]:hover { color: #00FFFF !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00FFFF22, #00A3CC22) !important;
    color: #00FFFF !important;
    border-bottom: 2px solid #00FFFF !important;
}

/* ===== BOTONES ===== */
.stButton > button {
    background: linear-gradient(135deg, #00FFFF, #00A3CC) !important;
    color: #050C22 !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.5rem !important;
    letter-spacing: 0.5px;
    transition: all 0.3s !important;
    box-shadow: 0 0 12px #00FFFF44 !important;
}
.stButton > button:hover {
    box-shadow: 0 0 25px #00FFFFaa !important;
    transform: translateY(-2px) !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: #111933 !important;
    border: 2px dashed #00A3CC !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    animation: border-pulse 3s ease-in-out infinite;
}
@keyframes border-pulse {
    0%, 100% { border-color: #00A3CC; box-shadow: 0 0 6px #00A3CC33; }
    50% { border-color: #00FFFF; box-shadow: 0 0 16px #00FFFF44; }
}
[data-testid="stFileUploader"] * { color: #B0B0B0 !important; }

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: #111933 !important;
    border-left: 3px solid #00FFFF !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 0 10px #00FFFF22;
}
[data-testid="stMetricValue"] {
    color: #00FFFF !important;
    font-size: 1.8em !important;
    font-weight: 900 !important;
}
[data-testid="stMetricLabel"] { color: #B0B0B0 !important; }

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00A3CC, #00FFFF) !important;
    box-shadow: 0 0 8px #00FFFF66;
}

/* ===== SELECT / INPUT ===== */
.stSelectbox [data-baseweb="select"] > div,
[data-testid="stSelectbox"] > div > div {
    background: #111933 !important;
    border: 1px solid #00A3CC55 !important;
    color: #FAFAFA !important;
    border-radius: 8px !important;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: #111933 !important;
    color: #00A3CC !important;
    border-radius: 8px !important;
    border: 1px solid #00A3CC33 !important;
}
.streamlit-expanderContent {
    background: #0A1A3F !important;
    border: 1px solid #00A3CC22 !important;
    border-top: none !important;
}

/* ===== ALERTS / INFO / WARNING ===== */
.stAlert {
    background: #111933 !important;
    border-radius: 10px !important;
    border-left: 4px solid #00A3CC !important;
}

/* ===== SEPARADORES ===== */
hr { border-color: #00A3CC33 !important; }

/* ===== DIVIDER DE SECCIONES ===== */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00A3CC, transparent);
    margin: 2rem 0;
}

/* ===== FEEDBACK CARDS ===== */
.feedback-card {
    padding: 11px 16px;
    margin: 7px 0;
    border-radius: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
}
.feedback-card:hover { transform: translateX(4px); }
.badge {
    padding: 3px 12px;
    border-radius: 6px;
    font-size: 0.72em;
    font-weight: 800;
    color: #050C22;
    white-space: nowrap;
    min-width: 58px;
    text-align: center;
    letter-spacing: 0.5px;
}
.badge-ok    { background: #00FFFF; box-shadow: 0 0 8px #00FFFF66; }
.badge-warn  { background: #FFB300; box-shadow: 0 0 8px #FFB30066; }
.badge-error { background: #FF4444; box-shadow: 0 0 8px #FF444466; }
.badge-info  { background: #00A3CC; box-shadow: 0 0 8px #00A3CC66; }
.card-ok     { background: #0A2A1A; border-left: 4px solid #00FFFF; box-shadow: 0 0 8px #00FFFF11; }
.card-warn   { background: #2A1F0A; border-left: 4px solid #FFB300; box-shadow: 0 0 8px #FFB30011; }
.card-error  { background: #2A0A0A; border-left: 4px solid #FF4444; box-shadow: 0 0 8px #FF444411; }
.card-info   { background: #0A1A2A; border-left: 4px solid #00A3CC; box-shadow: 0 0 8px #00A3CC11; }
.card-msg    { font-size: 0.92em; color: #FAFAFA; flex: 1; }
.freq-tag    { font-size: 0.73em; color: #B0B0B0; margin-left: auto; white-space: nowrap;
               background: #050C22; padding: 2px 8px; border-radius: 4px; }

/* ===== SECCION HEADER ===== */
.view-header {
    background: linear-gradient(135deg, #111933, #0A1A3F);
    border: 1px solid #00A3CC44;
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.view-badge {
    background: linear-gradient(135deg, #00FFFF, #00A3CC);
    color: #050C22;
    font-weight: 900;
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 0.85em;
    letter-spacing: 1px;
    white-space: nowrap;
}
.view-desc { color: #B0B0B0; font-size: 0.88em; }

/* ===== VIDEO PLAYER ===== */
video { border-radius: 12px !important; box-shadow: 0 0 20px #00FFFF22 !important; }
img { border-radius: 12px !important; }

/* ===== DATAFRAME / TABLE ===== */
.stDataFrame { border: 1px solid #00A3CC33 !important; border-radius: 10px !important; }

/* ===== SIDEBAR SELECT ===== */
[data-testid="stSidebar"] .stSelectbox label { color: #00A3CC !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def feedback_cards(feedback_list):
    """Renderiza lista de feedback como tarjetas con badge de color."""
    for fb in feedback_list:
        if "[OK]" in fb:
            msg   = fb.replace("[OK]", "").strip()
            card  = "card-ok"
            badge = "badge-ok"
            label = "OK"
        elif "[WARN]" in fb:
            msg   = fb.replace("[WARN]", "").strip()
            card  = "card-warn"
            badge = "badge-warn"
            label = "AVISO"
        elif "[ERROR]" in fb:
            msg   = fb.replace("[ERROR]", "").strip()
            card  = "card-error"
            badge = "badge-error"
            label = "ERROR"
        else:
            msg   = fb
            card  = "card-info"
            badge = "badge-info"
            label = "INFO"

        st.markdown(f"""
        <div class="feedback-card {card}">
            <span class="badge {badge}">{label}</span>
            <span class="card-msg">{msg}</span>
        </div>
        """, unsafe_allow_html=True)


def feedback_cards_with_freq(feedback_counter, total_frames):
    """
    Renderiza feedback consolidado por frecuencia.
    Solo muestra mensajes que aparecieron en > 15% de los frames.
    Agrega etiqueta de frecuencia (constante / frecuente / ocasional).
    """
    if total_frames == 0:
        return

    # Ordenar: errores primero, luego warns, luego oks
    def sort_key(item):
        msg, count = item
        if "[ERROR]" in msg:
            return (0, -count)
        elif "[WARN]" in msg:
            return (1, -count)
        elif "[OK]" in msg:
            return (2, -count)
        return (3, -count)

    sorted_items = sorted(feedback_counter.items(), key=sort_key)

    shown = 0
    for raw_msg, count in sorted_items:
        pct = count / total_frames * 100

        # Ignorar mensajes que aparecen en menos del 15% de los frames
        if pct < 15:
            continue

        # Etiqueta de frecuencia
        if pct >= 70:
            freq_label = "constante"
        elif pct >= 40:
            freq_label = "frecuente"
        else:
            freq_label = "ocasional"

        if "[OK]" in raw_msg:
            msg   = raw_msg.replace("[OK]", "").strip()
            card  = "card-ok"
            badge = "badge-ok"
            label = "OK"
        elif "[WARN]" in raw_msg:
            msg   = raw_msg.replace("[WARN]", "").strip()
            card  = "card-warn"
            badge = "badge-warn"
            label = "AVISO"
        elif "[ERROR]" in raw_msg:
            msg   = raw_msg.replace("[ERROR]", "").strip()
            card  = "card-error"
            badge = "badge-error"
            label = "ERROR"
        else:
            msg   = raw_msg
            card  = "card-info"
            badge = "badge-info"
            label = "INFO"

        st.markdown(f"""
        <div class="feedback-card {card}">
            <span class="badge {badge}">{label}</span>
            <span class="card-msg">{msg}</span>
            <span class="freq-tag">{freq_label} ({pct:.0f}%)</span>
        </div>
        """, unsafe_allow_html=True)
        shown += 1

    if shown == 0:
        st.info("No hay suficientes datos de feedback para este video.")


def score_gauge(score):
    """Muestra el score como barra de progreso con color dinamico."""
    color = "#4caf50" if score >= 80 else "#ff9800" if score >= 60 else "#e53935"
    st.markdown(f"""
    <div style="margin: 10px 0 4px 0; font-size: 0.85em; color: #555;">Score de forma</div>
    <div style="background:#e0e0e0; border-radius: 8px; height: 22px; width: 100%;">
        <div style="background:{color}; width:{score}%; height: 100%;
                    border-radius: 8px; display: flex; align-items: center; padding-left: 10px;">
            <span style="color:white; font-weight:bold; font-size:0.85em;">{score:.0f}/100</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def reencode_for_browser(input_path, output_path):
    """
    Intenta crear un video H.264 reproducible en navegador.
    Metodo 1: ffmpeg en PATH
    Metodo 2: imageio con plugin ffmpeg
    Retorna el path del video que se puede reproducir.
    """
    # Metodo 1: ffmpeg en PATH (Windows/Mac/Linux)
    for cmd in ['ffmpeg', 'ffmpeg.exe']:
        try:
            result = subprocess.run([
                cmd, '-y', '-i', input_path,
                '-vcodec', 'libx264', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ], capture_output=True, timeout=180)
            if result.returncode == 0 and os.path.getsize(output_path) > 1000:
                return output_path
        except (FileNotFoundError, Exception):
            continue

    # Metodo 2: imageio con ffmpeg plugin
    try:
        import imageio
        reader = imageio.get_reader(input_path)
        meta   = reader.get_meta_data()
        fps_v  = meta.get('fps', 30)
        writer = imageio.get_writer(
            output_path, fps=fps_v,
            codec='libx264', quality=7,
            pixelformat='yuv420p'
        )
        for frame in reader:
            writer.append_data(frame)
        writer.close()
        reader.close()
        if os.path.getsize(output_path) > 1000:
            return output_path
    except Exception:
        pass

    # Sin conversion disponible: devolver video original
    return input_path


# ====== SIDEBAR ======
st.sidebar.title("Configuracion")

exercise_type = st.sidebar.selectbox(
    "Selecciona ejercicio:",
    ["Sentadilla (Squat)", "Flexion (Push-up)", "Peso Muerto (Deadlift)"],
    index=0
)

exercise_map = {
    "Sentadilla (Squat)":     "squat",
    "Flexion (Push-up)":      "pushup",
    "Peso Muerto (Deadlift)": "deadlift"
}
selected_exercise = exercise_map[exercise_type]

# ── Recuadro lateral con guia del ejercicio ──
_guide_data = {
    "squat": {
        "titulo": "Sentadilla",
        "icono": "SQUAT",
        "items": [
            ("Rodilla", "75 - 120 grados"),
            ("Espalda", "+/-45 grados"),
            ("Pies", "Ancho de hombros"),
            ("Rodillas", "Alineadas con pie"),
        ],
        "alerta": "Riesgo: valgus de rodilla",
        "vistas": "Perfil + Frente",
    },
    "pushup": {
        "titulo": "Flexion",
        "icono": "PUSH-UP",
        "items": [
            ("Codo", "70 - 120 grados"),
            ("Alineacion", "> 155 grados"),
            ("Manos", "Ancho de hombros"),
            ("Hombros", "Nivelados"),
        ],
        "alerta": "Riesgo: cadera hundida",
        "vistas": "Perfil + Frente",
    },
    "deadlift": {
        "titulo": "Peso Muerto",
        "icono": "DEADLIFT",
        "items": [
            ("Bisagra cadera", "60 - 170 grados"),
            ("Rodilla (baja)", "20 - 80 grados"),
            ("Espalda", "Neutra — critico"),
            ("Pies", "Ancho de caderas"),
        ],
        "alerta": "Riesgo: espalda redondeada",
        "vistas": "Perfil + Frente",
    },
}

_g = _guide_data[selected_exercise]
_items_html = "".join(
    f'''<div style="display:flex;justify-content:space-between;padding:5px 0;
        border-bottom:1px solid #00A3CC22;">
        <span style="color:#B0B0B0;font-size:0.82em;">{k}</span>
        <span style="color:#00FFFF;font-size:0.82em;font-weight:700;">{v}</span>
    </div>'''
    for k, v in _g["items"]
)

st.sidebar.markdown(f"""
<div style="background:linear-gradient(135deg,#0A1A3F,#111933);
            border:1px solid #00A3CC55; border-radius:14px;
            padding:1rem 1.1rem 1rem 1.1rem; margin-top:0.8rem;">

  <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.8rem;">
    <div style="background:linear-gradient(135deg,#00FFFF,#00A3CC);
                color:#050C22;font-weight:900;font-size:0.7em;
                padding:3px 10px;border-radius:6px;letter-spacing:1px;">
      {_g["icono"]}
    </div>
    <span style="color:#FAFAFA;font-weight:700;font-size:0.95em;">{_g["titulo"]}</span>
  </div>

  {_items_html}

  <div style="margin-top:0.8rem;background:#2A0A0A;border-left:3px solid #FF4444;
              border-radius:6px;padding:6px 10px;font-size:0.78em;color:#FF8888;">
    {_g["alerta"]}
  </div>

  <div style="margin-top:0.6rem;background:#0A1A2A;border-left:3px solid #00A3CC;
              border-radius:6px;padding:5px 10px;font-size:0.77em;color:#B0B0B0;">
    Filmar: <span style="color:#00FFFF;font-weight:700;">{_g["vistas"]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ====== TITULO ======
# ===== HEADER CON LOGO =====
import base64
from pathlib import Path

_logo_path = Path(__file__).parent / "assets" / "logo.png"

if _logo_path.exists():
    # Logo real del proyecto
    with open(_logo_path, "rb") as _lf:
        _logo_b64 = base64.b64encode(_lf.read()).decode()
    st.markdown(f"""
    <div class="alinia-header">
      <div class="alinia-logo">
        <img src="data:image/png;base64,{_logo_b64}" width="170" height="170"
             style="object-fit:contain; filter:drop-shadow(0 0 10px #00FFFF88);">
      </div>
      <div>
        <div class="alinia-title">AlinIA</div>
        <div class="alinia-subtitle">Tecnologia de Alineacion Inteligente</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback: SVG inline mientras no existe logo.png
    st.markdown("""
    <div class="alinia-header">
      <div class="alinia-logo">
        <svg viewBox="0 0 120 115" width="90" height="90" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glow-h"><feGaussianBlur stdDeviation="2.5" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#00FFFF"/><stop offset="100%" stop-color="#00A3CC"/>
            </linearGradient>
          </defs>
          <line x1="8" y1="72" x2="112" y2="72" stroke="#00A3CC" stroke-width="3.5" filter="url(#glow-h)"/>
          <rect x="5" y="62" width="11" height="20" rx="2" fill="none" stroke="#00FFFF" stroke-width="2" filter="url(#glow-h)"/>
          <rect x="104" y="62" width="11" height="20" rx="2" fill="none" stroke="#00FFFF" stroke-width="2" filter="url(#glow-h)"/>
          <circle cx="82" cy="22" r="9" fill="none" stroke="#00FFFF" stroke-width="2.2" filter="url(#glow-h)"/>
          <line x1="80" y1="31" x2="46" y2="58" stroke="url(#g1)" stroke-width="3" stroke-linecap="round" filter="url(#glow-h)"/>
          <line x1="76" y1="40" x2="56" y2="72" stroke="#00FFFF" stroke-width="2" filter="url(#glow-h)"/>
          <circle cx="46" cy="58" r="3.5" fill="#00FFFF" filter="url(#glow-h)"/>
          <line x1="46" y1="58" x2="34" y2="84" stroke="url(#g1)" stroke-width="2.8" stroke-linecap="round" filter="url(#glow-h)"/>
          <line x1="34" y1="84" x2="28" y2="106" stroke="#00FFFF" stroke-width="2.5" filter="url(#glow-h)"/>
          <line x1="46" y1="58" x2="60" y2="84" stroke="url(#g1)" stroke-width="2.8" stroke-linecap="round" filter="url(#glow-h)"/>
          <line x1="60" y1="84" x2="62" y2="106" stroke="#00FFFF" stroke-width="2.5" filter="url(#glow-h)"/>
          <circle cx="34" cy="84" r="3" fill="#00A3CC" filter="url(#glow-h)"/>
          <circle cx="60" cy="84" r="3" fill="#00A3CC" filter="url(#glow-h)"/>
          <line x1="28" y1="106" x2="16" y2="106" stroke="#00FFFF" stroke-width="2.5" stroke-linecap="round" filter="url(#glow-h)"/>
          <line x1="62" y1="106" x2="76" y2="106" stroke="#00FFFF" stroke-width="2.5" stroke-linecap="round" filter="url(#glow-h)"/>
        </svg>
      </div>
      <div>
        <div class="alinia-title">AlinIA</div>
        <div class="alinia-subtitle">Tecnologia de Alineacion Inteligente</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""<div style="height:1px; background:linear-gradient(90deg,transparent,#00A3CC,transparent); margin-bottom:1.2rem;"></div>""",
            unsafe_allow_html=True)

st.markdown("""
<div style="background:#111933; border:1px solid #00A3CC33; border-radius:12px;
            padding:10px 18px; margin-bottom:1rem; color:#B0B0B0; font-size:0.88em;
            display:flex; align-items:center; gap:10px;">
  <span style="color:#FFB300; font-size:1.1em; font-weight:bold;">!</span>
  Herramienta educativa — no reemplaza la consulta con un entrenador profesional.
</div>
""", unsafe_allow_html=True)

# ====== TABS ======
tab1, tab2, tab3 = st.tabs(["Procesar Video", "Procesar Imagen", "Rangos de Referencia"])

# ---------- TAB 1: VIDEO ----------
with tab1:
    st.header("Analizar Video")
    st.markdown("Sube **dos videos**: uno lateral y otro de frente para un analisis completo.")

    # ---- SECCION 1: LATERAL ----
    st.markdown('''<div class="view-header">
      <span class="view-badge">LATERAL</span>
      <span class="view-desc">Mide angulos de rodilla, cadera y espalda con precision</span>
    </div>''', unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Video de LATERAL (perfil)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Filma desde el lado — cuerpo completo visible",
        key="video_lateral"
    )

    if video_file is not None:
        temp_dir       = tempfile.mkdtemp()
        temp_input     = os.path.join(temp_dir, "input_video.mp4")
        temp_raw       = os.path.join(temp_dir, "output_raw.mp4")   # mp4v (analisis)
        temp_output    = os.path.join(temp_dir, "output_video.mp4") # h264 (reproduccion)

        with open(temp_input, "wb") as f:
            f.write(video_file.getbuffer())

        status_box = st.info("Procesando video...")

        cap          = cv2.VideoCapture(temp_input)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # H.264 requiere dimensiones pares
        width        = width if width % 2 == 0 else width - 1
        height       = height if height % 2 == 0 else height - 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if selected_exercise == 'squat':
            exercise_analyzer = SquatAnalyzer()
        elif selected_exercise == 'pushup':
            exercise_analyzer = PushupAnalyzer()
        else:
            exercise_analyzer = DeadliftAnalyzer()

        analyzer = exercise_analyzer.analyzer

        # Intentar H.264 nativo (avc1) — funciona si OpenCV lo soporta
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out    = cv2.VideoWriter(temp_raw, fourcc, fps, (width, height))
        if not out.isOpened():
            # Fallback a mp4v (requiere re-codificacion posterior)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out    = cv2.VideoWriter(temp_raw, fourcc, fps, (width, height))

        frame_count       = 0
        scores            = []
        feedback_counter  = Counter()   # cuenta cuantos frames tuvo cada mensaje
        frames_with_pose  = 0

        progress_bar = st.progress(0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                results = analyzer.detect_pose(frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    analysis  = exercise_analyzer.analyze(frame, landmarks)

                    frame = analyzer.draw_skeleton(frame, landmarks)

                    score_val   = analysis['overall_score']
                    score_text  = f"Score: {score_val:.0f}/100"
                    score_color = (
                        (0, 200, 0) if score_val >= 80
                        else (0, 165, 255) if score_val >= 60
                        else (0, 0, 220)
                    )
                    cv2.putText(frame, score_text, (width - 260, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, score_color, 3)

                    scores.append(score_val)
                    frames_with_pose += 1
                    for fb in analysis['feedback']:
                        # Normalizar: quitar angulos especificos para agrupar mensajes similares
                        key = re.sub(r'\s*\(-?[\d.]+\s+grados[^)]*\)', '', fb).strip()
                        key = re.sub(r'\s*\(diff:[^)]*\)', '', key).strip()
                        feedback_counter[key] += 1
                else:
                    cv2.putText(frame, "Esperando pose...", (50, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

                out.write(frame)
                if total_frames > 0:
                    progress_bar.progress(frame_count / total_frames)

        finally:
            cap.release()
            out.release()

        progress_bar.empty()

        # Re-codificar a H.264 para reproduccion en el navegador
        status_box.info("Optimizando video para reproduccion...")
        video_to_show = reencode_for_browser(temp_raw, temp_output)

        status_box.success("Video procesado")

        # ----- Resultados -----
        col_video, col_results = st.columns([3, 2])

        with col_video:
            st.subheader("Video con skeleton")
            if os.path.exists(video_to_show) and os.path.getsize(video_to_show) > 1000:
                with open(video_to_show, 'rb') as _vf:
                    st.video(_vf.read())
            else:
                st.error(f"Error al generar video. Path: {video_to_show} | Existe: {os.path.exists(video_to_show)} | Tamaño: {os.path.getsize(video_to_show) if os.path.exists(video_to_show) else 'N/A'}")

        with col_results:
            avg_score = float(np.mean(scores)) if scores else 0
            score_gauge(avg_score)

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Maximo",  f"{max(scores) if scores else 0:.0f}/100")
            with c2:
                st.metric("Minimo", f"{min(scores) if scores else 0:.0f}/100")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Feedback de tu ejercicio**")
            st.caption(f"Basado en {frames_with_pose} frames con pose detectada")
            feedback_cards_with_freq(feedback_counter, frames_with_pose)

        # Grafico debajo
        st.subheader("Evolucion del Score")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=scores, mode='lines', name='Score',
            line=dict(color='#378ADD', width=2),
            fill='tozeroy', fillcolor='rgba(55,138,221,0.1)'
        ))
        fig.add_hline(y=80, line_dash="dot", line_color="green",
                      annotation_text="Buena forma",
                      annotation_position="bottom right")
        fig.add_hline(y=60, line_dash="dot", line_color="orange",
                      annotation_text="Mejorable",
                      annotation_position="bottom right")
        fig.update_layout(
            xaxis_title="Frame", yaxis_title="Score",
            yaxis=dict(range=[0, 105]),
            hovermode='x unified', height=300, margin=dict(t=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- VIDEO FRENTE ----
    st.markdown("---")
    front_hints_v = {
        'squat':    "Verifica ancho de pies, alineacion de rodillas y nivel de caderas",
        'pushup':   "Verifica ancho de manos y nivel de hombros",
        'deadlift': "Verifica ancho de pies vs caderas y nivel durante el levantamiento",
    }
    st.markdown(f'''<div class="view-header"><span class="view-badge">FRENTE</span>
     <span class="view-desc">{front_hints_v.get(selected_exercise, "")}</span></div>''', unsafe_allow_html=True)

    video_file_front = st.file_uploader(
        "Video de FRENTE",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="De frente a la camara, cuerpo completo visible",
        key="video_frente"
    )

    if video_file_front is not None:
        temp_dir_f   = tempfile.mkdtemp()
        temp_input_f = os.path.join(temp_dir_f, "input_front.mp4")
        temp_raw_f   = os.path.join(temp_dir_f, "output_front_raw.mp4")
        temp_out_f   = os.path.join(temp_dir_f, "output_front.mp4")

        with open(temp_input_f, "wb") as fv:
            fv.write(video_file_front.getbuffer())

        cap_f        = cv2.VideoCapture(temp_input_f)
        fps_f        = cap_f.get(cv2.CAP_PROP_FPS) or 30
        width_f      = int(cap_f.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_f     = int(cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # H.264 requiere dimensiones pares
        width_f      = width_f if width_f % 2 == 0 else width_f - 1
        height_f     = height_f if height_f % 2 == 0 else height_f - 1
        total_f      = int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT))

        front_analyzer = FrontViewAnalyzer()
        fa_core        = front_analyzer.analyzer

        fourcc_f = cv2.VideoWriter_fourcc(*'avc1')
        out_f    = cv2.VideoWriter(temp_raw_f, fourcc_f, fps_f, (width_f, height_f))
        if not out_f.isOpened():
            fourcc_f = cv2.VideoWriter_fourcc(*'mp4v')
            out_f    = cv2.VideoWriter(temp_raw_f, fourcc_f, fps_f, (width_f, height_f))

        fb_counter_f = Counter()
        frames_f     = 0
        prog_f       = st.progress(0)

        try:
            while cap_f.isOpened():
                ret_f, frm_f = cap_f.read()
                if not ret_f:
                    break
                frames_f += 1
                res_f = fa_core.detect_pose(frm_f)
                if res_f.pose_landmarks:
                    lm_f     = res_f.pose_landmarks.landmark
                    ana_f    = front_analyzer.analyze(frm_f, lm_f, selected_exercise)
                    frm_f    = fa_core.draw_skeleton(frm_f, lm_f)
                    for fb in ana_f['feedback']:
                        key_f = re.sub(r'\s*\(-?[\d.]+x[^)]*\)', '', fb).strip()
                        fb_counter_f[key_f] += 1
                out_f.write(frm_f)
                if total_f > 0:
                    prog_f.progress(frames_f / total_f)
        finally:
            cap_f.release()
            out_f.release()

        prog_f.empty()

        video_front_show = reencode_for_browser(temp_raw_f, temp_out_f)

        col_fv, col_fr = st.columns([3, 2])
        with col_fv:
            st.subheader("Video de frente con skeleton")
            with open(video_front_show, 'rb') as _vf:
                st.video(_vf.read())
        with col_fr:
            st.markdown("**Feedback de frente**")
            st.caption(f"Basado en {frames_f} frames")
            feedback_cards_with_freq(fb_counter_f, frames_f)

# ---------- TAB 2: IMAGEN ----------
with tab2:
    st.header("Analizar Imagen Estatica")
    st.markdown("Sube **dos imagenes**: una de perfil y otra de frente para un analisis completo.")

    # ---- SECCION 1: PERFIL ----
    st.markdown("---")
    st.markdown('''<div class="view-header"><span class="view-badge">PERFIL</span>
     <span class="view-desc">Mide angulos de rodilla, cadera y espalda con precision</span></div>''', unsafe_allow_html=True)
    st.info("Filmado desde el lado — permite medir angulos de rodilla, cadera y espalda.")

    image_file = st.file_uploader(
        "Imagen de PERFIL",
        type=['jpg', 'jpeg', 'png'],
        help="JPG o PNG. Cuerpo completo visible de lado",
        key="img_perfil"
    )

    if image_file is not None:
        temp_dir   = tempfile.mkdtemp()
        temp_image = os.path.join(temp_dir, "image_perfil.jpg")

        with open(temp_image, "wb") as f:
            f.write(image_file.getbuffer())

        frame = cv2.imread(temp_image)

        if selected_exercise == 'squat':
            exercise_analyzer = SquatAnalyzer()
        elif selected_exercise == 'pushup':
            exercise_analyzer = PushupAnalyzer()
        else:
            exercise_analyzer = DeadliftAnalyzer()

        analyzer = exercise_analyzer.analyzer
        results  = analyzer.detect_pose(frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            analysis  = exercise_analyzer.analyze(frame, landmarks)
            frame     = analyzer.draw_skeleton(frame, landmarks)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col2:
                score_gauge(analysis['overall_score'])
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Feedback de perfil**")
                feedback_cards(analysis['feedback'])
        else:
            st.error("No se pudo detectar pose. Asegurate de estar de perfil con cuerpo completo visible.")

    # ---- SECCION 2: FRENTE ----
    st.markdown("---")
    front_hints = {
        'squat':    "Verifica ancho de pies vs hombros y alineacion de rodillas",
        'pushup':   "Verifica ancho de manos vs hombros y nivel de hombros",
        'deadlift': "Verifica ancho de pies vs caderas y nivel de caderas/hombros",
    }
    st.markdown(f'''<div class="view-header"><span class="view-badge">FRENTE</span>
     <span class="view-desc">{front_hints.get(selected_exercise, "")}</span></div>''', unsafe_allow_html=True)

    image_file_front = st.file_uploader(
        "Imagen de FRENTE",
        type=['jpg', 'jpeg', 'png'],
        help="JPG o PNG. De frente a la camara, cuerpo completo visible",
        key="img_frente"
    )

    if image_file_front is not None:
        temp_dir2   = tempfile.mkdtemp()
        temp_image2 = os.path.join(temp_dir2, "image_frente.jpg")

        with open(temp_image2, "wb") as f:
            f.write(image_file_front.getbuffer())

        frame2 = cv2.imread(temp_image2)
        front_analyzer = FrontViewAnalyzer()
        fa_core        = front_analyzer.analyzer
        results2       = fa_core.detect_pose(frame2)

        if results2.pose_landmarks:
            landmarks2 = results2.pose_landmarks.landmark
            analysis2  = front_analyzer.analyze(frame2, landmarks2, selected_exercise)
            frame2     = fa_core.draw_skeleton(frame2, landmarks2)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col2:
                score_gauge(analysis2['overall_score'])
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Feedback de frente**")
                feedback_cards(analysis2['feedback'])
        else:
            st.error("No se pudo detectar pose. Asegurate de estar de frente con cuerpo completo visible.")


# ---------- TAB 3: RANGOS ----------
with tab3:
    st.header("Rangos Biomecanicos de Referencia")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Sentadilla")
        st.markdown("""
        | Metrica | Rango correcto |
        |---|---|
        | Rodilla | 75-120 grados |
        | Espalda (inclinacion) | +/-45 grados |
        | Diferencia rodillas | < 10 grados |
        """)
        st.info("Riesgo principal: valgus de rodilla (rodillas hacia adentro)")

    with col2:
        st.subheader("Flexion")
        st.markdown("""
        | Metrica | Rango correcto |
        |---|---|
        | Codo | 70-120 grados |
        | Alineacion cuerpo | > 155 grados |
        """)
        st.info("Riesgo principal: cadera hundida o elevada (falta de core)")

    with col3:
        st.subheader("Peso Muerto")
        st.markdown("""
        | Metrica | Rango correcto |
        |---|---|
        | Bisagra cadera | 60-170 grados |
        | Rodilla (fase baja) | 20-80 grados |
        | Espalda | Neutra (no redondeada) |
        """)
        st.error("Riesgo principal: espalda redondeada (hernia discal)")

    st.markdown("---")
    st.markdown("""
    **Tecnologia utilizada**
    - MediaPipe Pose: deteccion de 33 puntos corporales en tiempo real
    - OpenCV: procesamiento de imagen y video frame a frame
    - NumPy: calculo de angulos (producto punto, arccos)
    - Streamlit + Plotly: interfaz web y graficos interactivos
    """)

# ====== FOOTER ======
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Powered by MediaPipe Pose")
with col2:
    st.caption("AlinIA - Analisis biomecanico automatico")
with col3:
    st.caption("Los datos no se almacenan")
