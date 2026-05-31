"""
streamlit_fitness_app.py
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
from pose_analyzer import PoseAnalyzer, SquatAnalyzer, PushupAnalyzer, DeadliftAnalyzer, FrontViewAnalyzer

# ====== CONFIG ======
st.set_page_config(
    page_title="AlinIA - Fitness Form Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .feedback-card {
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .badge {
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: bold;
        color: white;
        white-space: nowrap;
        min-width: 54px;
        text-align: center;
    }
    .badge-ok    { background: #2e7d32; }
    .badge-warn  { background: #e65100; }
    .badge-error { background: #b71c1c; }
    .badge-info  { background: #1565c0; }
    .card-ok     { background: #f1f8f1; border-left: 4px solid #4caf50; }
    .card-warn   { background: #fff8f0; border-left: 4px solid #ff6d00; }
    .card-error  { background: #fff5f5; border-left: 4px solid #e53935; }
    .card-info   { background: #f0f4ff; border-left: 4px solid #1976d2; }
    .card-msg    { font-size: 0.93em; color: #333; flex: 1; }
    .freq-tag    { font-size: 0.75em; color: #888; margin-left: auto; white-space: nowrap; }
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

with st.sidebar.expander("Guia de ejercicio"):
    if selected_exercise == "squat":
        st.markdown("""
        **Sentadilla Correcta**
        - Rodillas: 75-120 grados
        - Espalda: inclinacion moderada (+/-45 grados)
        - Rodillas alineadas: sin rotacion
        - Peso en talones
        """)
    elif selected_exercise == "pushup":
        st.markdown("""
        **Flexion Correcta**
        - Codo: 70-120 grados
        - Cuerpo recto en plank (alineacion > 155 grados)
        - Cabeza alineada con espalda
        - Manos a ancho de hombros
        """)
    else:
        st.markdown("""
        **Peso Muerto Correcto**
        - Espalda neutra (no redondeada) - critico
        - Bisagra de cadera: 60-170 grados
        - Rodilla: 20-80 grados en fase baja
        - Pecho elevado, peso cercano al cuerpo
        """)

# ====== TITULO ======
st.title("AlinIA - Fitness Form Analyzer")
st.markdown("Analiza tu forma en ejercicios y recibe feedback automatico")
st.warning("Esta herramienta es educativa. No reemplaza a un entrenador profesional.")

# ====== TABS ======
tab1, tab2, tab3 = st.tabs(["Procesar Video", "Procesar Imagen", "Rangos de Referencia"])

# ---------- TAB 1: VIDEO ----------
with tab1:
    st.header("Analizar Video")

    video_file = st.file_uploader(
        "Sube un video de tu ejercicio",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Maximo 200 MB. Filma de PERFIL (vista lateral) para mejor precision"
    )

    st.info("Graba el video de PERFIL (vista lateral). "
              "El cuerpo debe estar completamente visible de lado para medir "
              "angulos de rodilla, cadera y espalda con precision.")

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
            st.video(video_to_show)

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
    st.subheader("2. Vista de Frente")

    if selected_exercise == 'squat':
        front_hint_v = "Sube un video de frente para verificar ancho de pies, alineacion de rodillas y nivel de caderas."
    elif selected_exercise == 'pushup':
        front_hint_v = "Sube un video de frente para verificar ancho de manos y nivel de hombros."
    else:
        front_hint_v = "Sube un video de frente para verificar ancho de pies vs caderas y nivel de caderas."
    st.info(front_hint_v)

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
            st.video(video_front_show)
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
    st.subheader("1. Vista de Perfil (lateral)")
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
    st.subheader("2. Vista de Frente")

    if selected_exercise == 'squat':
        front_hint = "Permite verificar ancho de pies vs hombros y alineacion de rodillas."
    elif selected_exercise == 'pushup':
        front_hint = "Permite verificar ancho de manos vs hombros y nivel de hombros."
    else:
        front_hint = "Permite verificar ancho de pies vs caderas y nivel de caderas/hombros."
    st.info(front_hint)

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
