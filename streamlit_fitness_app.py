"""
streamlit_fitness_app.py
App Streamlit para analisis de forma en ejercicios
"""

import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import plotly.graph_objects as go
from pose_analyzer import PoseAnalyzer, SquatAnalyzer, PushupAnalyzer, DeadliftAnalyzer

# ====== CONFIG ======
st.set_page_config(
    page_title="AlinIA - Fitness Form Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #378ADD;
        background: linear-gradient(135deg, #E6F1FB 0%, #F5F8FC 100%);
        margin: 1rem 0;
    }
    .feedback-good    { color: #00AA00; font-weight: bold; }
    .feedback-warning { color: #FF9900; font-weight: bold; }
    .feedback-bad     { color: #FF0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ====== SIDEBAR ======
st.sidebar.title("Configuracion")

exercise_type = st.sidebar.selectbox(
    "Selecciona ejercicio:",
    ["Sentadilla (Squat)", "Flexion (Push-up)", "Peso Muerto (Deadlift)"],
    index=0
)

exercise_map = {
    "Sentadilla (Squat)": "squat",
    "Flexion (Push-up)":  "pushup",
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

st.warning(
    "[WARN] Esta herramienta es para propositos educativos/informativos. "
    "No reemplaza consulta con entrenador profesional."
)

# ====== TABS ======
tab1, tab2, tab3 = st.tabs(["Procesar Video", "Procesar Imagen", "Ejemplos y Rangos"])

# ---------- TAB 1: VIDEO ----------
with tab1:
    st.header("Analizar Video")

    video_file = st.file_uploader(
        "Sube un video de tu ejercicio",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Maximo 200 MB. Asegurate de que te veas completo en el frame"
    )

    if video_file is not None:
        temp_dir    = tempfile.mkdtemp()
        temp_input  = os.path.join(temp_dir, "input_video.mp4")
        temp_output = os.path.join(temp_dir, "output_video.mp4")

        with open(temp_input, "wb") as f:
            f.write(video_file.getbuffer())

        st.info("Procesando video... Esto puede tomar un momento")

        cap          = cv2.VideoCapture(temp_input)
        fps          = cap.get(cv2.CAP_PROP_FPS)
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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        frame_count  = 0
        scores       = []
        all_feedback = []
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
                    frame = analyzer.draw_feedback(frame, analysis['feedback'], start_y=40)

                    score_text  = f"Score: {analysis['overall_score']:.0f}/100"
                    score_color = (
                        (0, 255, 0) if analysis['overall_score'] >= 80
                        else (0, 165, 255) if analysis['overall_score'] >= 60
                        else (0, 0, 255)
                    )
                    cv2.putText(frame, score_text, (width - 300, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

                    scores.append(analysis['overall_score'])
                    all_feedback.extend(analysis['feedback'])
                else:
                    cv2.putText(frame, "Esperando pose...", (50, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                out.write(frame)
                progress_bar.progress(frame_count / total_frames)

        finally:
            cap.release()
            out.release()

        progress_bar.empty()
        st.success("[OK] Video procesado correctamente")

        col1, col2, col3 = st.columns(3)
        with col1:
            avg_score = np.mean(scores) if scores else 0
            st.metric("Score Promedio", f"{avg_score:.1f}/100",
                      delta=f"{avg_score - 50:.1f}" if avg_score > 50 else None)
        with col2:
            st.metric("Score Maximo", f"{max(scores) if scores else 0:.1f}/100")
        with col3:
            st.metric("Score Minimo", f"{min(scores) if scores else 0:.1f}/100")

        st.subheader("Evolucion del Score")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=scores, mode='lines+markers', name='Score',
            line=dict(color='#378ADD', width=3), marker=dict(size=6)
        ))
        fig.update_layout(
            title="Score a lo largo del video",
            xaxis_title="Frame", yaxis_title="Score",
            hovermode='x unified', height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feedback General")
        for feedback in list(set(all_feedback)):
            if "[OK]" in feedback:
                st.success(feedback)
            elif "[WARN]" in feedback:
                st.warning(feedback)
            elif "[ERROR]" in feedback:
                st.error(feedback)
            else:
                st.info(feedback)

        with open(temp_output, 'rb') as f:
            st.download_button(
                label="Descargar video procesado",
                data=f.read(),
                file_name=f"{selected_exercise}_analizado.mp4",
                mime="video/mp4"
            )

# ---------- TAB 2: IMAGEN ----------
with tab2:
    st.header("Analizar Imagen Estatica")

    image_file = st.file_uploader(
        "Sube una imagen de tu ejercicio",
        type=['jpg', 'jpeg', 'png'],
        help="JPG o PNG. Maximo 10 MB"
    )

    if image_file is not None:
        temp_dir   = tempfile.mkdtemp()
        temp_image = os.path.join(temp_dir, "image.jpg")

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

            frame = analyzer.draw_skeleton(frame, landmarks)
            frame = analyzer.draw_feedback(frame, analysis['feedback'], start_y=50)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            with col2:
                st.metric("Score", f"{analysis['overall_score']:.0f}/100")
                st.subheader("Feedback")
                for feedback in analysis['feedback']:
                    if "[OK]" in feedback:
                        st.success(feedback)
                    elif "[WARN]" in feedback:
                        st.warning(feedback)
                    elif "[ERROR]" in feedback:
                        st.error(feedback)
                    else:
                        st.info(feedback)
        else:
            st.error("[ERROR] No se pudo detectar pose. Intenta con otra imagen mas clara")

# ---------- TAB 3: EJEMPLOS ----------
with tab3:
    st.header("Ejemplos y Rangos Biomecánicos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forma Correcta")
        st.markdown("""
        - Cuerpo completamente visible en el frame
        - Buena iluminacion
        - Fondo sin distracciones
        - Vista lateral para sentadilla y peso muerto
        """)

    with col2:
        st.subheader("Errores Comunes")
        st.markdown("""
        - Cuerpo parcialmente cortado
        - Luz de espaldas
        - Movimiento muy rapido
        - Fondo muy congestionado
        """)

    st.subheader("Rangos de Angulos por Ejercicio")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Sentadilla**")
        st.markdown("""
        - Rodilla: 75-120 grados
        - Espalda: +/-45 grados
        - Diferencia rodillas: < 10 grados
        """)

    with col2:
        st.markdown("**Flexion**")
        st.markdown("""
        - Codo: 70-120 grados
        - Alineacion cuerpo: > 155 grados
        """)

    with col3:
        st.markdown("**Peso Muerto**")
        st.markdown("""
        - Bisagra cadera: 60-170 grados
        - Rodilla: 20-80 grados (fase baja)
        - Espalda: neutra (no redondeada)
        """)

    st.info(
        "Tecnologia: MediaPipe Pose (33 puntos) + OpenCV + NumPy + Streamlit"
    )

# ====== FOOTER ======
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Powered by MediaPipe")
with col2:
    st.caption("Analisis biomecánico automatico")
with col3:
    st.caption("Los datos no se guardan")
