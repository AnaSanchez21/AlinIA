"""
streamlit_fitness_app.py
App Streamlit para análisis de forma en ejercicios
"""

import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from pose_analyzer import PoseAnalyzer, SquatAnalyzer, PushupAnalyzer, DeadliftAnalyzer

# ====== CONFIG ======
st.set_page_config(
    page_title="Fitness Form Analyzer",
    page_icon="🏃",
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
    .feedback-good {
        color: #00AA00;
        font-weight: bold;
    }
    .feedback-warning {
        color: #FF9900;
        font-weight: bold;
    }
    .feedback-bad {
        color: #FF0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ====== SIDEBAR ======
st.sidebar.title("⚙️ Configuración")

exercise_type = st.sidebar.selectbox(
    "Selecciona ejercicio:",
    ["Sentadilla (Squat)", "Flexión (Push-up)", "Peso Muerto (Deadlift)"],
    index=0
)

# Mapeo de nombres
exercise_map = {
    "Sentadilla (Squat)": "squat",
    "Flexión (Push-up)": "pushup",
    "Peso Muerto (Deadlift)": "deadlift"
}

selected_exercise = exercise_map[exercise_type]

# Información del ejercicio
with st.sidebar.expander("ℹ️ Guía de ejercicio"):
    if selected_exercise == "squat":
        st.markdown("""
        ### Sentadilla Correcta
        - **Rodillas:** 80-110°
        - **Espalda:** Recta (±15°)
        - **Rodillas alineadas:** Sin rotación
        - **Peso:** En talones
        """)
    elif selected_exercise == "pushup":
        st.markdown("""
        ### Flexión Correcta
        - **Codos:** 80-100°
        - **Espalda:** Recta, sin cadera caída
        - **Cabeza:** Alineada con espalda
        - **Manos:** Ancho hombros
        """)
    else:
        st.markdown("""
        ### Peso Muerto Correcto
        - **Espalda:** Neutra (muy importante)
        - **Rodillas:** 20-80°
        - **Pecho elevado**
        - **Peso cercano al cuerpo**
        """)

# ====== TÍTULO ======
st.title("🏃 Fitness Form Analyzer")
st.markdown("Analiza tu forma en ejercicios y recibe feedback instantáneo")

# Advertencia
st.warning("""
⚠️ **Descargo de responsabilidad:** Esta herramienta es para propósitos educativos/informativos. 
No reemplaza consulta con entrenador profesional. Consulta a experto antes de hacer ejercicios nuevos.
""")

# ====== TABS ======
tab1, tab2, tab3 = st.tabs(["📹 Procesar Video", "📸 Procesar Imagen", "📊 Ejemplos"])

with tab1:
    st.header("Analizar Video")
    
    video_file = st.file_uploader(
        "Sube un video de tu ejercicio",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Máximo 200 MB. Asegúrate de que te veas completo en el frame"
    )
    
    if video_file is not None:
        # Guardar video temporal
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, "input_video.mp4")
        temp_output = os.path.join(temp_dir, "output_video.mp4")
        
        with open(temp_input, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Procesar
        st.info("🔄 Procesando video... Esto puede tomar un momento")
        
        # Abrir video original
        cap = cv2.VideoCapture(temp_input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Seleccionar analizador
        if selected_exercise == 'squat':
            exercise_analyzer = SquatAnalyzer()
        elif selected_exercise == 'pushup':
            exercise_analyzer = PushupAnalyzer()
        else:
            exercise_analyzer = DeadliftAnalyzer()
        
        analyzer = exercise_analyzer.analyzer
        
        # Crear escritor
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        frame_count = 0
        scores = []
        all_feedback = []
        progress_bar = st.progress(0)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detectar pose
                results = analyzer.detect_pose(frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Analizar
                    analysis = exercise_analyzer.analyze(frame, landmarks)
                    
                    # Dibujar
                    frame = analyzer.draw_skeleton(frame, landmarks)
                    frame = analyzer.draw_feedback(frame, analysis['feedback'], start_y=40)
                    
                    # Score
                    score_text = f"Score: {analysis['overall_score']:.0f}/100"
                    score_color = (0, 255, 0) if analysis['overall_score'] >= 80 else (0, 165, 255) if analysis['overall_score'] >= 60 else (0, 0, 255)
                    cv2.putText(frame, score_text, (width - 300, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)
                    
                    scores.append(analysis['overall_score'])
                    all_feedback.extend(analysis['feedback'])
                else:
                    cv2.putText(frame, "Esperando pose...", (50, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                
                out.write(frame)
                progress = (frame_count / total_frames)
                progress_bar.progress(progress)
        
        finally:
            cap.release()
            out.release()
        
        progress_bar.empty()
        st.success("✅ Video procesado correctamente")
        
        # Mostrar resultados
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = np.mean(scores) if scores else 0
            st.metric("Score Promedio", f"{avg_score:.1f}/100", 
                     delta=f"{avg_score - 50:.1f}" if avg_score > 50 else None)
        
        with col2:
            max_score = max(scores) if scores else 0
            st.metric("Score Máximo", f"{max_score:.1f}/100")
        
        with col3:
            min_score = min(scores) if scores else 0
            st.metric("Score Mínimo", f"{min_score:.1f}/100")
        
        # Gráfico de scores
        st.subheader("📈 Evolución del Score")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines+markers',
            name='Score',
            line=dict(color='#378ADD', width=3),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Score a lo largo del video",
            xaxis_title="Frame",
            yaxis_title="Score",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feedback consolidado
        st.subheader("📋 Feedback General")
        feedback_unique = list(set(all_feedback))
        for feedback in feedback_unique:
            if "✅" in feedback:
                st.success(feedback)
            elif "⚠️" in feedback:
                st.warning(feedback)
            else:
                st.error(feedback)
        
        # Descargar video
        with open(temp_output, 'rb') as f:
            st.download_button(
                label="📥 Descargar video procesado",
                data=f.read(),
                file_name=f"{selected_exercise}_analyzed.mp4",
                mime="video/mp4"
            )

with tab2:
    st.header("Analizar Imagen Estática")
    
    image_file = st.file_uploader(
        "Sube una imagen de tu ejercicio",
        type=['jpg', 'jpeg', 'png'],
        help="JPG o PNG. Máximo 10 MB"
    )
    
    if image_file is not None:
        # Guardar temporal
        temp_dir = tempfile.mkdtemp()
        temp_image = os.path.join(temp_dir, "image.jpg")
        
        with open(temp_image, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Procesar
        frame = cv2.imread(temp_image)
        
        # Seleccionar analizador
        if selected_exercise == 'squat':
            exercise_analyzer = SquatAnalyzer()
        elif selected_exercise == 'pushup':
            exercise_analyzer = PushupAnalyzer()
        else:
            exercise_analyzer = DeadliftAnalyzer()
        
        analyzer = exercise_analyzer.analyzer
        
        # Detectar
        results = analyzer.detect_pose(frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Analizar
            analysis = exercise_analyzer.analyze(frame, landmarks)
            
            # Dibujar
            frame = analyzer.draw_skeleton(frame, landmarks)
            frame = analyzer.draw_feedback(frame, analysis['feedback'], start_y=50)
            
            # Mostrar
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with col2:
                st.metric("Score", f"{analysis['overall_score']:.0f}/100")
                st.subheader("Feedback")
                for feedback in analysis['feedback']:
                    if "✅" in feedback:
                        st.success(feedback)
                    elif "⚠️" in feedback:
                        st.warning(feedback)
                    else:
                        st.error(feedback)
        else:
            st.error("❌ No se pudo detectar pose. Intenta con otra imagen más clara")

with tab3:
    st.header("📚 Ejemplos y Recursos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Forma Correcta")
        st.markdown("""
        - Cuerpo completamente visible
        - Buena iluminación
        - Fondo sin distracciones
        - Ángulos adecuados para cada ejercicio
        """)
    
    with col2:
        st.subheader("❌ Errores Comunes")
        st.markdown("""
        - Cuerpo parcialmente cortado
        - Luz de espaldas
        - Movimiento muy rápido
        - Fondo muy congestionado
        """)
    
    st.info("""
    ### Tecnología utilizada
    - **MediaPipe Pose:** Detección de 33 puntos corporales
    - **OpenCV:** Procesamiento de video
    - **NumPy:** Cálculos de ángulos
    - **Streamlit:** Interfaz web
    """)

# ====== FOOTER ======
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🤖 Powered by MediaPipe")

with col2:
    st.caption("⚡ Análisis en tiempo real")

with col3:
    st.caption("🔒 Los datos no se guardan")
