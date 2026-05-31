"""
test_video.py
DÍA 5: Procesamiento de Video en Tiempo Real
===============================================

Procesa video completo frame-by-frame:
  - Detecta pose en cada frame
  - Calcula ángulos por frame
  - Valida forma (SquatAnalyzer)
  - Acumula scores
  - Genera gráfico de evolución
  - Exporta video procesado

Duración: 10 segundos video = ~30-60 segundos procesamiento (sin GPU)

Uso:
  python test_video.py -i video.mp4 -e squat
  
Opciones:
  -i, --input    Ruta del video (requerido)
  -e, --exercise Tipo: squat, pushup, deadlift (default: squat)
  -o, --output   Ruta salida (default: video_procesado.mp4)
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from pose_analyzer import SquatAnalyzer, PushupAnalyzer, DeadliftAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime

# ====== CONFIGURACIÓN ======
EJERCICIOS = {
    'squat': SquatAnalyzer,
    'pushup': PushupAnalyzer,
    'deadlift': DeadliftAnalyzer
}

print("=" * 80)
print("PROCESADOR DE VIDEO - DÍA 5")
print("=" * 80)

# ====== ARGUMENTOS ======
parser = argparse.ArgumentParser(description='Procesa video de ejercicio')
parser.add_argument('-i', '--input', required=True, help='Ruta video entrada')
parser.add_argument('-e', '--exercise', default='squat', 
                    choices=['squat', 'pushup', 'deadlift'],
                    help='Tipo de ejercicio')
parser.add_argument('-o', '--output', default='video_procesado.mp4',
                    help='Ruta video salida')

args = parser.parse_args()

input_path = args.input
exercise_type = args.exercise
output_path = args.output

# ====== PASO 1: VALIDAR ENTRADA ======
print(f"\n1. Validando entrada...")
if not Path(input_path).exists():
    print(f"   [ERROR] Video no encontrado: {input_path}")
    exit(1)

print(f"   [OK] Video encontrado: {input_path}")

# ====== PASO 2: CARGAR VIDEO ======
print(f"\n2. Cargando video...")
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print(f"   [ERROR] No se pudo abrir video")
    exit(1)

# Propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = total_frames / fps if fps > 0 else 0

print(f"   [OK] Propiedades:")
print(f"       - FPS: {fps}")
print(f"       - Resolución: {frame_width}x{frame_height}")
print(f"       - Total frames: {total_frames}")
print(f"       - Duración: {duration_seconds:.1f} segundos")

# ====== PASO 3: INICIALIZAR ANALIZADOR ======
print(f"\n3. Inicializando analizador ({exercise_type})...")
if exercise_type not in EJERCICIOS:
    print(f"   [ERROR] Ejercicio desconocido: {exercise_type}")
    exit(1)

analyzer_class = EJERCICIOS[exercise_type]
analyzer = analyzer_class()
print(f"   [OK] {exercise_type.upper()} listo")

# ====== PASO 4: CREAR VIDEO WRITER ======
print(f"\n4. Configurando salida de video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"   [ERROR] No se pudo crear video salida")
    exit(1)

print(f"   [OK] Video writer listo: {output_path}")

# ====== PASO 5: PROCESAR FRAMES ======
print(f"\n5. Procesando frames...")
print(f"   {'Frame':<10} {'Score':<10} {'Feedback':<40} {'Status':<10}")
print("   " + "-" * 80)

scores_por_frame = []
feedbacks_por_frame = []
frame_count = 0
errors = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Detectar pose
    results = analyzer.analyzer.detect_pose(frame)
    
    if results.pose_landmarks is None:
        # Pose no detectada, score 0
        score = 0
        feedback = "[WARN] Pose no detectada"
        status = "SKIP"
        errors += 1
    else:
        # Analizar ejercicio
        landmarks = results.pose_landmarks.landmark
        analysis = analyzer.analyze(frame, landmarks)
        
        score = analysis['overall_score']
        feedback = analysis['feedback'][0] if analysis['feedback'] else "[OK] Correcto"
        status = "OK"
    
    scores_por_frame.append(score)
    feedbacks_por_frame.append(feedback)
    
    # Mostrar progreso cada 10 frames
    if frame_count % 10 == 0 or frame_count == 1:
        feedback_short = feedback[:35].ljust(35)
        print(f"   {frame_count:<10} {score:<10.1f} {feedback_short:<40} {status:<10}")
    
    # Dibuja información en frame
    frame_display = frame.copy()
    
    # Dibuja skeleton si pose fue detectada
    if results.pose_landmarks is not None:
        frame_display = analyzer.analyzer.draw_skeleton(frame_display, landmarks)
    
    # Dibuja score grande
    cv2.putText(frame_display, f"Score: {score:.0f}/100", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Dibuja feedback
    cv2.putText(frame_display, feedback[:50], (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Dibuja contador de frame
    cv2.putText(frame_display, f"Frame: {frame_count}/{total_frames}", 
               (frame_width - 250, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Escribe frame procesado
    out.write(frame_display)

cap.release()
out.release()

print("   " + "-" * 80)
print(f"   [OK] Procesamiento completado")
print(f"       - Frames procesados: {frame_count}")
print(f"       - Errores (pose no detectada): {errors}")

# ====== PASO 6: CALCULAR ESTADÍSTICAS ======
print(f"\n6. Calculando estadísticas...")

if len(scores_por_frame) == 0:
    print(f"   [ERROR] No hay datos para procesar")
    exit(1)

scores_validos = [s for s in scores_por_frame if s > 0]
promedio = np.mean(scores_validos) if scores_validos else 0
maximo = np.max(scores_validos) if scores_validos else 0
minimo = np.min(scores_validos) if scores_validos else 0
frames_buenos = sum(1 for s in scores_validos if s >= 80)
frames_malos = sum(1 for s in scores_validos if s < 50)

print(f"   [OK] Estadísticas finales:")
print(f"       - Score promedio: {promedio:.1f}/100")
print(f"       - Score máximo: {maximo:.0f}/100")
print(f"       - Score mínimo: {minimo:.0f}/100")
print(f"       - Frames buenos (>=80): {frames_buenos}/{len(scores_validos)}")
print(f"       - Frames malos (<50): {frames_malos}/{len(scores_validos)}")

# ====== PASO 7: GENERAR GRÁFICO ======
print(f"\n7. Generando gráfico...")

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(scores_por_frame) + 1), scores_por_frame, 
         linewidth=2, color='#378ADD', label='Score por frame')

# Líneas de referencia
plt.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Bueno (>=80)')
plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Malo (<50)')

plt.xlabel('Frame', fontsize=12)
plt.ylabel('Score (0-100)', fontsize=12)
plt.title(f'Evolución de Scores - {exercise_type.upper()} ({duration_seconds:.1f}s)', 
          fontsize=14)
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

graph_path = 'grafico_scores.png'
plt.savefig(graph_path, dpi=100)
plt.close()

print(f"   [OK] Gráfico guardado: {graph_path}")

# ====== PASO 8: GUARDAR JSON ======
print(f"\n8. Guardando estadísticas...")

estadisticas = {
    'timestamp': datetime.now().isoformat(),
    'ejercicio': exercise_type,
    'video_entrada': input_path,
    'video_salida': output_path,
    'duracion_segundos': duration_seconds,
    'total_frames': total_frames,
    'fps': fps,
    'scores': {
        'promedio': float(promedio),
        'maximo': float(maximo),
        'minimo': float(minimo),
        'frames_buenos_80plus': frames_buenos,
        'frames_malos_menos50': frames_malos,
        'frames_totales_validos': len(scores_validos)
    },
    'todos_scores': [float(s) for s in scores_por_frame]
}

json_path = 'estadisticas.json'
with open(json_path, 'w') as f:
    json.dump(estadisticas, f, indent=2)

print(f"   [OK] Estadísticas guardadas: {json_path}")

# ====== RESUMEN FINAL ======
print("\n" + "=" * 80)
print("PROCESAMIENTO COMPLETADO")
print("=" * 80)

print(f"\n[OK] ARCHIVOS GENERADOS:")
print(f"     1. {output_path} - Video procesado con overlay")
print(f"     2. {graph_path} - Gráfico de evolución")
print(f"     3. {json_path} - Estadísticas en JSON")

print(f"\n[OK] RESULTADOS:")
print(f"     Score promedio: {promedio:.1f}/100")
print(f"     Score máximo:   {maximo:.0f}/100")
print(f"     Score mínimo:   {minimo:.0f}/100")
print(f"     Duración:       {duration_seconds:.1f} segundos")
print(f"     Calidad:        {100 * frames_buenos / len(scores_validos):.0f}% de frames buenos")

print(f"\n[OK] Próximo paso: Visualiza video_procesado.mp4 en un reproductor")
print("=" * 80)