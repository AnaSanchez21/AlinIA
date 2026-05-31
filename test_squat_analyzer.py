"""
test_squat_analyzer.py
DÍA 4: Analizador de Sentadillas
===============================================

Valida forma de sentadilla:
  - Rodilla: 75-120 grados (correcta)
  - Espalda: +/-15 grados (vertical)
  - Rodillas alineadas: <10 grados diferencia

Genera FEEDBACK automático como un coach:
  [OK] Rodilla correcta
  [WARN] Espalda ligeramente inclinada
  [ERROR] Rodilla muy poco flexionada

SCORE: 0-100 (100 = forma perfecta)

Duración: ~2 minutos por imagen
"""

import cv2
import numpy as np
from pose_analyzer import SquatAnalyzer

# ====== CONFIGURACIÓN ======
IMAGE_PATH = "sentadillas.jpg"  
OUTPUT_PATH = "resultado_squat.jpg"

print("=" * 70)
print(" ANALIZADOR DE SENTADILLA - DÍA 4")
print("=" * 70)

# ====== PASO 1: CREAR ANALIZADOR ======
print("\n Inicializando analizador...")
squat = SquatAnalyzer()
print("   [OK] Listo")

# ====== PASO 2: CARGAR IMAGEN ======
print("\n Cargando imagen...")
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"   [ERROR] Error: No se pudo cargar '{IMAGE_PATH}'")
    print(f"   [INFO] Verifica que sea una sentadilla y que exista el archivo")
    exit()

print(f"   [OK] Imagen cargada: {frame.shape}")

# ====== PASO 3: ANALIZAR SENTADILLA ======
print("\n Analizando forma de sentadilla...")

# Detectar pose
results = squat.analyzer.detect_pose(frame)

if results.pose_landmarks is None:
    print("   [ERROR] Pose NO detectada")
    print("   [INFO] Intenta con imagen más clara")
    exit()

landmarks = results.pose_landmarks.landmark
print("   [OK] Pose detectada")

# ANALIZAR (esto es lo nuevo hoy)
analysis = squat.analyze(frame, landmarks)

print("   [OK] Análisis completado")

# ====== PASO 4: MOSTRAR RESULTADOS ======
print("\n RESULTADOS DETALLADOS:")
print("-" * 70)

print(f"\n   ÁNGULOS MEDIDOS:")
print(f"    - Rodilla izquierda:  {analysis['knee_angle']:.1f} grados")
print(f"    - Espalda:            {analysis['back_angle']:.1f} grados")

print(f"\n   FEEDBACK:")
for msg in analysis['feedback']:
    print(f"    {msg}")

print(f"\n   SCORE GENERAL: {analysis['overall_score']:.0f}/100")

# ====== PASO 5: INTERPRETAR SCORE ======
print("\n INTERPRETACIÓN:")
print("-" * 70)

score = analysis['overall_score']

if score >= 90:
    verdict = " EXCELENTE - Forma practicamente perfecta"
    color = "[OK]"
elif score >= 80:
    verdict = " BUENA - Forma correcta con pequeños ajustes"
    color = "[OK]"
elif score >= 70:
    verdict = " OK - Funciona pero hay errores notables"
    color = "[WARN]"
elif score >= 50:
    verdict = "[ERROR] MALA - Forma incorrecta, riesgo de lesion"
    color = "[WARN]"
else:
    verdict = " MUY MALA - PELIGROSO, corrige inmediatamente"
    color = "[ERROR]"

print(f"\n  {color} {verdict}")
print(f"     Score: {score:.0f}/100")

# Recomendaciones específicas
print(f"\n  [INFO] RECOMENDACIONES:")
if score < 70:
    print(f"     - Mira un video tutorial de sentadilla correcta")
    print(f"     - Practica sin peso primero")
    print(f"     - Pide feedback a un entrenador")
elif score < 90:
    print(f"     - Casi perfecto, fine-tuning de técnica")
    print(f"     - Lee el feedback arriba")
else:
    print(f"     - ¡Forma excelente! Mantén así")
    print(f"     - Aumenta peso/repeticiones")

print("-" * 70)

# ====== PASO 6: DIBUJAR EN IMAGEN ======
print("\n Dibujando análisis en imagen...")

h, w, _ = frame.shape

# Dibuja skeleton primero
frame = squat.analyzer.draw_skeleton(frame, landmarks)

# Dibuja texto con información
y_pos = 40

# Título
cv2.putText(frame, f"SQUAT ANALYSIS - Score: {analysis['overall_score']:.0f}/100",
            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
y_pos += 40

# Ángulos
cv2.putText(frame, f"Knee Angle: {analysis['knee_angle']:.1f} grados (75-120)",
            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
y_pos += 30

cv2.putText(frame, f"Back Angle: {analysis['back_angle']:.1f} grados (+-15)",
            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
y_pos += 30

# Feedback
y_pos += 10
for msg in analysis['feedback']:
    # Determinar color según tipo
    if "[OK]" in msg:
        color = (0, 255, 0)  # Verde
    elif "[WARN]" in msg:
        color = (0, 165, 255)  # Naranja
    else:
        color = (0, 0, 255)  # Rojo
    
    cv2.putText(frame, msg, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    y_pos += 30

# Verdict
cv2.putText(frame, verdict, (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if score >= 70 else (0, 0, 255), 2)

cv2.imwrite(OUTPUT_PATH, frame)
print(f"   [OK] Guardado en: {OUTPUT_PATH}")

# ====== PASO 7: INFORMACIÓN ÚTIL ======
print("\n" + "=" * 70)
print("[OK] ÉXITO - Abre resultado_squat.jpg para ver análisis visual")
print("=" * 70)

print("\n[INFO] REFERENCIA RÁPIDA DE RANGOS:")
print("   SENTADILLA:")
print("   - Rodilla: 75-120 grados (< 75 = baja más, > 120 = sube)")
print("   - Espalda: +/-15 grados (0 grados = vertical, > 15 grados = inclínate menos)")
print("   - Rodillas alineadas: diferencia < 10 grados")



