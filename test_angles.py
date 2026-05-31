"""
test_angles.py
DÍA 3: Cálculo de Ángulos Corporales
===============================================

Calcula ángulos reales de tu cuerpo en ejercicios:
  - Ángulo de rodilla (sentadilla)
  - Ángulo de espalda (inclinación)
  - Ángulo de codo (flexión)

Duración: ~2 minutos por imagen
"""

import cv2
import numpy as np
from math import degrees, acos
from pose_analyzer import PoseAnalyzer

# ====== CONFIGURACIÓN ======
IMAGE_PATH = "tu_foto.jpg"  #  CAMBIA ESTO
OUTPUT_PATH = "resultado_angulos.jpg"

print("=" * 70)
print(" CÁLCULO DE ÁNGULOS - DÍA 3")
print("=" * 70)

# ====== PASO 1: CARGAR Y DETECTAR ======
print("\n Cargando imagen y detectando pose...")
analyzer = PoseAnalyzer()
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"   [ERROR] No se pudo cargar '{IMAGE_PATH}'")
    exit()

print(f"   [OK] Imagen cargada")

results = analyzer.detect_pose(frame)
if results.pose_landmarks is None:
    print("   [ERROR] Pose NO detectada")
    exit()

landmarks = results.pose_landmarks.landmark
print(f"   [OK] Pose detectada - 33 puntos")

# ====== PASO 2: OBTENER PUNTOS CLAVE ======
print("\n Extrayendo puntos clave...")

# LADO IZQUIERDO
left_shoulder = (landmarks[11].x, landmarks[11].y, landmarks[11].z)
left_hip = (landmarks[23].x, landmarks[23].y, landmarks[23].z)
left_knee = (landmarks[25].x, landmarks[25].y, landmarks[25].z)
left_ankle = (landmarks[27].x, landmarks[27].y, landmarks[27].z)

# LADO DERECHO
right_shoulder = (landmarks[12].x, landmarks[12].y, landmarks[12].z)
right_hip = (landmarks[24].x, landmarks[24].y, landmarks[24].z)
right_knee = (landmarks[26].x, landmarks[26].y, landmarks[26].z)
right_ankle = (landmarks[28].x, landmarks[28].y, landmarks[28].z)

# BRAZOS
left_elbow = (landmarks[13].x, landmarks[13].y, landmarks[13].z)
right_elbow = (landmarks[14].x, landmarks[14].y, landmarks[14].z)
left_wrist = (landmarks[15].x, landmarks[15].y, landmarks[15].z)
right_wrist = (landmarks[16].x, landmarks[16].y, landmarks[16].z)

print(f"   [OK] Puntos extraídos")

# ====== PASO 3: CALCULAR ÁNGULOS ======
print("\n Calculando ángulos...")

# RODILLAS
left_knee_angle = analyzer.calculate_angle(left_hip, left_knee, left_ankle)
right_knee_angle = analyzer.calculate_angle(right_hip, right_knee, right_ankle)

# CODOS
left_elbow_angle = analyzer.calculate_angle(left_shoulder, left_elbow, left_wrist)
right_elbow_angle = analyzer.calculate_angle(right_shoulder, right_elbow, right_wrist)

# ESPALDA (inclinación)
# Calculamos la inclinación de la columna respecto a la vertical
dy_spine = left_hip[1] - left_shoulder[1]  # negativo = cadera arriba
dx_spine = left_hip[0] - left_shoulder[0]  # positivo = cadera derecha
back_angle = degrees(np.arctan2(abs(dx_spine), abs(dy_spine)))

# CADERAS (flexión)
# Ángulo entre espalda y pierna
hip_angle = analyzer.calculate_angle(left_shoulder, left_hip, left_knee)

print(f"   [OK] Ángulos calculados")

# ====== PASO 4: MOSTRAR RESULTADOS ======
print("\n RESULTADOS:")
print("-" * 70)
print(f"\n  RODILLAS:")
print(f"    - Rodilla izquierda:  {left_knee_angle:6.1f} grados")
print(f"    - Rodilla derecha:    {right_knee_angle:6.1f} grados")
print(f"    - Diferencia:         {abs(left_knee_angle - right_knee_angle):6.1f} grados")

print(f"\n  CODOS:")
print(f"    - Codo izquierdo:     {left_elbow_angle:6.1f} grados")
print(f"    - Codo derecho:       {right_elbow_angle:6.1f} grados")
print(f"    - Diferencia:         {abs(left_elbow_angle - right_elbow_angle):6.1f} grados")

print(f"\n  POSTURA:")
print(f"    - Inclinación espalda: {back_angle:6.1f} grados  (0 grados = vertical, 90 grados = horizontal)")
print(f"    - Ángulo cadera:       {hip_angle:6.1f} grados")

print("-" * 70)

# ====== PASO 5: INTERPRETACIÓN ======
print("\n INTERPRETACIÓN:")
print("-" * 70)

# Rodillas
if abs(left_knee_angle - right_knee_angle) > 10:
    print(f"   [WARN] Rodillas desalineadas (diferencia: {abs(left_knee_angle - right_knee_angle):.1f} grados)")
else:
    print(f"   [OK] Rodillas alineadas")

# Espalda
if back_angle < 5:
    print(f"   [OK] Espalda vertical (buena postura)")
elif back_angle < 20:
    print(f"   [WARN] Espalda ligeramente inclinada ({back_angle:.1f} grados)")
else:
    print(f"   [ERROR] Espalda muy inclinada ({back_angle:.1f} grados) - Riesgo de lesión")

# Codos
if abs(left_elbow_angle - right_elbow_angle) > 15:
    print(f"   [WARN] Codos desalineados")
else:
    print(f"   [OK] Codos alineados")

print("-" * 70)

# ====== PASO 6: DIBUJAR EN IMAGEN ======
print("\n Dibujando ángulos en imagen...")

h, w, _ = frame.shape

# Dibuja skeleton primero
frame = analyzer.draw_skeleton(frame, landmarks)

# Dibuja ángulos como texto
y_pos = 40
cv2.putText(frame, f"Rodilla Izq: {left_knee_angle:.1f}", (10, y_pos),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
y_pos += 30

cv2.putText(frame, f"Rodilla Der: {right_knee_angle:.1f}", (10, y_pos),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
y_pos += 30

cv2.putText(frame, f"Espalda: {back_angle:.1f}", (10, y_pos),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
y_pos += 30

cv2.putText(frame, f"Cadera: {hip_angle:.1f}", (10, y_pos),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Dibuja puntos importantes en rojo
cv2.circle(frame, (int(left_knee[0]*w), int(left_knee[1]*h)), 8, (0, 0, 255), -1)
cv2.circle(frame, (int(right_knee[0]*w), int(right_knee[1]*h)), 8, (0, 0, 255), -1)
cv2.circle(frame, (int(left_hip[0]*w), int(left_hip[1]*h)), 8, (0, 0, 255), -1)
cv2.circle(frame, (int(left_shoulder[0]*w), int(left_shoulder[1]*h)), 8, (0, 0, 255), -1)

cv2.imwrite(OUTPUT_PATH, frame)
print(f"   [OK] Guardado en: {OUTPUT_PATH}")

print("\n" + "=" * 70)
print("[OK] ÉXITO - Abre resultado_angulos.jpg para ver ángulos dibujados")
print("=" * 70)

# ====== INFORMACIÓN ÚTIL ======
print("\n[INFO] RANGOS TÍPICOS SEGÚN EJERCICIO:")
print("   - Sentadilla:")
print("     - Rodilla: 75-110 grados")
print("     - Espalda: +/-15 grados")
print("   - Flexión:")
print("     - Codo: 80-120 grados")
print("     - Espalda: +/-10 grados")
print("   - Peso muerto:")
print("     - Espalda: +/-5 grados (muy importante)")
print("     - Rodilla: 20-80 grados")
