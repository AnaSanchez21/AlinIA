"""
test_pose_image.py
DÍA 2: Detector de Pose en Imagen Simple
===============================================

Este script es tu PRIMER programa de IA.
Solo copia/pega esto en un archivo nuevo y ejecuta.

Qué hace:
1. Carga una imagen
2. Detecta 33 puntos del cuerpo (MediaPipe)
3. Dibuja skeleton (esqueleto)
4. Guarda resultado

Duración: ~2 minutos por imagen
"""

import cv2
from src.pose_analyzer import PoseAnalyzer

# ====== CONFIGURACIÓN ======
IMAGE_PATH = "pose.png"  # 
OUTPUT_PATH = "resultado.jpg"

print("=" * 60)
print(" DETECTOR DE POSE - DÍA 2")
print("=" * 60)

# ====== PASO 1: CREAR ANALIZADOR ======
print("\n Inicializando MediaPipe Pose...")
analyzer = PoseAnalyzer()
print("   [OK] Listo")

# ====== PASO 2: CARGAR IMAGEN ======
print("\n Cargando imagen...")
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"   [ERROR] Error: No se pudo cargar '{IMAGE_PATH}'")
    print(f"   [INFO] Verifica:")
    print(f"      - El archivo existe en esta carpeta")
    print(f"      - Se llama exactamente así (mayúsculas/minúsculas importa)")
    print(f"      - Es JPG o PNG")
    exit()

print(f"   [OK] Imagen cargada: {frame.shape}")
print(f"     Tamaño: {frame.shape[1]}x{frame.shape[0]} píxeles")

# ====== PASO 3: DETECTAR POSE ======
print("\n Detectando pose (IA analizando imagen)...")
results = analyzer.detect_pose(frame)

if results.pose_landmarks is None:
    print("   [ERROR] No se detectó pose")
    print("   [INFO] Razones comunes:")
    print("      - Persona de espaldas (difícil)")
    print("      - Iluminación muy oscura")
    print("      - Cuerpo parcialmente cortado")
    print("   -> Intenta con otra imagen")
    exit()

landmarks = results.pose_landmarks.landmark
print(f"   [OK] ¡Pose detectada!")
print(f"   [OK] {len(landmarks)} puntos detectados")

# ====== PASO 4: ANALIZAR PUNTOS ======
print("\n Analizando puntos detectados...")
visible_count = 0
for i, lm in enumerate(landmarks):
    if lm.visibility > 0.5:  # Punto visible
        visible_count += 1

print(f"   [OK] Puntos visibles: {visible_count}/{len(landmarks)}")
print(f"   [OK] Confianza promedio: {sum(lm.visibility for lm in landmarks)/len(landmarks):.1%}")

# ====== PASO 5: DIBUJAR SKELETON ======
print("\n Dibujando skeleton...")
frame = analyzer.draw_skeleton(frame, landmarks)
print("   [OK] Skeleton dibujado")

# ====== PASO 6: GUARDAR RESULTADO ======
print("\n Guardando resultado...")
cv2.imwrite(OUTPUT_PATH, frame)
print(f"   [OK] Guardado en: {OUTPUT_PATH}")

print("\n" + "=" * 60)
print("[OK] ÉXITO - Abre resultado.jpg para ver el skeleton")
print("=" * 60)

