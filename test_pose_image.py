import cv2
from pose_analyzer import PoseAnalyzer

# Crear analizador
analyzer = PoseAnalyzer()

# Cargar imagen
image_path = "pose.png" # CAMBIA ESTO
frame = cv2.imread(image_path)

if frame is None:
    print(f"❌ No se pudo cargar: {image_path}")
else:
    print(f"✓ Imagen cargada: {frame.shape}")
    
    # Detectar pose
    results = analyzer.detect_pose(frame)
    
    if results.pose_landmarks:
        print("✓ Pose detectada!")
        landmarks = results.pose_landmarks.landmark
        
        # Dibujar skeleton
        frame = analyzer.draw_skeleton(frame, landmarks)
        
        # Guardar
        output_path = "resultado.jpg"
        cv2.imwrite(output_path, frame)
        print(f"✓ Guardado en: {output_path}")
    else:
        print("❌ Pose NO detectada")