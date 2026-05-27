"""
process_video.py
Procesa video de ejercicio y genera video de salida con feedback
"""

import cv2
import argparse
from pathlib import Path
from pose_analyzer import PoseAnalyzer, SquatAnalyzer, PushupAnalyzer, DeadliftAnalyzer

def process_exercise_video(input_path, output_path, exercise_type='squat'):
    """
    Procesa video de ejercicio y genera salida con análisis
    
    Args:
        input_path: ruta al video de entrada
        output_path: ruta al video de salida
        exercise_type: tipo de ejercicio ('squat', 'pushup', 'deadlift')
    """
    
    # Seleccionar analizador
    if exercise_type.lower() == 'squat':
        exercise_analyzer = SquatAnalyzer()
        analyzer = exercise_analyzer.analyzer
    elif exercise_type.lower() == 'pushup':
        exercise_analyzer = PushupAnalyzer()
        analyzer = exercise_analyzer.analyzer
    elif exercise_type.lower() == 'deadlift':
        exercise_analyzer = DeadliftAnalyzer()
        analyzer = exercise_analyzer.analyzer
    else:
        raise ValueError(f"Ejercicio desconocido: {exercise_type}")
    
    print(f"🏋️ Analizando {exercise_type.upper()}...")
    print(f"  📥 Input: {input_path}")
    print(f"  📤 Output: {output_path}")
    
    # Abrir video
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {input_path}")
        return False
    
    # Obtener propiedades
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  📽️  FPS: {fps}, Resolución: {width}x{height}, Frames: {total_frames}")
    
    # Crear escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    scores = []
    
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
                
                # Analizar ejercicio
                analysis = exercise_analyzer.analyze(frame, landmarks)
                
                # Dibujar skeleton
                frame = analyzer.draw_skeleton(frame, landmarks)
                
                # Dibujar feedback
                frame = analyzer.draw_feedback(frame, analysis['feedback'], start_y=40)
                
                # Dibujar score
                score_text = f"Score: {analysis['overall_score']:.0f}/100"
                score_color = (0, 255, 0) if analysis['overall_score'] >= 80 else (0, 165, 255) if analysis['overall_score'] >= 60 else (0, 0, 255)
                cv2.putText(frame, score_text, (width - 300, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3)
                
                # Dibujar ángulos principales
                y_offset = height - 80
                if 'knee_angle' in analysis and analysis['knee_angle'] > 0:
                    cv2.putText(frame, f"Rodilla: {analysis['knee_angle']:.1f}°", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                if 'elbow_angle' in analysis and analysis['elbow_angle'] > 0:
                    cv2.putText(frame, f"Codo: {analysis['elbow_angle']:.1f}°", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                if 'back_angle' in analysis:
                    cv2.putText(frame, f"Espalda: {analysis['back_angle']:.1f}°", 
                               (width // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                scores.append(analysis['overall_score'])
            else:
                cv2.putText(frame, "⚠️ Esperando detección de pose...", 
                           (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            # Escribir frame
            out.write(frame)
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  ⏳ {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    except Exception as e:
        print(f"❌ Error durante procesamiento: {e}")
        return False
    
    finally:
        cap.release()
        out.release()
    
    # Estadísticas
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        print("\n✅ ANÁLISIS COMPLETADO")
        print(f"  📊 Score promedio: {avg_score:.1f}/100")
        print(f"  📈 Score máximo: {max_score:.1f}/100")
        print(f"  📉 Score mínimo: {min_score:.1f}/100")
        print(f"  📹 Video guardado en: {output_path}")
    
    return True


def process_image(image_path, exercise_type='squat'):
    """
    Analiza pose en una imagen estática
    """
    
    if exercise_type.lower() == 'squat':
        exercise_analyzer = SquatAnalyzer()
        analyzer = exercise_analyzer.analyzer
    elif exercise_type.lower() == 'pushup':
        exercise_analyzer = PushupAnalyzer()
        analyzer = exercise_analyzer.analyzer
    elif exercise_type.lower() == 'deadlift':
        exercise_analyzer = DeadliftAnalyzer()
        analyzer = exercise_analyzer.analyzer
    else:
        raise ValueError(f"Ejercicio desconocido: {exercise_type}")
    
    print(f"🖼️  Analizando imagen: {image_path}")
    
    # Cargar imagen
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ No se pudo cargar la imagen")
        return None
    
    # Detectar pose
    results = analyzer.detect_pose(frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Analizar
        analysis = exercise_analyzer.analyze(frame, landmarks)
        
        # Dibujar
        frame = analyzer.draw_skeleton(frame, landmarks)
        frame = analyzer.draw_feedback(frame, analysis['feedback'], start_y=40)
        
        print(f"✅ Análisis completado")
        print(f"   Score: {analysis['overall_score']:.0f}/100")
        for feedback in analysis['feedback']:
            print(f"   {feedback}")
        
        return frame
    else:
        print("❌ No se detectó pose en la imagen")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analiza ejercicio en video')
    parser.add_argument('input', help='Ruta al video de entrada')
    parser.add_argument('-o', '--output', default='output.mp4', help='Ruta al video de salida')
    parser.add_argument('-e', '--exercise', default='squat', 
                       choices=['squat', 'pushup', 'deadlift'],
                       help='Tipo de ejercicio')
    parser.add_argument('--image', action='store_true', help='Procesar imagen en lugar de video')
    
    args = parser.parse_args()
    
    if args.image:
        result_frame = process_image(args.input, args.exercise)
        if result_frame is not None:
            output_path = args.output.replace('.mp4', '.jpg')
            cv2.imwrite(output_path, result_frame)
            print(f"   💾 Imagen guardada en: {output_path}")
    else:
        process_exercise_video(args.input, args.output, args.exercise)
