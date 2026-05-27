"""
pose_analyzer.py
Módulo core para análisis de forma física en ejercicios
"""

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class PoseLandmark:
    """Representa un punto corporal detectado"""
    x: float
    y: float
    z: float
    visibility: float

class PoseAnalyzer:
    """Analizador de poses corporales"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Índices de puntos (MediaPipe Pose)
        self.LANDMARKS = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
    
    def detect_pose(self, frame):
        """Detecta pose en frame RGB"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
    
    def get_landmark(self, landmarks, name):
        """Obtiene coordenadas de un landmark específico"""
        if landmarks is None:
            return None
        idx = self.LANDMARKS.get(name)
        if idx is None:
            raise ValueError(f"Landmark '{name}' no existe")
        
        lm = landmarks[idx]
        return (lm.x, lm.y, lm.z, lm.visibility)
    
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Calcula ángulo entre 3 puntos (en radianes convertido a grados)
        p1, p2, p3 son tuplas (x, y) o (x, y, z)
        p2 es el vértice del ángulo
        """
        # Extraer solo x, y si hay z
        p1 = np.array([p1[0], p1[1]])
        p2 = np.array([p2[0], p2[1]])
        p3 = np.array([p3[0], p3[1]])
        
        # Vectores desde p2
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Producto punto y magnitudes
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if magnitude == 0:
            return 0
        
        cos_angle = dot_product / magnitude
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = acos(cos_angle)
        angle_deg = degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def calculate_back_angle(shoulder, hip):
        """Calcula inclinación de espalda (0° = vertical, 90° = horizontal)"""
        dy = hip[1] - shoulder[1]  # positivo = cadera abajo
        dx = hip[0] - shoulder[0]  # positivo = cadera derecha
        
        # Ángulo respecto a vertical
        angle = np.arctan2(dx, abs(dy))
        return degrees(angle)
    
    def draw_skeleton(self, frame, landmarks):
        """Dibuja skeleton corporal en frame"""
        if landmarks is None:
            return frame
        
        # Conexiones entre puntos
        connections = [
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('left_shoulder', 'right_shoulder'),
            ('left_hip', 'right_hip'),
        ]
        
        h, w, _ = frame.shape
        
        # Dibujar líneas
        for start, end in connections:
            try:
                p1 = self.get_landmark(landmarks, start)
                p2 = self.get_landmark(landmarks, end)
                
                if p1 and p2 and p1[3] > 0.5 and p2[3] > 0.5:  # visibility > 0.5
                    x1, y1 = int(p1[0] * w), int(p1[1] * h)
                    x2, y2 = int(p2[0] * w), int(p2[1] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except:
                continue
        
        # Dibujar puntos
        for name, idx in self.LANDMARKS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.5:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        return frame
    
    def draw_angle(self, frame, angle_value, position, label="", color=(0, 255, 0)):
        """Dibuja un ángulo en el frame"""
        text = f"{label}: {angle_value:.1f}°"
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
        return frame
    
    def draw_feedback(self, frame, feedback_list, start_y=30):
        """Dibuja lista de feedback en el frame"""
        y = start_y
        for feedback in feedback_list:
            # Determinar color según tipo
            if "✅" in feedback:
                color = (0, 255, 0)  # Verde
            elif "⚠️" in feedback:
                color = (0, 165, 255)  # Naranja
            elif "❌" in feedback:
                color = (0, 0, 255)  # Rojo
            else:
                color = (255, 255, 255)  # Blanco
            
            cv2.putText(frame, feedback, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.65, color, 2)
            y += 30
        
        return frame


class SquatAnalyzer:
    """Analizador específico para sentadillas"""
    
    # Rangos de ángulos correctos
    KNEE_ANGLE_RANGE = (75, 120)
    BACK_ANGLE_RANGE = (-15, 15)  # ±15° de vertical
    
    def __init__(self):
        self.analyzer = PoseAnalyzer()
    
    def analyze(self, frame, landmarks) -> Dict:
        """Analiza forma de sentadilla"""
        results = {
            'knee_angle': 0,
            'back_angle': 0,
            'feedback': [],
            'overall_score': 100
        }
        
        if landmarks is None:
            return results
        
        # Obtener puntos
        left_hip = self.analyzer.get_landmark(landmarks, 'left_hip')
        left_knee = self.analyzer.get_landmark(landmarks, 'left_knee')
        left_ankle = self.analyzer.get_landmark(landmarks, 'left_ankle')
        left_shoulder = self.analyzer.get_landmark(landmarks, 'left_shoulder')
        
        if not all([left_hip, left_knee, left_ankle, left_shoulder]):
            results['feedback'].append("❌ No se detectan puntos suficientes")
            return results
        
        # Calcular ángulos
        knee_angle = self.analyzer.calculate_angle(left_hip, left_knee, left_ankle)
        back_angle = self.analyzer.calculate_back_angle(left_shoulder, left_hip)
        
        results['knee_angle'] = knee_angle
        results['back_angle'] = back_angle
        
        # Validar rodilla
        if knee_angle < self.KNEE_ANGLE_RANGE[0]:
            results['feedback'].append(f"⚠️ Rodilla insuficiente ({knee_angle:.1f}°). Baja más")
            results['overall_score'] -= 15
        elif knee_angle > self.KNEE_ANGLE_RANGE[1]:
            results['feedback'].append(f"⚠️ Flexión excesiva ({knee_angle:.1f}°). Sube un poco")
            results['overall_score'] -= 10
        else:
            results['feedback'].append(f"✅ Rodilla correcta ({knee_angle:.1f}°)")
        
        # Validar espalda
        if abs(back_angle) > self.BACK_ANGLE_RANGE[1]:
            results['feedback'].append(f"❌ Espalda inclinada ({back_angle:.1f}°). Mantén torso vertical")
            results['overall_score'] -= 25
        else:
            results['feedback'].append(f"✅ Espalda recta ({back_angle:.1f}°)")
        
        # Validar alineación de rodillas
        right_hip = self.analyzer.get_landmark(landmarks, 'right_hip')
        right_knee = self.analyzer.get_landmark(landmarks, 'right_knee')
        right_ankle = self.analyzer.get_landmark(landmarks, 'right_ankle')
        
        if all([right_hip, right_knee, right_ankle]):
            right_knee_angle = self.analyzer.calculate_angle(right_hip, right_knee, right_ankle)
            angle_diff = abs(knee_angle - right_knee_angle)
            
            if angle_diff > 10:
                results['feedback'].append(f"⚠️ Rodillas desalineadas (diff: {angle_diff:.1f}°)")
                results['overall_score'] -= 15
        
        results['overall_score'] = max(0, results['overall_score'])
        return results


class PushupAnalyzer:
    """Analizador específico para flexiones"""
    
    ELBOW_ANGLE_RANGE = (70, 120)
    BACK_ANGLE_RANGE = (-15, 15)
    
    def __init__(self):
        self.analyzer = PoseAnalyzer()
    
    def analyze(self, frame, landmarks) -> Dict:
        """Analiza forma de flexión"""
        results = {
            'elbow_angle': 0,
            'back_angle': 0,
            'feedback': [],
            'overall_score': 100
        }
        
        if landmarks is None:
            return results
        
        # Obtener puntos
        left_shoulder = self.analyzer.get_landmark(landmarks, 'left_shoulder')
        left_elbow = self.analyzer.get_landmark(landmarks, 'left_elbow')
        left_wrist = self.analyzer.get_landmark(landmarks, 'left_wrist')
        left_hip = self.analyzer.get_landmark(landmarks, 'left_hip')
        
        if not all([left_shoulder, left_elbow, left_wrist, left_hip]):
            results['feedback'].append("❌ No se detectan puntos suficientes")
            return results
        
        # Calcular ángulos
        elbow_angle = self.analyzer.calculate_angle(left_shoulder, left_elbow, left_wrist)
        back_angle = self.analyzer.calculate_back_angle(left_shoulder, left_hip)
        
        results['elbow_angle'] = elbow_angle
        results['back_angle'] = back_angle
        
        # Validar codo
        if elbow_angle < self.ELBOW_ANGLE_RANGE[0]:
            results['feedback'].append(f"⚠️ Flexión insuficiente ({elbow_angle:.1f}°). Baja más")
            results['overall_score'] -= 15
        elif elbow_angle > self.ELBOW_ANGLE_RANGE[1]:
            results['feedback'].append(f"⚠️ Flexión excesiva ({elbow_angle:.1f}°). Sube un poco")
            results['overall_score'] -= 10
        else:
            results['feedback'].append(f"✅ Codo correcto ({elbow_angle:.1f}°)")
        
        # Validar espalda
        if abs(back_angle) > self.BACK_ANGLE_RANGE[1]:
            results['feedback'].append(f"❌ Espalda inclinada ({back_angle:.1f}°). Mantén alineado")
            results['overall_score'] -= 25
        else:
            results['feedback'].append(f"✅ Espalda alineada")
        
        results['overall_score'] = max(0, results['overall_score'])
        return results


class DeadliftAnalyzer:
    """Analizador específico para peso muerto"""
    
    BACK_ANGLE_RANGE = (-10, 10)  # Más estricto
    KNEE_ANGLE_RANGE = (20, 80)
    
    def __init__(self):
        self.analyzer = PoseAnalyzer()
    
    def analyze(self, frame, landmarks) -> Dict:
        """Analiza forma de peso muerto"""
        results = {
            'back_angle': 0,
            'knee_angle': 0,
            'feedback': [],
            'overall_score': 100
        }
        
        if landmarks is None:
            return results
        
        # Obtener puntos
        shoulder = self.analyzer.get_landmark(landmarks, 'left_shoulder')
        hip = self.analyzer.get_landmark(landmarks, 'left_hip')
        knee = self.analyzer.get_landmark(landmarks, 'left_knee')
        ankle = self.analyzer.get_landmark(landmarks, 'left_ankle')
        
        if not all([shoulder, hip, knee, ankle]):
            results['feedback'].append("❌ No se detectan puntos suficientes")
            return results
        
        # Calcular ángulos
        back_angle = self.analyzer.calculate_back_angle(shoulder, hip)
        knee_angle = self.analyzer.calculate_angle(hip, knee, ankle)
        
        results['back_angle'] = back_angle
        results['knee_angle'] = knee_angle
        
        # Validar espalda (crítico en peso muerto)
        if abs(back_angle) > self.BACK_ANGLE_RANGE[1]:
            results['feedback'].append(f"❌ PELIGRO: Espalda inclinada ({back_angle:.1f}°). Riesgo de lesión")
            results['overall_score'] -= 40
        else:
            results['feedback'].append(f"✅ Espalda neutra")
        
        # Validar rodilla
        if knee_angle < self.KNEE_ANGLE_RANGE[0]:
            results['feedback'].append(f"⚠️ Rodilla muy flexionada ({knee_angle:.1f}°)")
            results['overall_score'] -= 15
        elif knee_angle > self.KNEE_ANGLE_RANGE[1]:
            results['feedback'].append(f"⚠️ Rodilla muy extendida ({knee_angle:.1f}°)")
            results['overall_score'] -= 10
        else:
            results['feedback'].append(f"✅ Rodilla correcta ({knee_angle:.1f}°)")
        
        results['overall_score'] = max(0, results['overall_score'])
        return results
