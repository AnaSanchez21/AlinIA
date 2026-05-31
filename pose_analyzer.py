"""
pose_analyzer.py
Modulo core para analisis de forma fisica en ejercicios
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
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

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
        """Obtiene coordenadas de un landmark especifico"""
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
        Calcula angulo entre 3 puntos (grados).
        p2 es el vertice del angulo.
        """
        p1 = np.array([p1[0], p1[1]])
        p2 = np.array([p2[0], p2[1]])
        p3 = np.array([p3[0], p3[1]])

        v1 = p1 - p2
        v2 = p3 - p2

        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        if magnitude == 0:
            return 0

        cos_angle = np.dot(v1, v2) / magnitude
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return degrees(acos(cos_angle))

    @staticmethod
    def calculate_back_angle(shoulder, hip):
        """
        Calcula inclinacion del tronco respecto a la vertical.
        0 deg = vertical, 90 deg = horizontal.
        """
        dy = hip[1] - shoulder[1]
        dx = hip[0] - shoulder[0]
        angle = np.arctan2(dx, abs(dy))
        return degrees(angle)

    def draw_skeleton(self, frame, landmarks):
        """Dibuja skeleton corporal en frame"""
        if landmarks is None:
            return frame

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

        for start, end in connections:
            try:
                p1 = self.get_landmark(landmarks, start)
                p2 = self.get_landmark(landmarks, end)
                if p1 and p2 and p1[3] > 0.5 and p2[3] > 0.5:
                    x1, y1 = int(p1[0] * w), int(p1[1] * h)
                    x2, y2 = int(p2[0] * w), int(p2[1] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception:
                continue

        for name, idx in self.LANDMARKS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.5:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        return frame

    def draw_angle_at_joint(self, frame, landmark_name, angle_value, landmarks,
                            color=(255, 255, 0)):
        """Dibuja el valor del angulo sobre la articulacion en el frame."""
        if landmarks is None:
            return frame
        h, w, _ = frame.shape
        lm_data = self.get_landmark(landmarks, landmark_name)
        if lm_data and lm_data[3] > 0.3:
            x = int(lm_data[0] * w) + 10
            y = int(lm_data[1] * h) - 10
            cv2.putText(frame, f"{angle_value:.0f}g", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def draw_angle(self, frame, angle_value, position, label="", color=(0, 255, 0)):
        """Dibuja angulo en posicion fija del frame"""
        text = f"{label}: {angle_value:.1f} grados"
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def draw_feedback(self, frame, feedback_list, start_y=30):
        """Dibuja lista de feedback en el frame"""
        y = start_y
        for feedback in feedback_list:
            if "[OK]" in feedback:
                color = (0, 255, 0)
            elif "[WARN]" in feedback:
                color = (0, 165, 255)
            elif "[ERROR]" in feedback:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            cv2.putText(frame, feedback, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y += 30
        return frame


class SquatAnalyzer:
    """Analizador especifico para sentadillas"""

    KNEE_ANGLE_RANGE = (75, 120)
    BACK_ANGLE_RANGE = (-45, 45)

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

        left_hip      = self.analyzer.get_landmark(landmarks, 'left_hip')
        left_knee     = self.analyzer.get_landmark(landmarks, 'left_knee')
        left_ankle    = self.analyzer.get_landmark(landmarks, 'left_ankle')
        left_shoulder = self.analyzer.get_landmark(landmarks, 'left_shoulder')

        if not all([
            left_hip      and left_hip[3]      > 0.5,
            left_knee     and left_knee[3]     > 0.5,
            left_ankle    and left_ankle[3]    > 0.5,
            left_shoulder and left_shoulder[3] > 0.5,
        ]):
            results['feedback'].append("[ERROR] No se detectan puntos suficientes")
            return results

        knee_angle = self.analyzer.calculate_angle(left_hip, left_knee, left_ankle)
        back_angle = self.analyzer.calculate_back_angle(left_shoulder, left_hip)

        results['knee_angle'] = knee_angle
        results['back_angle'] = back_angle

        if knee_angle < self.KNEE_ANGLE_RANGE[0]:
            results['feedback'].append(
                f"[WARN] Rodilla insuficiente ({knee_angle:.1f} grados). Baja mas")
            results['overall_score'] -= 15
        elif knee_angle > self.KNEE_ANGLE_RANGE[1]:
            results['feedback'].append(
                f"[WARN] Flexion excesiva ({knee_angle:.1f} grados). Sube un poco")
            results['overall_score'] -= 10
        else:
            results['feedback'].append(f"[OK] Rodilla correcta ({knee_angle:.1f} grados)")

        if abs(back_angle) > 50:
            results['feedback'].append(
                f"[ERROR] PELIGRO: Espalda muy inclinada ({back_angle:.1f} grados). Endereza")
            results['overall_score'] -= 30
        elif abs(back_angle) > 35:
            results['feedback'].append(
                f"[WARN] Espalda algo inclinada ({back_angle:.1f} grados). Puedes enderezar")
            results['overall_score'] -= 10
        else:
            results['feedback'].append(f"[OK] Espalda en buen rango ({back_angle:.1f} grados)")

        right_hip   = self.analyzer.get_landmark(landmarks, 'right_hip')
        right_knee  = self.analyzer.get_landmark(landmarks, 'right_knee')
        right_ankle = self.analyzer.get_landmark(landmarks, 'right_ankle')

        if all([
            right_hip   and right_hip[3]   > 0.5,
            right_knee  and right_knee[3]  > 0.5,
            right_ankle and right_ankle[3] > 0.5,
        ]):
            right_knee_angle = self.analyzer.calculate_angle(right_hip, right_knee, right_ankle)
            angle_diff = abs(knee_angle - right_knee_angle)
            if angle_diff > 10:
                results['feedback'].append(
                    f"[WARN] Rodillas desalineadas (diff: {angle_diff:.1f} grados)")
                results['overall_score'] -= 15

        results['overall_score'] = max(0, results['overall_score'])
        return results


class PushupAnalyzer:
    """
    Analizador especifico para flexiones (push-ups).

    Metricas:
      - elbow_angle (shoulder-elbow-wrist): flexion del codo (70-120 deg)
      - body_alignment (shoulder-hip-ankle): alineacion del cuerpo en plank
        (~170-180 deg = cuerpo recto). Menor de 155 deg = cadera alta o hundida.
    """

    ELBOW_ANGLE_RANGE = (70, 120)
    BODY_ALIGNMENT_MIN = 155

    def __init__(self):
        self.analyzer = PoseAnalyzer()

    def analyze(self, frame, landmarks) -> Dict:
        """Analiza forma de flexion"""
        results = {
            'elbow_angle': 0,
            'body_alignment': 0,
            'feedback': [],
            'overall_score': 100
        }

        if landmarks is None:
            return results

        left_shoulder = self.analyzer.get_landmark(landmarks, 'left_shoulder')
        left_elbow    = self.analyzer.get_landmark(landmarks, 'left_elbow')
        left_wrist    = self.analyzer.get_landmark(landmarks, 'left_wrist')
        left_hip      = self.analyzer.get_landmark(landmarks, 'left_hip')
        left_ankle    = self.analyzer.get_landmark(landmarks, 'left_ankle')

        if not all([
            left_shoulder and left_shoulder[3] > 0.5,
            left_elbow    and left_elbow[3]    > 0.5,
            left_wrist    and left_wrist[3]    > 0.5,
            left_hip      and left_hip[3]      > 0.5,
        ]):
            results['feedback'].append("[ERROR] No se detectan puntos suficientes")
            return results

        # Angulo de codo
        elbow_angle = self.analyzer.calculate_angle(left_shoulder, left_elbow, left_wrist)
        results['elbow_angle'] = elbow_angle

        if elbow_angle < self.ELBOW_ANGLE_RANGE[0]:
            results['feedback'].append(
                f"[WARN] Flexion insuficiente ({elbow_angle:.1f} grados). Baja mas")
            results['overall_score'] -= 15
        elif elbow_angle > self.ELBOW_ANGLE_RANGE[1]:
            results['feedback'].append(
                f"[WARN] Flexion excesiva ({elbow_angle:.1f} grados). Sube un poco")
            results['overall_score'] -= 10
        else:
            results['feedback'].append(f"[OK] Codo correcto ({elbow_angle:.1f} grados)")

        # Alineacion corporal (shoulder-hip-ankle)
        if left_ankle and left_ankle[3] > 0.3:
            body_alignment = self.analyzer.calculate_angle(left_shoulder, left_hip, left_ankle)
            results['body_alignment'] = body_alignment

            if body_alignment < 140:
                results['feedback'].append(
                    f"[ERROR] Cuerpo muy desalineado ({body_alignment:.1f} grados). Activa el core")
                results['overall_score'] -= 25
            elif body_alignment < self.BODY_ALIGNMENT_MIN:
                results['feedback'].append(
                    f"[WARN] Cadera algo desalineada ({body_alignment:.1f} grados). Activa el core")
                results['overall_score'] -= 10
            else:
                results['feedback'].append(f"[OK] Cuerpo alineado ({body_alignment:.1f} grados)")
        else:
            results['feedback'].append("[WARN] Tobillo no visible - verifica alineacion")

        results['overall_score'] = max(0, results['overall_score'])
        return results


class DeadliftAnalyzer:
    """
    Analizador especifico para peso muerto.

    Metricas:
      - hip_angle (shoulder-hip-knee): bisagra de cadera.
        Standing ~160-180 deg | Bending ~70-130 deg.
      - knee_angle (hip-knee-ankle): flexion de rodilla (20-80 deg en fase baja).
      - back_lean: deteccion de espalda redondeada.
        Compara inclinacion tronco (shoulder->hip) vs muslo (hip->knee).
        Si el tronco cae mucho mas que el muslo hay redondeo.
    """

    HIP_ANGLE_RANGE = (60, 170)
    KNEE_ANGLE_RANGE = (20, 80)
    ROUNDING_THRESHOLD = 30

    def __init__(self):
        self.analyzer = PoseAnalyzer()

    def analyze(self, frame, landmarks) -> Dict:
        """Analiza forma de peso muerto"""
        results = {
            'hip_angle': 0,
            'knee_angle': 0,
            'back_lean': 0,
            'feedback': [],
            'overall_score': 100
        }

        if landmarks is None:
            return results

        shoulder = self.analyzer.get_landmark(landmarks, 'left_shoulder')
        hip      = self.analyzer.get_landmark(landmarks, 'left_hip')
        knee     = self.analyzer.get_landmark(landmarks, 'left_knee')
        ankle    = self.analyzer.get_landmark(landmarks, 'left_ankle')

        if not all([
            shoulder and shoulder[3] > 0.5,
            hip      and hip[3]      > 0.5,
            knee     and knee[3]     > 0.5,
            ankle    and ankle[3]    > 0.5,
        ]):
            results['feedback'].append("[ERROR] No se detectan puntos suficientes")
            return results

        # Bisagra de cadera (shoulder-hip-knee)
        hip_angle = self.analyzer.calculate_angle(shoulder, hip, knee)
        results['hip_angle'] = hip_angle

        if hip_angle < self.HIP_ANGLE_RANGE[0]:
            results['feedback'].append(
                f"[ERROR] PELIGRO: Cadera demasiado cerrada ({hip_angle:.1f} grados). Extiende mas")
            results['overall_score'] -= 30
        else:
            results['feedback'].append(
                f"[OK] Bisagra de cadera correcta ({hip_angle:.1f} grados)")

        # Angulo de rodilla (hip-knee-ankle)
        knee_angle = self.analyzer.calculate_angle(hip, knee, ankle)
        results['knee_angle'] = knee_angle

        if knee_angle < self.KNEE_ANGLE_RANGE[0]:
            results['feedback'].append(
                f"[WARN] Rodilla muy flexionada ({knee_angle:.1f} grados)")
            results['overall_score'] -= 15
        elif knee_angle > self.KNEE_ANGLE_RANGE[1]:
            if hip_angle < 130:
                results['feedback'].append(
                    f"[WARN] Rodilla muy extendida en fase baja ({knee_angle:.1f} grados). "
                    f"Flexiona un poco")
                results['overall_score'] -= 10
            else:
                results['feedback'].append(f"[OK] Rodilla correcta ({knee_angle:.1f} grados)")
        else:
            results['feedback'].append(f"[OK] Rodilla correcta ({knee_angle:.1f} grados)")

        # Deteccion de espalda redondeada
        torso_lean = abs(self.analyzer.calculate_back_angle(shoulder, hip))
        thigh_lean = abs(self.analyzer.calculate_back_angle(hip, knee))
        rounding = torso_lean - thigh_lean
        results['back_lean'] = rounding

        if rounding > self.ROUNDING_THRESHOLD + 20:
            results['feedback'].append(
                f"[ERROR] PELIGRO: Espalda muy redondeada. Eleva el pecho y activa lumbar")
            results['overall_score'] -= 40
        elif rounding > self.ROUNDING_THRESHOLD:
            results['feedback'].append(
                f"[WARN] Espalda algo redondeada. Manten pecho arriba")
            results['overall_score'] -= 15
        else:
            results['feedback'].append(f"[OK] Espalda neutra")

        results['overall_score'] = max(0, results['overall_score'])
        return results
