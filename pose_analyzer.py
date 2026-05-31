"""
pose_analyzer.py
Modulo core para analisis de forma fisica en ejercicios.
Disenado para videos tomados de PERFIL (lateral).
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
    """Analizador de poses corporales. Optimizado para vista de perfil."""

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

    def get_best_side(self, landmarks):
        """
        Detecta que lado del cuerpo es mas visible en el frame.
        En un video de perfil, un lado tendra visibilidad mucho mayor.
        Retorna 'left' o 'right'.
        """
        if landmarks is None:
            return 'left'

        key_points = ['shoulder', 'hip', 'knee', 'ankle']
        left_vis  = 0.0
        right_vis = 0.0

        for pt in key_points:
            lm_l = self.get_landmark(landmarks, f'left_{pt}')
            lm_r = self.get_landmark(landmarks, f'right_{pt}')
            left_vis  += lm_l[3] if lm_l else 0.0
            right_vis += lm_r[3] if lm_r else 0.0

        return 'left' if left_vis >= right_vis else 'right'

    def get_side_landmark(self, landmarks, point, side=None):
        """
        Obtiene un landmark del lado mas visible (o del lado especificado).
        Ejemplo: get_side_landmark(lm, 'knee') -> left_knee o right_knee
        segun cual lado sea mas visible en el frame.
        """
        if side is None:
            side = self.get_best_side(landmarks)
        return self.get_landmark(landmarks, f'{side}_{point}')

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
        0 grados = columna vertical | 90 grados = horizontal.
        En vista de perfil este angulo es muy preciso.
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
    """
    Analizador de sentadillas.
    Vista recomendada: PERFIL (lateral). El angulo de rodilla y espalda
    son precisos cuando la camara esta a 90 grados del plano de movimiento.
    """

    KNEE_ANGLE_RANGE = (75, 120)
    BACK_ANGLE_RANGE = (-45, 45)

    def __init__(self):
        self.analyzer = PoseAnalyzer()

    def analyze(self, frame, landmarks) -> Dict:
        """
        Analiza forma de sentadilla en vista de perfil.
        Detecta automaticamente si el lado visible es izquierdo o derecho.
        """
        results = {
            'knee_angle': 0,
            'back_angle': 0,
            'side': 'left',
            'feedback': [],
            'overall_score': 100
        }

        if landmarks is None:
            return results

        # Detectar lado mas visible (perfil izq o derecho)
        side = self.analyzer.get_best_side(landmarks)
        results['side'] = side

        hip      = self.analyzer.get_side_landmark(landmarks, 'hip', side)
        knee     = self.analyzer.get_side_landmark(landmarks, 'knee', side)
        ankle    = self.analyzer.get_side_landmark(landmarks, 'ankle', side)
        shoulder = self.analyzer.get_side_landmark(landmarks, 'shoulder', side)

        # Verificar visibilidad minima
        if not all([
            hip      and hip[3]      > 0.5,
            knee     and knee[3]     > 0.5,
            ankle    and ankle[3]    > 0.5,
            shoulder and shoulder[3] > 0.5,
        ]):
            results['feedback'].append("[ERROR] No se detectan puntos suficientes. "
                                       "Asegurate de estar de perfil y cuerpo completo visible")
            return results

        knee_angle = self.analyzer.calculate_angle(hip, knee, ankle)
        back_angle = self.analyzer.calculate_back_angle(shoulder, hip)

        results['knee_angle'] = knee_angle
        results['back_angle'] = back_angle

        # Validar rodilla
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

        # Validar espalda
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

        # Simetria de rodillas (usa ambos lados)
        opp_side = 'right' if side == 'left' else 'left'
        opp_hip    = self.analyzer.get_side_landmark(landmarks, 'hip', opp_side)
        opp_knee   = self.analyzer.get_side_landmark(landmarks, 'knee', opp_side)
        opp_ankle  = self.analyzer.get_side_landmark(landmarks, 'ankle', opp_side)

        if all([
            opp_hip   and opp_hip[3]   > 0.4,
            opp_knee  and opp_knee[3]  > 0.4,
            opp_ankle and opp_ankle[3] > 0.4,
        ]):
            opp_knee_angle = self.analyzer.calculate_angle(opp_hip, opp_knee, opp_ankle)
            angle_diff = abs(knee_angle - opp_knee_angle)
            if angle_diff > 10:
                results['feedback'].append(
                    f"[WARN] Rodillas desalineadas (diff: {angle_diff:.1f} grados)")
                results['overall_score'] -= 15

        results['overall_score'] = max(0, results['overall_score'])
        return results


class PushupAnalyzer:
    """
    Analizador de flexiones (push-ups).
    Vista recomendada: PERFIL (lateral). Permite medir alineacion
    del cuerpo (plank) y angulo de codo con precision.
    """

    ELBOW_ANGLE_RANGE = (70, 120)
    BODY_ALIGNMENT_MIN = 155  # angulo shoulder-hip-ankle < 155 = cadera desalineada

    def __init__(self):
        self.analyzer = PoseAnalyzer()

    def analyze(self, frame, landmarks) -> Dict:
        """
        Analiza forma de flexion en vista de perfil.
        Detecta automaticamente si el lado visible es izquierdo o derecho.
        """
        results = {
            'elbow_angle': 0,
            'body_alignment': 0,
            'side': 'left',
            'feedback': [],
            'overall_score': 100
        }

        if landmarks is None:
            return results

        side = self.analyzer.get_best_side(landmarks)
        results['side'] = side

        shoulder = self.analyzer.get_side_landmark(landmarks, 'shoulder', side)
        elbow    = self.analyzer.get_side_landmark(landmarks, 'elbow', side)
        wrist    = self.analyzer.get_side_landmark(landmarks, 'wrist', side)
        hip      = self.analyzer.get_side_landmark(landmarks, 'hip', side)
        ankle    = self.analyzer.get_side_landmark(landmarks, 'ankle', side)

        if not all([
            shoulder and shoulder[3] > 0.5,
            elbow    and elbow[3]    > 0.5,
            wrist    and wrist[3]    > 0.5,
            hip      and hip[3]      > 0.5,
        ]):
            results['feedback'].append("[ERROR] No se detectan puntos suficientes. "
                                       "Asegurate de estar de perfil y cuerpo completo visible")
            return results

        # Angulo de codo (shoulder-elbow-wrist)
        elbow_angle = self.analyzer.calculate_angle(shoulder, elbow, wrist)
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

        # Alineacion corporal (shoulder-hip-ankle): debe ser ~170-180 en plank
        if ankle and ankle[3] > 0.3:
            body_alignment = self.analyzer.calculate_angle(shoulder, hip, ankle)
            results['body_alignment'] = body_alignment

            if body_alignment < 140:
                results['feedback'].append(
                    f"[ERROR] Cuerpo muy desalineado ({body_alignment:.1f} grados). "
                    f"Activa el core y mantén cuerpo recto")
                results['overall_score'] -= 25
            elif body_alignment < self.BODY_ALIGNMENT_MIN:
                results['feedback'].append(
                    f"[WARN] Cadera algo desalineada ({body_alignment:.1f} grados). Activa el core")
                results['overall_score'] -= 10
            else:
                results['feedback'].append(f"[OK] Cuerpo alineado ({body_alignment:.1f} grados)")
        else:
            results['feedback'].append("[WARN] Tobillo no visible - verifica alineacion de cadera")

        results['overall_score'] = max(0, results['overall_score'])
        return results


class DeadliftAnalyzer:
    """
    Analizador de peso muerto.
    Vista recomendada: PERFIL (lateral). La vista lateral es CRITICA
    para detectar redondeo de espalda y correcta bisagra de cadera.
    """

    HIP_ANGLE_RANGE = (60, 170)
    KNEE_ANGLE_RANGE = (20, 80)
    ROUNDING_THRESHOLD = 30

    def __init__(self):
        self.analyzer = PoseAnalyzer()

    def analyze(self, frame, landmarks) -> Dict:
        """
        Analiza forma de peso muerto en vista de perfil.
        Detecta automaticamente si el lado visible es izquierdo o derecho.
        """
        results = {
            'hip_angle': 0,
            'knee_angle': 0,
            'back_lean': 0,
            'side': 'left',
            'feedback': [],
            'overall_score': 100
        }

        if landmarks is None:
            return results

        side = self.analyzer.get_best_side(landmarks)
        results['side'] = side

        shoulder = self.analyzer.get_side_landmark(landmarks, 'shoulder', side)
        hip      = self.analyzer.get_side_landmark(landmarks, 'hip', side)
        knee     = self.analyzer.get_side_landmark(landmarks, 'knee', side)
        ankle    = self.analyzer.get_side_landmark(landmarks, 'ankle', side)

        if not all([
            shoulder and shoulder[3] > 0.5,
            hip      and hip[3]      > 0.5,
            knee     and knee[3]     > 0.5,
            ankle    and ankle[3]    > 0.5,
        ]):
            results['feedback'].append("[ERROR] No se detectan puntos suficientes. "
                                       "Asegurate de estar de perfil y cuerpo completo visible")
            return results

        # Bisagra de cadera (shoulder-hip-knee) — indicador de fase del movimiento.
        # En el peso muerto el angulo varia de ~60 grados (posicion baja) a ~175 grados (lockout).
        # No se penaliza: la critica es la espalda neutra, no el angulo de cadera.
        hip_angle = self.analyzer.calculate_angle(shoulder, hip, knee)
        results['hip_angle'] = hip_angle
        # Solo mostrar si el angulo es muy extremo (< 30) lo que podria indicar mala camara
        if hip_angle < 30:
            results['feedback'].append(
                f"[WARN] Angulo de cadera muy cerrado ({hip_angle:.1f} grados). "
                f"Verifica que el video sea de perfil")
        else:
            results['feedback'].append(
                f"[OK] Bisagra de cadera ({hip_angle:.1f} grados)")

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

        # Deteccion de espalda redondeada (comparar inclinacion tronco vs muslo)
        torso_lean = abs(self.analyzer.calculate_back_angle(shoulder, hip))
        thigh_lean = abs(self.analyzer.calculate_back_angle(hip, knee))
        rounding   = torso_lean - thigh_lean
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


class FrontViewAnalyzer:
    """
    Analiza ejercicios desde VISTA FRONTAL (de frente a la camara).
    Mide: ancho de pies vs hombros, valgus de rodilla,
    ancho de manos (flexiones), nivel de caderas y hombros.
    """

    FOOT_SHOULDER_RATIO  = (0.8, 1.5)   # rango aceptable pies/hombros
    HAND_SHOULDER_RATIO  = (0.8, 1.5)   # rango aceptable manos/hombros
    FOOT_HIP_RATIO       = (0.7, 2.0)   # rango aceptable pies/caderas (deadlift)
    VALGUS_THRESHOLD     = 0.04          # desplazamiento max de rodilla hacia adentro (normalizado)
    LEVEL_THRESHOLD      = 0.04          # diferencia max de altura entre puntos simetricos

    def __init__(self):
        self.analyzer = PoseAnalyzer()

    def _width(self, lm1, lm2):
        """Distancia horizontal entre dos landmarks."""
        if lm1 is None or lm2 is None:
            return None
        return abs(lm1[0] - lm2[0])

    def analyze(self, frame, landmarks, exercise_type='squat') -> Dict:
        """Punto de entrada unificado. exercise_type: 'squat', 'pushup', 'deadlift'."""
        if landmarks is None:
            return {'feedback': ["[ERROR] No se detecto pose"], 'overall_score': 100}
        if exercise_type == 'squat':
            return self.analyze_squat(frame, landmarks)
        elif exercise_type == 'pushup':
            return self.analyze_pushup(frame, landmarks)
        elif exercise_type == 'deadlift':
            return self.analyze_deadlift(frame, landmarks)
        return {'feedback': [], 'overall_score': 100}

    def analyze_squat(self, frame, landmarks) -> Dict:
        """
        Sentadilla desde frente:
        - Ancho de pies vs hombros (debe ser ~1x ancho de hombros)
        - Valgus de rodilla (rodillas cayendo hacia adentro)
        - Nivel de caderas
        """
        results = {'feedback': [], 'overall_score': 100, 'foot_ratio': 0}
        a = self.analyzer

        ls = a.get_landmark(landmarks, 'left_shoulder')
        rs = a.get_landmark(landmarks, 'right_shoulder')
        lk = a.get_landmark(landmarks, 'left_knee')
        rk = a.get_landmark(landmarks, 'right_knee')
        la = a.get_landmark(landmarks, 'left_ankle')
        ra = a.get_landmark(landmarks, 'right_ankle')
        lh = a.get_landmark(landmarks, 'left_hip')
        rh = a.get_landmark(landmarks, 'right_hip')

        vis_base = all([
            ls and ls[3] > 0.5, rs and rs[3] > 0.5,
            la and la[3] > 0.5, ra and ra[3] > 0.5,
        ])
        if not vis_base:
            results['feedback'].append(
                "[ERROR] Asegurate de estar de frente con cuerpo completo visible")
            return results

        # Ancho pies vs hombros
        sw = self._width(ls, rs)
        fw = self._width(la, ra)
        if sw and fw and sw > 0:
            ratio = fw / sw
            results['foot_ratio'] = ratio
            if ratio < self.FOOT_SHOULDER_RATIO[0]:
                results['feedback'].append(
                    f"[WARN] Pies muy juntos ({ratio:.2f}x hombros). Abre los pies a ancho de hombros")
                results['overall_score'] -= 15
            elif ratio > self.FOOT_SHOULDER_RATIO[1]:
                results['feedback'].append(
                    f"[WARN] Pies muy abiertos ({ratio:.2f}x hombros). Cierra un poco la postura")
                results['overall_score'] -= 10
            else:
                results['feedback'].append(
                    f"[OK] Ancho de pies correcto ({ratio:.2f}x ancho de hombros)")

        # Valgus de rodilla
        if all([lk and lk[3] > 0.4, la and la[3] > 0.4,
                rk and rk[3] > 0.4, ra and ra[3] > 0.4]):
            # Rodilla izquierda: si lk.x > la.x, la rodilla cae hacia la derecha (adentro)
            left_valgus  = lk[0] - la[0]
            # Rodilla derecha: si rk.x < ra.x, la rodilla cae hacia la izquierda (adentro)
            right_valgus = ra[0] - rk[0]
            max_valgus = max(left_valgus, right_valgus)

            if max_valgus > self.VALGUS_THRESHOLD:
                results['feedback'].append(
                    "[ERROR] Rodillas cayendo hacia adentro (valgus). "
                    "Empuja rodillas hacia afuera alineadas con el pie")
                results['overall_score'] -= 25
            else:
                results['feedback'].append("[OK] Rodillas bien alineadas sobre los pies")

        # Nivel de caderas
        if lh and rh and lh[3] > 0.4 and rh[3] > 0.4:
            diff = abs(lh[1] - rh[1])
            if diff > self.LEVEL_THRESHOLD:
                results['feedback'].append(
                    "[WARN] Caderas desniveladas. Mantén ambas caderas a la misma altura")
                results['overall_score'] -= 10
            else:
                results['feedback'].append("[OK] Caderas niveladas")

        results['overall_score'] = max(0, results['overall_score'])
        return results

    def analyze_pushup(self, frame, landmarks) -> Dict:
        """
        Flexion desde frente:
        - Ancho de manos vs hombros (manos a ancho de hombros o ligeramente mas)
        - Nivel de hombros
        - Codos hacia atras (no en T)
        """
        results = {'feedback': [], 'overall_score': 100, 'hand_ratio': 0}
        a = self.analyzer

        ls = a.get_landmark(landmarks, 'left_shoulder')
        rs = a.get_landmark(landmarks, 'right_shoulder')
        lw = a.get_landmark(landmarks, 'left_wrist')
        rw = a.get_landmark(landmarks, 'right_wrist')
        le = a.get_landmark(landmarks, 'left_elbow')
        re = a.get_landmark(landmarks, 'right_elbow')

        if not all([ls and ls[3] > 0.5, rs and rs[3] > 0.5,
                    lw and lw[3] > 0.5, rw and rw[3] > 0.5]):
            results['feedback'].append(
                "[ERROR] Asegurate de estar de frente con brazos completamente visibles")
            return results

        # Ancho manos vs hombros
        sw = self._width(ls, rs)
        hw = self._width(lw, rw)
        if sw and hw and sw > 0:
            ratio = hw / sw
            results['hand_ratio'] = ratio
            if ratio < self.HAND_SHOULDER_RATIO[0]:
                results['feedback'].append(
                    f"[WARN] Manos muy juntas ({ratio:.2f}x hombros). Separa mas las manos")
                results['overall_score'] -= 15
            elif ratio > self.HAND_SHOULDER_RATIO[1]:
                results['feedback'].append(
                    f"[WARN] Manos muy abiertas ({ratio:.2f}x hombros). Cierra un poco")
                results['overall_score'] -= 10
            else:
                results['feedback'].append(
                    f"[OK] Ancho de manos correcto ({ratio:.2f}x ancho de hombros)")

        # Nivel de hombros
        if ls[3] > 0.4 and rs[3] > 0.4:
            diff = abs(ls[1] - rs[1])
            if diff > self.LEVEL_THRESHOLD:
                results['feedback'].append(
                    "[WARN] Hombros desnivelados. Mantén hombros a la misma altura")
                results['overall_score'] -= 15
            else:
                results['feedback'].append("[OK] Hombros nivelados")

        # Codos: no deben estar en T (a 90 grados del cuerpo)
        # Si el codo esta mas afuera que la muneca, los codos estan muy abiertos
        if le and re and le[3] > 0.4 and re[3] > 0.4:
            left_flare  = le[0] - lw[0]   # positivo = codo mas a la izquierda que muneca
            right_flare = rw[0] - re[0]   # positivo = codo mas a la derecha que muneca
            if left_flare > 0.06 or right_flare > 0.06:
                results['feedback'].append(
                    "[WARN] Codos muy abiertos (posicion T). Junta los codos mas cerca del cuerpo")
                results['overall_score'] -= 10
            else:
                results['feedback'].append("[OK] Posicion de codos correcta")

        results['overall_score'] = max(0, results['overall_score'])
        return results

    def analyze_deadlift(self, frame, landmarks) -> Dict:
        """
        Peso muerto desde frente:
        - Ancho de pies vs caderas (convencional: ancho de caderas)
        - Nivel de caderas durante el levantamiento
        - Nivel de hombros
        """
        results = {'feedback': [], 'overall_score': 100, 'foot_ratio': 0}
        a = self.analyzer

        ls = a.get_landmark(landmarks, 'left_shoulder')
        rs = a.get_landmark(landmarks, 'right_shoulder')
        lh = a.get_landmark(landmarks, 'left_hip')
        rh = a.get_landmark(landmarks, 'right_hip')
        la = a.get_landmark(landmarks, 'left_ankle')
        ra = a.get_landmark(landmarks, 'right_ankle')

        if not all([la and la[3] > 0.5, ra and ra[3] > 0.5,
                    lh and lh[3] > 0.5, rh and rh[3] > 0.5]):
            results['feedback'].append(
                "[ERROR] Asegurate de estar de frente con cuerpo completo visible")
            return results

        # Ancho de pies vs caderas
        hiw = self._width(lh, rh)
        fw  = self._width(la, ra)
        if hiw and fw and hiw > 0:
            ratio = fw / hiw
            results['foot_ratio'] = ratio
            if ratio < self.FOOT_HIP_RATIO[0]:
                results['feedback'].append(
                    f"[WARN] Pies muy juntos ({ratio:.2f}x caderas). "
                    f"Abre los pies a ancho de caderas")
                results['overall_score'] -= 15
            elif ratio > self.FOOT_HIP_RATIO[1]:
                results['feedback'].append(
                    f"[WARN] Pies muy abiertos ({ratio:.2f}x caderas). "
                    f"Estilo sumo requiere tecnica especifica")
                results['overall_score'] -= 5
            else:
                results['feedback'].append(
                    f"[OK] Ancho de pies correcto ({ratio:.2f}x ancho de caderas)")

        # Nivel de caderas
        if lh[3] > 0.4 and rh[3] > 0.4:
            diff = abs(lh[1] - rh[1])
            if diff > self.LEVEL_THRESHOLD:
                results['feedback'].append(
                    "[ERROR] Caderas desniveladas al levantar. "
                    "Extiende ambas piernas por igual")
                results['overall_score'] -= 20
            else:
                results['feedback'].append("[OK] Caderas niveladas")

        # Nivel de hombros
        if ls and rs and ls[3] > 0.4 and rs[3] > 0.4:
            diff = abs(ls[1] - rs[1])
            if diff > self.LEVEL_THRESHOLD:
                results['feedback'].append(
                    "[WARN] Hombros desnivelados. Mantén hombros a la misma altura")
                results['overall_score'] -= 10
            else:
                results['feedback'].append("[OK] Hombros nivelados")

        results['overall_score'] = max(0, results['overall_score'])
        return results
