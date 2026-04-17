import numpy as np
import time

class GestureAnalyzer:


    def __init__(self):
        
        self.angle_threshold = 6.0

        # Tiempo máximo permitido sin gesticulación
        self.inactivity_limit = 6.0

        # Suavizado simple para reducir jitter
        self.smoothed_angles = {"left": None, "right": None}

        # Anclaje para evitar comparar frame-a-frame y eliminar el retraso
        self.reference_angles = {"left": None, "right": None}

        # Estado global
        self.last_valid_gesture_time = time.time()
        self.is_inactive = False

    def calculate_angle_2d(self, p1, p2, p3):
        """Calcula el ángulo 2D hombro-codo-muñeca."""
        v1 = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
        v2 = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))

    def _smooth_angle(self, current_angle, side):
        if self.smoothed_angles[side] is None:
            self.smoothed_angles[side] = current_angle
        else:
            self.smoothed_angles[side] = (
                0.7 * self.smoothed_angles[side] + 0.3 * current_angle
            )
        return self.smoothed_angles[side]

    def _has_valid_arm_motion(self, current_angle, side):
        """
        Detecta cambio angular usando un ancla, eliminando el retraso de detección.
        """
        smoothed = self._smooth_angle(current_angle, side)

        if self.reference_angles[side] is None:
            self.reference_angles[side] = smoothed
            return False

        # Comparamos contra el ancla, no contra el frame anterior
        diff = abs(smoothed - self.reference_angles[side])

        if diff >= self.angle_threshold:
            
            self.reference_angles[side] = smoothed
            return True

        
        self.reference_angles[side] = 0.98 * self.reference_angles[side] + 0.02 * smoothed

        return False

    def _check_inactivity(self):
        if (time.time() - self.last_valid_gesture_time) >= self.inactivity_limit:
            self.is_inactive = True

    def _build_response(self, gesture_detected):
        return {
            "gesture_detected": gesture_detected,
            "inactivity_alert": self.is_inactive,
        }

    def process_landmarks(self, pose_landmarks, pose_world_landmarks, mp_pose):
        """
        Procesa landmarks.
        Requiere pose_world_landmarks para el filtro de rotación.
        """
        if pose_landmarks is None:
            self._check_inactivity()
            return self._build_response(False)

        # -------------------------
        # FILTRO DE ROTACIÓN
        # -------------------------
        if pose_world_landmarks:
            l_shldr_3d = pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shldr_3d = pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            z_diff = abs(l_shldr_3d.z - r_shldr_3d.z)

            if z_diff > 0.08:
                self._check_inactivity()
                return self._build_response(False)

        # -------------------------
        # EVALUACIÓN 2D NORMAL
        # -------------------------
        l_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_shldr = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shldr = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        torso_visible = (
            l_shldr.visibility > 0.6 and r_shldr.visibility > 0.6 and
            l_elbow.visibility > 0.6 and r_elbow.visibility > 0.6 and
            l_hip.visibility > 0.5 and r_hip.visibility > 0.5
        )

        if not torso_visible:
            self._check_inactivity()
            return self._build_response(False)

        left_torso_height = l_hip.y - l_shldr.y
        right_torso_height = r_hip.y - r_shldr.y

        upper_limit_left = l_shldr.y + 0.10 * left_torso_height
        lower_limit_left = l_shldr.y + 0.75 * left_torso_height

        upper_limit_right = r_shldr.y + 0.10 * right_torso_height
        lower_limit_right = r_shldr.y + 0.75 * right_torso_height

        l_valid_zone = (
            l_wrist.visibility > 0.7 and
            upper_limit_left < l_wrist.y < lower_limit_left
        )

        r_valid_zone = (
            r_wrist.visibility > 0.7 and
            upper_limit_right < r_wrist.y < lower_limit_right
        )

        l_motion = False
        r_motion = False

        if l_valid_zone:
            l_angle = self.calculate_angle_2d(l_shldr, l_elbow, l_wrist)
            l_motion = self._has_valid_arm_motion(l_angle, "left")
        else:
            self.reference_angles["left"] = None
            self.smoothed_angles["left"] = None

        if r_valid_zone:
            r_angle = self.calculate_angle_2d(r_shldr, r_elbow, r_wrist)
            r_motion = self._has_valid_arm_motion(r_angle, "right")
        else:
            self.reference_angles["right"] = None
            self.smoothed_angles["right"] = None

        gesture_detected = l_motion or r_motion

        if gesture_detected:
            self.last_valid_gesture_time = time.time()
            self.is_inactive = False
        else:
            self._check_inactivity()

        return self._build_response(gesture_detected)


class PostureSwayAnalyzer:


    def __init__(self):
        self.sway_count = 0
        self.sway_window_start = time.time()
        self.last_hip_x = None
        self.trend = 0
        
       
        self.smoothed_center_x = None
        self.sway_threshold = 0.025

        self.is_alert_active = False
        self.alert_start_time = 0
        self.alert_duration = 2.0

    def check_sway(self, pose_landmarks, mp_pose):
        current_time = time.time()

        if self.is_alert_active:
            if (current_time - self.alert_start_time) < self.alert_duration:
                return True
            else:
                self.is_alert_active = False
                self.sway_window_start = current_time
                self.sway_count = 0

        
        if (current_time - self.sway_window_start) >= 6.0:
            self.sway_count = 0
            self.sway_window_start = current_time

        l_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        if l_hip.visibility < 0.6 or r_hip.visibility < 0.6:
            return False

        # Centro bruto detectado por MediaPipe
        raw_center_x = (l_hip.x + r_hip.x) / 2.0

        
        
        if self.smoothed_center_x is None:
            self.smoothed_center_x = raw_center_x
        else:
            self.smoothed_center_x = 0.8 * self.smoothed_center_x + 0.2 * raw_center_x

        if self.last_hip_x is None:
            self.last_hip_x = self.smoothed_center_x
            return False

        # Comparamos usando el centro suavizado, no el bruto
        diff = self.smoothed_center_x - self.last_hip_x

        if abs(diff) > self.sway_threshold:
            current_trend = 1 if diff > 0 else -1

            if self.trend == 0:
               
                self.trend = current_trend
            elif current_trend != self.trend:
                # 🌟 
                self.sway_count += 1
                self.trend = current_trend
            
            self.last_hip_x = self.smoothed_center_x

        
        if self.sway_count >= 5:
            self.is_alert_active = True
            self.alert_start_time = current_time
            return True

        return False
        
class BodyOrientationAnalyzer:

    def __init__(self):
        
        self.enter_profile_threshold = 0.22  
        self.exit_profile_threshold = 0.35   
        
        self.is_profile = False 
        
        
        self.profile_start_time = None
        self.time_limit = 2.0  

    def check_orientation(self, pose_landmarks, mp_pose):
        if pose_landmarks is None:
            self.profile_start_time = None
            return {"is_facing": True, "error": None}

        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        l_shldr = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shldr = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # 1. REGLA DE LA ESPALDA (Back to audience)
        if l_shldr.visibility > 0.5 and r_shldr.visibility > 0.5:
            
            if nose.visibility < 0.15:
                return {"is_facing": False, "error": "ESPALDA AL PUBLICO (Cuerpo)"}

        # 2. REGLA DEL PERFIL 
        shoulder_width = abs(l_shldr.x - r_shldr.x)
        torso_height = abs(((l_shldr.y + r_shldr.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0))

        # Evitar errores matemáticos si el usuario se agacha o sale de cuadro
        if torso_height < 0.05:
            return {"is_facing": not self.is_profile, "error": "POSTURA DE PERFIL" if self.is_profile else None}

        ratio = shoulder_width / torso_height

       
        if self.is_profile:
            
            if ratio > self.exit_profile_threshold:
                self.is_profile = False
                self.profile_start_time = None
        else:
           
            if ratio < self.enter_profile_threshold:
                if self.profile_start_time is None:
                    
                    self.profile_start_time = time.time()
                elif (time.time() - self.profile_start_time) > self.time_limit:
                    
                    self.is_profile = True
            else:
                
                self.profile_start_time = None

        if self.is_profile:
            return {"is_facing": False, "error": "POSTURA DE PERFIL (Cuerpo)"}

        return {"is_facing": True, "error": None}

class PostureAmplitudeAnalyzer:

    def __init__(self):
        
        self.closed_posture_limit = 5.0
        self.closed_start_time = time.time()
        self.is_closed_alert = False

    def check_amplitude(self, pose_landmarks, mp_pose):
        if pose_landmarks is None:
            return {"is_closed": False, "error": None}

        l_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_shldr = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shldr = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        
        l_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        # Si no vemos los brazos o los hombros, mantenemos el estado actual
        if l_wrist.visibility < 0.5 or r_wrist.visibility < 0.5 or l_shldr.visibility < 0.5 or r_shldr.visibility < 0.5:
            return {"is_closed": self.is_closed_alert, "error": "POSTURA CONTRAIDA (Brazos pegados)" if self.is_closed_alert else None}

        # Ancho de referencia del cuerpo
        shoulder_width = abs(l_shldr.x - r_shldr.x)
        if shoulder_width < 0.05: shoulder_width = 0.05

        # Distancia máxima horizontal que están ocupando las manos (El Bounding Box)
        hands_width = abs(l_wrist.x - r_wrist.x)
        chest_y = (l_shldr.y + l_hip.y) / 2.0
        
        
        if hands_width < (shoulder_width * 1.2) and (l_wrist.y > chest_y and r_wrist.y > chest_y):
            if not self.is_closed_alert:
                if (time.time() - self.closed_start_time) > self.closed_posture_limit:
                    self.is_closed_alert = True
        else:
            
            self.closed_start_time = time.time()
            self.is_closed_alert = False

        error_msg = "POSTURA CONTRAIDA (T-Rex arms)" if self.is_closed_alert else None
        return {"is_closed": self.is_closed_alert, "error": error_msg}