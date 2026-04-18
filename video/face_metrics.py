import time

class HeadPoseAnalyzer:
    """
    Analizador 2D de Movimiento de Cabeza (Tiempo Real).
    Alineado al estándar OpenOPAF: cámara frontal a la altura de los ojos,
    sin compensación por ángulos de contrapicado.
    """
    def __init__(self):
        # --- UMBRALES FRONTALES PUROS ---
        # Se evalúa la posición geométrica real de la nariz respecto a las orejas.
        # En una toma frontal nivelada, la nariz suele estar ligeramente por debajo de las orejas.
        
        self.ratio_down_enter = 0.35
        self.ratio_down_exit = 0.25
        self.ratio_up_enter = -0.15
        self.ratio_up_exit = -0.05

        self.head_state = "NORMAL"
        self.violation_start_time = None
        
        self.time_limit = 2.0 

    def process_head_pose(self, face_landmarks, pose_landmarks, mp_pose, frame_shape=None):
        instant_state = "NORMAL"
        
        if face_landmarks:
            lm = face_landmarks.landmark

            # ==========================================
            # MACRO-MOVIMIENTO 2D (Nariz vs Orejas)
            # ==========================================
            nose = lm[1]
            l_ear = lm[234]
            r_ear = lm[454]

            face_width = abs(l_ear.x - r_ear.x)
            if face_width < 0.01: face_width = 0.01

            avg_ear_y = (l_ear.y + r_ear.y) / 2.0
            
            # Cálculo de Inclinación 2D (Pitch ratio)
            pitch_ratio = (nose.y - avg_ear_y) / face_width

            if self.head_state == "DOWN":
                if pitch_ratio < self.ratio_down_exit: self.head_state = "NORMAL"
            elif self.head_state == "UP":
                if pitch_ratio > self.ratio_up_exit: self.head_state = "NORMAL"
            else:
                if pitch_ratio > self.ratio_down_enter: self.head_state = "DOWN"
                elif pitch_ratio < self.ratio_up_enter: self.head_state = "UP"
                    
            instant_state = self.head_state

        elif pose_landmarks:
            
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            l_ear = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            r_ear = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            
            ear_mean_y = (l_ear.y + r_ear.y) / 2.0
            face_width = abs(l_ear.x - r_ear.x)
            if face_width < 0.01: face_width = 0.01
            
            pitch_ratio = (nose.y - ear_mean_y) / face_width
            
            # Usar los umbrales frontales sin alteraciones
            if self.head_state == "DOWN":
                if pitch_ratio < self.ratio_down_exit: self.head_state = "NORMAL"
            elif self.head_state == "UP":
                if pitch_ratio > self.ratio_up_exit: self.head_state = "NORMAL"
            else:
                if pitch_ratio > self.ratio_down_enter: self.head_state = "DOWN"
                elif pitch_ratio < self.ratio_up_enter: self.head_state = "UP"
                
            instant_state = self.head_state
            
        else:
            self.violation_start_time = None
            return {"is_looking": False, "error_type": "Rostro no detectado"}

        # ==========================================
        # APLICACIÓN DEL BUFFER DE TIEMPO (2 SEGUNDOS)
        # ==========================================
        if instant_state == "NORMAL":
            self.violation_start_time = None
            return {"is_looking": True, "error_type": ""}
        else:
            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                return {"is_looking": True, "error_type": ""}
            elif (time.time() - self.violation_start_time) > self.time_limit:
                # Pasaron 2 segundos
                if instant_state == "DOWN":
                    error_msg = "MIRADA AL SUELO (Oculta)"
                else:
                    error_msg = "MIRADA AL TECHO (Duda)"
                    
                return {"is_looking": False, "error_type": error_msg}
            else:
                # Sigue dentro de los 2 segundos de gracia
                return {"is_looking": True, "error_type": ""}
