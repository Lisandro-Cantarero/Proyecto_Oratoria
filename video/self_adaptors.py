import numpy as np
import time
from collections import deque

class SelfAdaptorAnalyzer:
    """
    Módulo para la detección de Adaptadores Corporales y Postura Cerrada.
    Utiliza un enfoque Híbrido (Hands + Pose) para evadir la oclusión al frotar las manos.
    """
    
    def __init__(self):
        # Multiplicadores relativos al ancho de los hombros
        self.face_touch_multiplier = 0.50  
        self.ear_touch_multiplier = 0.50   
        self.hand_rub_multiplier = 0.45    # 🌟 AUMENTADO: Da margen cuando las manos se superponen
        self.pocket_multiplier = 0.30      
        
        # --- Motor de Frecuencia y Duración (Para el PDF) ---
        self.touch_history = deque()
        self.is_touching_now = False
        self.current_touch_start = 0
        self.rep_limit = 3         
        self.time_window = 60.0    
        self.prolonged_limit = 3.0 

        # --- Memoria Cinética MICRO (Fricción) ---
        self.finger_friction_history = deque(maxlen=15) 
        self.friction_std_threshold = 0.015 # 🌟 SENSIBILIDAD AJUSTADA para detectar frote rápido

    def check_adaptors(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, mp_pose):
        res = {
            "pocket": {"active": False, "type": None},
            "face": {"instant_active": False, "report_active": False, "type": None, "instant_type": None}
        }

        if pose_landmarks is None:
            self.is_touching_now = False
            self.current_touch_start = 0
            self.finger_friction_history.clear()
            return res

        # 1. Extraer landmarks del cuerpo 
        lm = pose_landmarks.landmark
        mpl = mp_pose.PoseLandmark

        l_wrist_pose = lm[mpl.LEFT_WRIST]
        r_wrist_pose = lm[mpl.RIGHT_WRIST]
        l_index_pose = lm[mpl.LEFT_INDEX]
        r_index_pose = lm[mpl.RIGHT_INDEX]
        
        l_elbow = lm[mpl.LEFT_ELBOW]
        r_elbow = lm[mpl.RIGHT_ELBOW]
        
        nose = lm[mpl.NOSE]
        l_ear = lm[mpl.LEFT_EAR]
        r_ear = lm[mpl.RIGHT_EAR]
        
        l_shldr = lm[mpl.LEFT_SHOULDER]
        r_shldr = lm[mpl.RIGHT_SHOULDER]
        l_hip = lm[mpl.LEFT_HIP]
        r_hip = lm[mpl.RIGHT_HIP]

        # Validar que el torso está visible
        if l_shldr.visibility < 0.5 or r_shldr.visibility < 0.5 or nose.visibility < 0.5:
            self.is_touching_now = False
            self.current_touch_start = 0
            return res

        shoulder_width = ((l_shldr.x - r_shldr.x)**2 + (l_shldr.y - r_shldr.y)**2)**0.5
        if shoulder_width == 0: shoulder_width = 0.1 

        face_threshold = shoulder_width * self.face_touch_multiplier
        ear_threshold = shoulder_width * self.ear_touch_multiplier

        def dist(p1, p2):
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

        # --- A) POSTURA CERRADA ---
        brazos_ocultos = (l_wrist_pose.visibility < 0.2) and (r_wrist_pose.visibility < 0.2)

        dist_muneca_i_cadera = dist(l_wrist_pose, l_hip)
        dist_muneca_d_cadera = dist(r_wrist_pose, r_hip)
        
        manos_en_bolsillos = (dist_muneca_i_cadera < (shoulder_width * 0.5)) and \
                             (dist_muneca_d_cadera < (shoulder_width * 0.5)) and \
                             (l_wrist_pose.y > l_hip.y - 0.1) 
                             
        manos_caidas = (l_wrist_pose.y > l_hip.y) and (r_wrist_pose.y > r_hip.y)

        # 🌟 LÓGICA DE BRAZOS CRUZADOS CORREGIDA Y ESTABILIZADA 🌟
        dist_cruce_1 = dist(l_wrist_pose, r_elbow)
        dist_cruce_2 = dist(r_wrist_pose, l_elbow)
        
        # 1. Ampliamos la tolerancia geométrica (0.45 -> 0.70) para agarrar el bíceps
        cruce_fisico = (dist_cruce_1 < (shoulder_width * 0.70)) or (dist_cruce_2 < (shoulder_width * 0.70))
        
        # 2. Revisamos ambas muñecas, damos margen a los hombros y protegemos contra fallos
        en_torso = ((l_shldr.y - 0.1) < l_wrist_pose.y < l_hip.y) or ((r_shldr.y - 0.1) < r_wrist_pose.y < r_hip.y)
        
        dist_codos = abs(l_elbow.x - r_elbow.x)
        # 3. Los codos cruzados miden normalmente ~0.95 del ancho de hombros, no 0.75
        codos_pegados_al_frente = (dist_codos < (shoulder_width * 0.95)) and \
                                  (l_elbow.y < l_hip.y) and (r_elbow.y < r_hip.y)

        brazos_cruzados = (cruce_fisico and en_torso) or codos_pegados_al_frente
        condicion_cerrada = manos_en_bolsillos or manos_caidas or brazos_cruzados or brazos_ocultos

        if condicion_cerrada:
            tipo_postura = "POSTURA CERRADA"
            if brazos_cruzados: tipo_postura = "BRAZOS CRUZADOS"
            elif manos_en_bolsillos or brazos_ocultos: tipo_postura = "MANOS OCULTAS/BOLSILLOS"
            elif manos_caidas: tipo_postura = "MANOS CAIDAS"
            res["pocket"] = {"active": True, "type": tipo_postura}

        # --- B) LÓGICA MICRO-CINÉTICA: FROTAMIENTO HÍBRIDO (Malla + Pose) ---
        face_active = False
        face_type = None

        dist_hands = 999.0
        dist_tracking = 999.0

        # PLAN A: Usamos la malla si MediaPipe logra ver ambas manos (ideal para contar con dedos o fricción suave)
        if left_hand_landmarks and right_hand_landmarks:
            l_palm = left_hand_landmarks.landmark[9]
            r_palm = right_hand_landmarks.landmark[9]
            l_tip = left_hand_landmarks.landmark[8]
            r_tip = right_hand_landmarks.landmark[8]
            
            dist_hands = dist(l_palm, r_palm)
            dist_tracking = dist(l_tip, r_tip)
        else:
            # PLAN B (ANTIFALLOS): Si te frotás loco y la oclusión ciega a MediaPipe Hands,
            # saltamos a usar las muñecas de Pose, que nunca desaparecen.
            dist_hands = dist(l_wrist_pose, r_wrist_pose)
            dist_tracking = dist_hands 

        # 1. ¿Están las manos pegadas y debajo de la cara?
        if dist_hands < (shoulder_width * self.hand_rub_multiplier) and (l_wrist_pose.y > nose.y or r_wrist_pose.y > nose.y):
            self.finger_friction_history.append(dist_tracking)
            
            # 2. Comprobación Cinética: Varianza del movimiento (Fricción)
            if len(self.finger_friction_history) == self.finger_friction_history.maxlen:
                std_dev_friction = np.std(self.finger_friction_history)
                
                # Si las manos están juntas pero NO se frotan (ej. contando), la std_dev será casi 0.
                # Si se frotan vigorosamente, la desviación estándar se dispara y rompe el umbral.
                if (std_dev_friction / shoulder_width) > self.friction_std_threshold:
                    face_active, face_type = True, "FROTAR MANOS (ANSIEDAD)"
        else:
            # Si se separan las manos, reseteamos la memoria de fricción
            self.finger_friction_history.clear()

        # --- C) TOQUES EN EL ROSTRO ---
        l_hand_visible_pose = l_wrist_pose.visibility > 0.5
        r_hand_visible_pose = r_wrist_pose.visibility > 0.5

        if not face_active and l_hand_visible_pose:
            dist_nose_l = dist(l_index_pose, nose)
            dist_ear_l = dist(l_index_pose, l_ear)
            
            if dist_nose_l < face_threshold:
                face_active, face_type = True, "MANO EN ROSTRO"
            elif dist_ear_l < ear_threshold:
                face_active, face_type = True, "TOCAR CABELLO/OREJA"

        if not face_active and r_hand_visible_pose:
            dist_nose_r = dist(r_index_pose, nose)
            dist_ear_r = dist(r_index_pose, r_ear)
            
            if dist_nose_r < face_threshold:
                face_active, face_type = True, "MANO EN ROSTRO"
            elif dist_ear_r < ear_threshold:
                face_active, face_type = True, "TOCAR CABELLO/OREJA"

        # --- MOTOR DE FRECUENCIA PARA EL REPORTE FINAL ---
        current_time = time.time()
        report_active = False
        final_report_type = face_type

        if face_active and not self.is_touching_now:
            self.touch_history.append(current_time)
            self.current_touch_start = current_time
        
        self.is_touching_now = face_active
        if not face_active:
            self.current_touch_start = 0

        while self.touch_history and self.touch_history[0] < current_time - self.time_window:
            self.touch_history.popleft()

        if face_active:
            if len(self.touch_history) >= self.rep_limit:
                report_active = True
                final_report_type = f"ANSIEDAD: TOQUES REPETITIVOS ({face_type})"
            elif self.current_touch_start > 0 and (current_time - self.current_touch_start) > self.prolonged_limit:
                report_active = True
                final_report_type = f"ANSIEDAD: MOVIMIENTO PROLONGADO ({face_type})"

        res["face"] = {
            "instant_active": face_active,                 
            "report_active": report_active,                
            "type": final_report_type,                     
            "instant_type": face_type if face_type else "" 
        }

        return res