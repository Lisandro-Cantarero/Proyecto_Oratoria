import cv2
import numpy as np
import math

class AnalizadorMirada:
    """
    Analizador de Dirección de la Mirada basado en Estimación de Pose 3D (solvePnP).
    Adaptado estrictamente al estándar OpenOPAF: cámara central sin offset de contrapicado,
    y seguimiento geométrico usando ojos, nariz, orejas y boca.
    """
    def __init__(self):
        self.raw_yaw = []
        self.raw_pitch = []
        self.raw_eye_gaze = [] 
        self.frames_none = 0
        
        # -------------------------------------------------------------------
        # 1. PARÁMETROS DE TOLERANCIA Y EVALUACIÓN
        # -------------------------------------------------------------------
        self.umbral_yaw = 35.0   
        self.umbral_pitch = 30.0 
        # El offset compensatorio se ha eliminado para grabación a nivel de los ojos

        self.umbral_excelente = 85.0 
        self.umbral_bueno = 70.0     

        # Puntos 3D del modelo facial estándar OpenOPAF
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nariz (1): Origen
            (0.0, 200.0, -50.0),         # Boca (14): Eje Y Positivo (Abajo), Ligeramente adelante
            (-225.0, -170.0, 135.0),     # Ojo Izq Pantalla (263)
            (225.0, -170.0, 135.0),      # Ojo Der Pantalla (33)
            (-350.0, 50.0, 200.0),       # Tragus Izq Pantalla (234)
            (350.0, 50.0, 200.0)         # Tragus Der Pantalla (454)
        ], dtype=np.float64)

        # Umbrales para los ojos (Iris Ratio)
        self.umbral_ojo_centro_min = 0.40
        self.umbral_ojo_centro_max = 0.60

    def _obtener_matriz_camara(self, frame_shape):
        alto, ancho = frame_shape
        focal_length = ancho  
        center = (ancho / 2, alto / 2)
        return np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def procesar_frame_cabeza(self, face_landmarks, frame_shape):
        if not face_landmarks:
            self.frames_none += 1
            return

        lms = face_landmarks.landmark
        alto, ancho = frame_shape

        # Mapeo de landmarks 2D correspondientes al estándar (Nariz, Boca, Ojos, Orejas)
        puntos_2d = np.array([
            (lms[1].x * ancho, lms[1].y * alto),     
            (lms[14].x * ancho, lms[14].y * alto),   # Se utiliza Landmark 14 (Centro de la boca) 
            (lms[263].x * ancho, lms[263].y * alto), 
            (lms[33].x * ancho, lms[33].y * alto),   
            (lms[234].x * ancho, lms[234].y * alto), 
            (lms[454].x * ancho, lms[454].y * alto)  
        ], dtype=np.float64)

        matriz_cam = self._obtener_matriz_camara(frame_shape)
        dist_coeffs = np.zeros((4, 1)) 
        
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points, puntos_2d, matriz_cam, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            pitch = math.degrees(math.asin(-rmat[1, 2]))
            yaw = math.degrees(math.atan2(rmat[0, 2], rmat[2, 2]))
            
            self.raw_yaw.append(yaw)
            # Registro directo del pitch calculado por la matriz
            self.raw_pitch.append(pitch)

            try:
                iris_der_x = lms[468].x * ancho
                iris_izq_x = lms[473].x * ancho
                
                ojo_der_ext = lms[33].x * ancho
                ojo_der_int = lms[133].x * ancho
                ojo_der_ancho = abs(ojo_der_ext - ojo_der_int) + 1e-6
                ratio_der = abs(iris_der_x - ojo_der_int) / ojo_der_ancho
                
                ojo_izq_ext = lms[263].x * ancho
                ojo_izq_int = lms[362].x * ancho
                ojo_izq_ancho = abs(ojo_izq_ext - ojo_izq_int) + 1e-6
                ratio_izq = abs(iris_izq_x - ojo_izq_int) / ojo_izq_ancho
                
                self.raw_eye_gaze.append((ratio_der + ratio_izq) / 2.0)
            except:
                self.raw_eye_gaze.append(0.5) 

    def generar_reporte_post_hoc(self):
        if len(self.raw_yaw) == 0:
            return {"error": "No se detectaron rostros"}

        arr_yaw = np.array(self.raw_yaw)
        arr_pitch = np.array(self.raw_pitch)

        if len(arr_yaw) > 5:
            kernel = np.ones(5) / 5.0
            arr_yaw = np.convolve(arr_yaw, kernel, mode='same')
            arr_pitch = np.convolve(arr_pitch, kernel, mode='same')

        media_yaw = np.median(arr_yaw)
        media_pitch = np.median(arr_pitch)

        yaw_norm = arr_yaw - media_yaw
        pitch_norm = arr_pitch - media_pitch

        conteo_mirada = {"Front": 0, "Up": 0, "Down": 0, "Right": 0, "Left": 0, "Back": 0, "None": self.frames_none}
        conteo_ojos = {"Fijamente": 0, "Distraido": 0}
        ojos_validos = []

        for i, (y, p) in enumerate(zip(yaw_norm, pitch_norm)):
            if abs(y) > 75.0: zone = "Back"
            elif y > self.umbral_yaw: zone = "Left"
            elif y < -self.umbral_yaw: zone = "Right"
            elif p > self.umbral_pitch: zone = "Down"
            elif p < -self.umbral_pitch: zone = "Up"
            else:
                zone = "Front"
                gaze = self.raw_eye_gaze[i]
                ojos_validos.append(gaze)
                if self.umbral_ojo_centro_min < gaze < self.umbral_ojo_centro_max:
                    conteo_ojos["Fijamente"] += 1
                else:
                    conteo_ojos["Distraido"] += 1
            
            conteo_mirada[zone] += 1

        total_f_validos = len(yaw_norm)
        pct_front = (conteo_mirada["Front"] / total_f_validos) * 100 if total_f_validos > 0 else 0.0

        if pct_front >= self.umbral_excelente:
            nivel_dominio = "OPAF Excelente"
            habilidad_dominada = True
        elif pct_front >= self.umbral_bueno:
            nivel_dominio = "Bueno"
            habilidad_dominada = True
        else:
            nivel_dominio = "Deficiente"
            habilidad_dominada = False

        return {
            "OpenOPAF": {
                "porcentaje_mirada_audiencia": round(pct_front, 2),
                "nivel_dominio": nivel_dominio,
                "habilidad_dominada": habilidad_dominada,
                "desglose": conteo_mirada
            },
            "Rastreo_Ocular_Secundario": {
                "porcentaje_contacto_visual_real": round((conteo_ojos["Fijamente"]/len(ojos_validos)*100), 2) if ojos_validos else 0,
                "sd_movimiento_ocular": round(float(np.std(ojos_validos)), 4) if ojos_validos else 0,
                "desglose_ojos_en_front": conteo_ojos
            },
            "Chen_Metrics": {
                "Face_Left_Right_Sd": float(np.std(yaw_norm)),
                "Face_Up_Down_Sd": float(np.std(pitch_norm))
            }
        }
