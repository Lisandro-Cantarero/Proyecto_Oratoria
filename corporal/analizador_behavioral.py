import numpy as np
import collections
import math

class AnalizadorBehavioral:
    """
    Analizador de Comportamiento y Evaluación Post-Hoc (Wörtwein et al., 2015).
    Combina la extracción estadística avanzada (mean, max, std) con el rigor 
    matemático angular dictado por la literatura.
    Adaptado con Zonas Muertas (Deadbands) para ignorar el jitter de grabación a mano alzada.
    """
    def __init__(self):
        # ---------------------------------------------------------
        # PARÁMETROS CIENTÍFICOS Y MITIGACIÓN DE MEDIAPIPE
        # ---------------------------------------------------------

        self.deadband_grados = 5.0       
        self.truncamiento_grados = 45.0  
        

        
        self.deadband_desplazamiento = 0.15 
        self.max_desplazamiento_frame = 0.5 
        
        # ---------------------------------------------------------
        # ESTADOS ACUMULATIVOS
        # ---------------------------------------------------------
        self.angulos_previos = {
            "hombro_i": None, "codo_i": None,
            "hombro_d": None, "codo_d": None
        }
        
        self.historial_energia_util = [] 
        self.frames_totales = 0
        self.frames_gesticulando = 0 
        
        # Tracking del escenario
        self.centro_cadera_x_previo = None
        self.distancia_escenario_recorrida = 0.0

    def _calcular_angulo_2d(self, p1, p2, p3):
        """Calcula el ángulo interno de la articulación usando vectores."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def procesar_frame_behavioral(self, pose_landmarks, mp_pose):
        if not pose_landmarks:
            return

        self.frames_totales += 1
        lms = pose_landmarks.landmark
        
        hombro_i, hombro_d = lms[mp_pose.PoseLandmark.LEFT_SHOULDER], lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        codo_i, codo_d = lms[mp_pose.PoseLandmark.LEFT_ELBOW], lms[mp_pose.PoseLandmark.RIGHT_ELBOW]
        muneca_i, muneca_d = lms[mp_pose.PoseLandmark.LEFT_WRIST], lms[mp_pose.PoseLandmark.RIGHT_WRIST]
        cadera_i, cadera_d = lms[mp_pose.PoseLandmark.LEFT_HIP], lms[mp_pose.PoseLandmark.RIGHT_HIP]

        # Si no hay visibilidad confiable de los hombros, pausamos el tracking
        if hombro_i.visibility < 0.5 or hombro_d.visibility < 0.5:
            # Reseteamos el tracking de cadera previo para no crear un falso "salto" al recuperar la visión
            self.centro_cadera_x_previo = None 
            return

        ancho_hombros = abs(hombro_i.x - hombro_d.x) or 0.01

        # =========================================================
        # REGLAS DE "THE BOX" (Caja de Gesticulación Válida)
        # =========================================================
        # El paper establece la diferencia a cero si ambas manos no están sobre las caderas
        manos_arriba = (muneca_i.y < cadera_i.y) and (muneca_d.y < cadera_d.y)
        manos_visibles = (muneca_i.visibility > 0.4 and muneca_d.visibility > 0.4)
        
        # Detección geométrica de brazos cruzados (Muñeca a Codo Opuesto)
        dist_cruce_1 = math.hypot(muneca_i.x - codo_d.x, muneca_i.y - codo_d.y)
        dist_cruce_2 = math.hypot(muneca_d.x - codo_i.x, muneca_d.y - codo_i.y)
        cruce_fisico = (dist_cruce_1 < (ancho_hombros * 0.45)) or (dist_cruce_2 < (ancho_hombros * 0.45))
        en_torso = (muneca_i.y < cadera_i.y) and (muneca_i.y > hombro_i.y)
        brazos_cruzados = cruce_fisico and en_torso

        manos_validas = manos_arriba and manos_visibles and not brazos_cruzados

        # =========================================================
        # MÉTRICA 1: MOTION ENERGY (Sumatoria de Diferencias Angulares)
        # =========================================================
        angulos_actuales = {
            "hombro_i": self._calcular_angulo_2d(cadera_i, hombro_i, codo_i),
            "codo_i": self._calcular_angulo_2d(hombro_i, codo_i, muneca_i),
            "hombro_d": self._calcular_angulo_2d(cadera_d, hombro_d, codo_d),
            "codo_d": self._calcular_angulo_2d(hombro_d, codo_d, muneca_d)
        }

        energia_frame = 0.0

        if manos_validas:
            self.frames_gesticulando += 1
            
            for key in angulos_actuales.keys():
                if self.angulos_previos[key] is not None:
                    delta = abs(angulos_actuales[key] - self.angulos_previos[key])
                    
                    if delta > self.deadband_grados:
                        # CORRECCIÓN DE GLITCH: Limitamos el delta a un máximo realista (45 grados)
                        delta = min(delta, self.truncamiento_grados)
                        energia_frame += delta
            
           
            if energia_frame <= 120.0:
                self.historial_energia_util.append(energia_frame)

        self.angulos_previos = angulos_actuales

        # =========================================================
        # MÉTRICA 2: STAGE USAGE (Cobertura del Escenario) - CON CLAMPING Y DEADBAND
        # =========================================================
        centro_cadera_x = (cadera_i.x + cadera_d.x) / 2.0
        
        if self.centro_cadera_x_previo is not None:
            delta_x = abs(centro_cadera_x - self.centro_cadera_x_previo)
            desplazamiento_relativo = delta_x / ancho_hombros
            
            
            if self.deadband_desplazamiento < desplazamiento_relativo < self.max_desplazamiento_frame:
                self.distancia_escenario_recorrida += desplazamiento_relativo
                self.centro_cadera_x_previo = centro_cadera_x 
            elif desplazamiento_relativo >= self.max_desplazamiento_frame:
                # Si ocurrió un glitch de posición, reseteamos el previo para anclar la nueva posición válida
                self.centro_cadera_x_previo = centro_cadera_x
        else:
            # Primera inicialización válida
            self.centro_cadera_x_previo = centro_cadera_x

    def generar_reporte_post_hoc(self):
        """
        Calcula las métricas maestras finales para inyectar en tu regresor (Wörtwein et al., 2015).
        """
        if len(self.historial_energia_util) > 0:
            arr_energia = np.array(self.historial_energia_util)
            mean_energy = float(np.mean(arr_energia))
            max_energy = float(np.max(arr_energia))
            std_energy = float(np.std(arr_energia))
        else:
            mean_energy = max_energy = std_energy = 0.0

        pct_valido = (self.frames_gesticulando / max(1, self.frames_totales)) * 100

        return {
            "motion_energy_mean": mean_energy,
            "motion_energy_max": max_energy,
            "motion_energy_std": std_energy,
            "total_stage_coverage": self.distancia_escenario_recorrida,
            "porcentaje_tiempo_zona_valida": pct_valido
        }