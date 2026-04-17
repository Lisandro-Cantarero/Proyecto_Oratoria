import numpy as np
import collections
import math

class RastreadorSway:
    """Clase auxiliar para rastrear oscilaciones con Suavizado (EMA) y Cooldown."""
    def __init__(self, umbral_ruido_pct, ventana_sec, cooldown_sec=0.4):
        self.umbral_ruido_pct = umbral_ruido_pct
        self.ventana_sec = ventana_sec
        self.cooldown_sec = cooldown_sec
        
        self.ultimo_extremo = None
        self.tendencia = 0
        self.tiempos_swings = collections.deque()
        
        self.valor_suavizado = None
        self.alpha = 0.15  

    def procesar(self, valor_crudo, ancho_hombros, timestamp):
        if self.valor_suavizado is None:
            self.valor_suavizado = valor_crudo
        else:
            self.valor_suavizado = (self.alpha * valor_crudo) + ((1 - self.alpha) * self.valor_suavizado)
            
        valor = self.valor_suavizado
        umbral_absoluto = ancho_hombros * self.umbral_ruido_pct
        
        if self.ultimo_extremo is None:
            self.ultimo_extremo = valor
            return

        delta = valor - self.ultimo_extremo
        
        if abs(delta) >= umbral_absoluto:
            nueva_tendencia = 1 if delta > 0 else -1
            
            if self.tendencia != 0 and nueva_tendencia != self.tendencia:
                if not self.tiempos_swings or (timestamp - self.tiempos_swings[-1]) >= self.cooldown_sec:
                    self.tiempos_swings.append(timestamp)
            
            self.tendencia = nueva_tendencia
            self.ultimo_extremo = valor

    def swings_activos(self, timestamp):
        while self.tiempos_swings and (timestamp - self.tiempos_swings[0]) > self.ventana_sec:
            self.tiempos_swings.popleft()
        return len(self.tiempos_swings)


class AnalizadorCinetico:
    """
    Analizador Cinético Multimodal - Ajuste de Ciclos Reales
    Incluye detección de Postura Cerrada, Gestualidad, Sway y Alineación de Hombros (Perfil).
    """
    def __init__(self):
        self.filtro_validacion_sec = 0.3      
        self.ventana_sway_sec = 4.0       
        self.max_inactividad_sec = 6.0    
        self.umbral_gesto_grados = 5.0    
        
        self.umbral_ruido_sway = 0.15     
        self.swings_permitidos = 4

        self.umbral_perfil_z = 0.25 
        self.tiempo_tolerancia_perfil_sec = 2.5 

        self.sway_x = RastreadorSway(self.umbral_ruido_sway, self.ventana_sway_sec)
        self.sway_z = RastreadorSway(self.umbral_ruido_sway, self.ventana_sway_sec)
        
        self.datos_brazo = {
            "izq": {"angulo_previo": None, "tendencia": 0, "ultimo_extremo": None},
            "der": {"angulo_previo": None, "tendencia": 0, "ultimo_extremo": None}
        }
        self.tiempo_ultimo_gesto = 0.0
        
        self.inicio_anomalia = {
            "postura_cerrada": None,
            "balanceo_excesivo": None,
            "inactividad_gestual": None,
            "cuerpo_de_perfil": None 
        }

        self.duracion_errores = {
            "postura_cerrada": 0.0,
            "balanceo_excesivo": 0.0,
            "inactividad_gestual": 0.0,
            "cuerpo_de_perfil": 0.0 
        }
        self.ultimo_timestamp = 0.0

    def _calcular_angulo_2d(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _validar_con_filtro(self, nombre_error, condicion_activa, timestamp_actual, umbral_tiempo_personalizado=None):
        tiempo_requerido = umbral_tiempo_personalizado if umbral_tiempo_personalizado is not None else self.filtro_validacion_sec
        estado_validado = False
        
        if condicion_activa:
            if self.inicio_anomalia[nombre_error] is None:
                self.inicio_anomalia[nombre_error] = timestamp_actual
            else:
                duracion_actual = timestamp_actual - self.inicio_anomalia[nombre_error]
                if duracion_actual >= tiempo_requerido:
                    estado_validado = True
                    delta_t = timestamp_actual - self.ultimo_timestamp
                    self.duracion_errores[nombre_error] += delta_t
        else:
            self.inicio_anomalia[nombre_error] = None
            
        return estado_validado

    def _es_gesto_valido(self, angulo_actual, lado):
        datos = self.datos_brazo[lado]
        if datos["angulo_previo"] is None:
            datos["angulo_previo"] = angulo_actual
            datos["ultimo_extremo"] = angulo_actual
            return False

        delta = angulo_actual - datos["angulo_previo"]
        tendencia_actual = 1 if delta > 0 else (-1 if delta < 0 else 0)

        gesto_detectado = False
        if tendencia_actual != 0 and tendencia_actual != datos["tendencia"]:
            amplitud = abs(angulo_actual - datos["ultimo_extremo"])
            if amplitud >= self.umbral_gesto_grados:
                gesto_detectado = True
            datos["ultimo_extremo"] = angulo_actual

        datos["tendencia"] = tendencia_actual if tendencia_actual != 0 else datos["tendencia"]
        datos["angulo_previo"] = angulo_actual
        return gesto_detectado

    def procesar_frame(self, pose_landmarks, mp_pose, timestamp_segundos):
        estado_salida = {
            "postura_cerrada": False, 
            "balanceo_excesivo": False, 
            "inactividad_gestual": False,
            "cuerpo_de_perfil": False 
        }
        
        if not pose_landmarks:
            self.ultimo_timestamp = timestamp_segundos
            return estado_salida 

        lms = pose_landmarks.landmark
        hombro_i, hombro_d = lms[mp_pose.PoseLandmark.LEFT_SHOULDER], lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        codo_i, codo_d = lms[mp_pose.PoseLandmark.LEFT_ELBOW], lms[mp_pose.PoseLandmark.RIGHT_ELBOW]
        muneca_i, muneca_d = lms[mp_pose.PoseLandmark.LEFT_WRIST], lms[mp_pose.PoseLandmark.RIGHT_WRIST]
        cadera_i, cadera_d = lms[mp_pose.PoseLandmark.LEFT_HIP], lms[mp_pose.PoseLandmark.RIGHT_HIP]

        if hombro_i.visibility < 0.5 or hombro_d.visibility < 0.5 or cadera_i.visibility < 0.5 or cadera_d.visibility < 0.5:
            self.ultimo_timestamp = timestamp_segundos
            return estado_salida

        ancho_hombros = abs(hombro_i.x - hombro_d.x) or 0.01 

        # ---------------------------------------------------------
        # MÉTRICA 0: ALINEACIÓN DE HOMBROS (Cuerpo de perfil)
        # ---------------------------------------------------------
        diferencia_z = abs(hombro_i.z - hombro_d.z)
        condicion_perfil = diferencia_z > self.umbral_perfil_z
        
        estado_salida["cuerpo_de_perfil"] = self._validar_con_filtro(
            "cuerpo_de_perfil", 
            condicion_perfil, 
            timestamp_segundos, 
            umbral_tiempo_personalizado=self.tiempo_tolerancia_perfil_sec
        )

        # ---------------------------------------------------------
        # MÉTRICA 1: POSTURA CERRADA (Ajustada para oclusión real)
        # ---------------------------------------------------------
        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        
        brazos_ocultos = (muneca_i.visibility < 0.5) and (muneca_d.visibility < 0.5)

       
        manos_atras_escondidas = (muneca_i.visibility < 0.6 and muneca_d.visibility < 0.6) and \
                                 (dist(muneca_i, muneca_d) < ancho_hombros * 0.4) and \
                                 (muneca_i.y > cadera_i.y - 0.2)

        brazos_ocultos = brazos_ocultos or manos_atras_escondidas

        dist_muneca_i_cadera = dist(muneca_i, cadera_i)
        dist_muneca_d_cadera = dist(muneca_d, cadera_d)
        
        manos_en_bolsillos = (dist_muneca_i_cadera < (ancho_hombros * 0.5)) and \
                             (dist_muneca_d_cadera < (ancho_hombros * 0.5)) and \
                             (muneca_i.y > cadera_i.y - 0.1) 
                             
        manos_caidas = (muneca_i.y > cadera_i.y) and (muneca_d.y > cadera_d.y)

        # Análisis principal de cruce (Si las muñecas son visibles)
        dist_cruce_1 = dist(muneca_i, codo_d)
        dist_cruce_2 = dist(muneca_d, codo_i)
        
        cruce_fisico = (dist_cruce_1 < (ancho_hombros * 0.45)) or (dist_cruce_2 < (ancho_hombros * 0.45))
        en_torso = (muneca_i.y < cadera_i.y) and (muneca_i.y > hombro_i.y)
        
        # MEJORA: Backup de oclusión mediante los codos
        # Si los codos están inusualmente pegados (menos del 75% del ancho de hombros) 
        # y están por encima de la cadera, el orador tiene los brazos cruzados o estrujados.
        dist_codos = abs(codo_i.x - codo_d.x)
        codos_pegados_al_frente = (dist_codos < (ancho_hombros * 0.75)) and \
                                  (codo_i.y < cadera_i.y) and (codo_d.y < cadera_d.y)

        # Se considera cruzado si se tocan los codos opuestos, O SI los codos están pegados por oclusión
        brazos_cruzados = (cruce_fisico and en_torso) or codos_pegados_al_frente

        condicion_cerrada = manos_en_bolsillos or manos_caidas or brazos_cruzados or brazos_ocultos
        
        estado_salida["postura_cerrada"] = self._validar_con_filtro("postura_cerrada", condicion_cerrada, timestamp_segundos)

        # ---------------------------------------------------------
        # MÉTRICA 2: ACTIVIDAD GESTUAL
        # ---------------------------------------------------------
        angulo_i = self._calcular_angulo_2d(hombro_i, codo_i, muneca_i)
        angulo_d = self._calcular_angulo_2d(hombro_d, codo_d, muneca_d)

        gesto_i = self._es_gesto_valido(angulo_i, "izq")
        gesto_d = self._es_gesto_valido(angulo_d, "der")

        if gesto_i or gesto_d:
            self.tiempo_ultimo_gesto = timestamp_segundos

        condicion_inactividad = (timestamp_segundos - self.tiempo_ultimo_gesto) >= self.max_inactividad_sec
        estado_salida["inactividad_gestual"] = self._validar_con_filtro("inactividad_gestual", condicion_inactividad, timestamp_segundos)

        # ---------------------------------------------------------
        # MÉTRICA 3: SWAY (Balanceo)
        # ---------------------------------------------------------
        centro_cadera_x = (cadera_i.x + cadera_d.x) / 2.0
        centro_cadera_z = (cadera_i.z + cadera_d.z) / 2.0 
        
        self.sway_x.procesar(centro_cadera_x, ancho_hombros, timestamp_segundos)
        self.sway_z.procesar(centro_cadera_z, ancho_hombros, timestamp_segundos)

        swings_x = self.sway_x.swings_activos(timestamp_segundos)
        swings_z = self.sway_z.swings_activos(timestamp_segundos)

        condicion_sway = (swings_x >= self.swings_permitidos) or (swings_z >= self.swings_permitidos)

        if condicion_sway:
            self.duracion_errores["balanceo_excesivo"] += (timestamp_segundos - self.ultimo_timestamp)

        estado_salida["balanceo_excesivo"] = condicion_sway

        self.ultimo_timestamp = timestamp_segundos
        return estado_salida

    def finalizar_sesion(self, tiempo_total_sesion):
        if tiempo_total_sesion <= 0: return {}
        return {
            "pTM_postura": self.duracion_errores["postura_cerrada"] / tiempo_total_sesion,
            "pTM_sway": self.duracion_errores["balanceo_excesivo"] / tiempo_total_sesion,
            "pTM_inactividad": self.duracion_errores["inactividad_gestual"] / tiempo_total_sesion,
            "pTM_perfil": self.duracion_errores["cuerpo_de_perfil"] / tiempo_total_sesion 
        }