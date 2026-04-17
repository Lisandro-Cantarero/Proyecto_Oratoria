import math
import collections
import numpy as np

class AnalizadorSelfAdaptors:
    def __init__(self):
        # =================================================================
        # CONFIGURACIÓN ROSTRO Y TOQUES (sFST)
        # =================================================================
        self.zonas_faciales = {
            "eje_central": [1, 2, 4, 164, 0, 17], 
            "lateral_izq": [234, 93, 132],        
            "lateral_der": [454, 323, 361]        
        }
        self.puntos_mano = [0, 4, 8, 12, 16, 20] 
        self.estado_facial = "REPOSO"
        self.inicio_t2_facial = 0
        self.ultimo_frame_con_toque = 0
        self.tolerancia_perdida_segundos = 0.5 
        self.total_toques_faciales = 0
        self.tiempo_total_facial = 0.0

        # =================================================================
        # CONFIGURACIÓN FROTAMIENTO (FIDGETING)
        # =================================================================
        self.estado_manos = "SEPARADAS"
        self.inicio_frotamiento = 0
        self.historial_distancias_manos = collections.deque(maxlen=15) 
        
        
        self.umbral_varianza_movimiento = 0.0005 
        
        self.total_frotamientos = 0
        self.tiempo_total_frotamiento = 0.0
        
        
        self.ultimo_timestamp = 0.0

    def _calcular_distancia_pixel(self, p1, p2, ancho, alto):
        return math.hypot((p1.x - p2.x) * ancho, (p1.y - p2.y) * alto)

    def _calcular_umbral_dinamico(self, face_landmarks, ancho, alto):
        if not face_landmarks: return ancho * 0.15
        sien_izq = face_landmarks.landmark[234]
        sien_der = face_landmarks.landmark[454]
        return self._calcular_distancia_pixel(sien_izq, sien_der, ancho, alto) * 0.20

    def _calcular_centro_palma(self, hand_landmarks):
        # MEJORA: Ahora extrae el Eje Z (Profundidad)
        p0 = hand_landmarks.landmark[0]
        p9 = hand_landmarks.landmark[9]
        class Punto: pass
        pt = Punto()
        pt.x, pt.y, pt.z = (p0.x + p9.x)/2, (p0.y + p9.y)/2, (p0.z + p9.z)/2
        return pt

    def procesar_frame_adaptors(self, face_landmarks, right_hand, left_hand, timestamp_actual, frame_shape):
       
        self.ultimo_timestamp = timestamp_actual 
        
        alerta_facial = None
        alerta_manos = None
        alto, ancho = frame_shape
        
        # =================================================================
        # 1. LÓGICA DE TOQUES FACIALES (sFST)
        # =================================================================
        toque_detectado = None
        puntos_evaluar = []
        
        if right_hand:
            puntos_evaluar.extend([right_hand.landmark[i] for i in self.puntos_mano])
        if left_hand:
            puntos_evaluar.extend([left_hand.landmark[i] for i in self.puntos_mano])
        
        if face_landmarks and puntos_evaluar:
            umbral_dinamico_facial = self._calcular_umbral_dinamico(face_landmarks, ancho, alto)
            
            for punto_mano in puntos_evaluar:
                for zona, indices_rostro in self.zonas_faciales.items():
                    for idx in indices_rostro:
                        punto_rostro = face_landmarks.landmark[idx]
                        distancia = self._calcular_distancia_pixel(punto_mano, punto_rostro, ancho, alto)
                        
                        if distancia < umbral_dinamico_facial:
                            toque_detectado = zona
                            break 
                    if toque_detectado: break
                if toque_detectado: break

        
        if toque_detectado:
            self.ultimo_frame_con_toque = timestamp_actual
            
            if self.estado_facial == "REPOSO":
                self.estado_facial = "T2_CONTACTO"
                self.inicio_t2_facial = timestamp_actual
                self.zona_contacto = toque_detectado
                alerta_facial = f"POSTURA CERRADA: TOQUE {self.zona_contacto.upper()}"
                
            elif self.estado_facial == "T2_CONTACTO":
                duracion = timestamp_actual - self.inicio_t2_facial
                alerta_facial = f"POSTURA CERRADA ({duracion:.1f}s)"
                
        else:
            if self.estado_facial == "T2_CONTACTO":
                tiempo_sin_toque = timestamp_actual - self.ultimo_frame_con_toque
                
                duracion = timestamp_actual - self.inicio_t2_facial
                alerta_facial = f"POSTURA CERRADA ({duracion:.1f}s)"
                
                if tiempo_sin_toque > self.tolerancia_perdida_segundos:
                    duracion_t2 = self.ultimo_frame_con_toque - self.inicio_t2_facial
                    
                    if duracion_t2 < 10.0:
                        self.total_toques_faciales += 1
                        self.tiempo_total_facial += duracion_t2
                    
                    self.estado_facial = "REPOSO"
                    self.zona_contacto = None
                    alerta_facial = None 

        # =================================================================
        # 2. LÓGICA DE FROTAMIENTO MEJORADA (Eje Z + Varianza)
        # =================================================================
        manos_juntas = False
        es_frotamiento_activo = False

        if right_hand and left_hand:
            centro_der = self._calcular_centro_palma(right_hand)
            centro_izq = self._calcular_centro_palma(left_hand)
            
            dist_manos_2d = self._calcular_distancia_pixel(centro_der, centro_izq, ancho, alto)
            dist_manos_z = abs(centro_der.z - centro_izq.z) # Validación de Profundidad 3D
            
            umbral_manos = self._calcular_umbral_dinamico(face_landmarks, ancho, alto) * 2.0
            
            # Exigimos proximidad visual en 2D Y proximidad real en Z (Menos de 0.15 de desfase)
            if dist_manos_2d < umbral_manos and dist_manos_z < 0.15:
                manos_juntas = True
                self.historial_distancias_manos.append(dist_manos_2d / ancho)
                
                if len(self.historial_distancias_manos) >= 10:
                    varianza = np.var(self.historial_distancias_manos)
                    if varianza > self.umbral_varianza_movimiento:
                        es_frotamiento_activo = True

        
        if manos_juntas and es_frotamiento_activo:
            if self.estado_manos == "SEPARADAS":
                self.estado_manos = "FROTANDO"
                self.inicio_frotamiento = timestamp_actual
            alerta_manos = f"FIDGETING: FROTANDO MANOS ({timestamp_actual - self.inicio_frotamiento:.1f}s)"
        
        elif not manos_juntas and self.estado_manos == "FROTANDO":
            if (timestamp_actual - self.inicio_frotamiento) > 0.8:
                duracion_final = timestamp_actual - self.inicio_frotamiento
                if duracion_final > 1.5: 
                    self.total_frotamientos += 1
                    self.tiempo_total_frotamiento += duracion_final
                self.estado_manos = "SEPARADAS"

        return alerta_facial, alerta_manos

    def generar_reporte_post_hoc(self):
        """
        Genera el diccionario con el resumen numérico.
        EJECUTA EL CIERRE FORZADO de cualquier evento que estuviera 
        ocurriendo exactamente cuando se acabó el video.
        """
        # 1. CIERRE FORZADO: Toques Faciales
        if self.estado_facial == "T2_CONTACTO":
            duracion_t2 = self.ultimo_frame_con_toque - self.inicio_t2_facial
            if duracion_t2 < 10.0:
                self.total_toques_faciales += 1
                self.tiempo_total_facial += duracion_t2
            self.estado_facial = "REPOSO"

        # 2. CIERRE FORZADO: Frotamiento de Manos
        if self.estado_manos == "FROTANDO":
            duracion_final = self.ultimo_timestamp - self.inicio_frotamiento
            if duracion_final > 1.5:
                self.total_frotamientos += 1
                self.tiempo_total_frotamiento += duracion_final
            self.estado_manos = "SEPARADAS"

        return {
            "Total_Toques_Faciales": self.total_toques_faciales,
            "Tiempo_Total_Facial_Segundos": round(self.tiempo_total_facial, 2),
            "Total_Frotamientos_Manos": self.total_frotamientos,
            "Tiempo_Total_Frotamiento_Segundos": round(self.tiempo_total_frotamiento, 2)
        }