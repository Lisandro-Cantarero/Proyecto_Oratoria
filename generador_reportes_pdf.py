import os
import glob
import json
import time
from xhtml2pdf import pisa

class GeneradorReporteElegante:
    """
    Lee los resultados JSON del Orquestador y genera un PDF diagnóstico multimodal completo.
    Utiliza un enfoque pedagógico y constructivo para motivar al orador, 
    reemplazando el lenguaje alarmista por oportunidades de mejora.
    """
    def __init__(self, ruta_json):
        self.ruta_json = ruta_json
        with open(ruta_json, 'r', encoding='utf-8') as f:
            self.datos = json.load(f)
            
        self.id_orador = self.datos.get('id_sesion', os.path.basename(ruta_json).replace('.json', ''))
        self.fecha = time.strftime('%d/%m/%Y %H:%M')

    # --- FUNCIONES HEURÍSTICAS DE DIAGNÓSTICO (ENFOQUE PEDAGÓGICO) ---
    def diag_fluidez(self, wpm):
        if wpm < 110: return "Ritmo pausado. Sugerencia: aumentar ligeramente la fluidez.", "#e67e22"
        elif 110 <= wpm <= 170: return "Ritmo ideal y conversacional.", "#2ecc71"
        else: return "Ritmo acelerado. Es recomendable hacer pausas entre ideas.", "#e67e22"

    def diag_contacto(self, contacto):
        if contacto < 60: return "Oportunidad para conectar más visualmente con la audiencia.", "#e67e22"
        elif 60 <= contacto <= 85: return "Buen nivel de conexión visual.", "#f1c40f"
        else: return "Excelente dominio y conexión visual.", "#2ecc71"

    def diag_tono(self, st):
        if st < 1.5: return "Tono estable. Sugerencia: añadir modulación para enfatizar ideas.", "#e67e22"
        elif 1.5 <= st <= 2.5: return "Modulación vocal adecuada.", "#f1c40f"
        else: return "Excelente variabilidad y expresividad vocal.", "#2ecc71"

    def diag_volumen(self, std_db):
        if std_db < 2.0: return "Volumen muy uniforme. Sugerencia: usar matices de intensidad para evitar monotonía.", "#e67e22"
        elif 2.0 <= std_db <= 6.0: return "Excelente control y dinámica de volumen.", "#2ecc71"
        else: return "Alta fluctuación de volumen. Se sugiere mantener una proyección más constante.", "#f1c40f"

    def diag_tics(self, tics):
        return ("Gestualidad tranquila y controlada.", "#2ecc71") if tics <= 3 else ("Movimientos adaptadores frecuentes. Oportunidad para relajar las manos.", "#e67e22")

    def diag_sway(self, sway):
        return ("Postura anclada y firme.", "#2ecc71") if sway < 5.0 else ("Balanceo continuo detectado. Se sugiere afirmar la postura base.", "#e67e22")

    def compilar_html(self):
        resultados = self.datos.get("crudos", {}).get("resultados", {})
        evaluacion = self.datos.get("evaluacion", {})
        transcripcion = self.datos.get("crudos", {}).get("metadata", {}).get("transcripcion", "Transcripción no disponible.")
        
        ritmo = resultados.get("ritmo_y_fluidez", {})
        lexico = resultados.get("lexico_y_pragmatica", {})
        disf = resultados.get("disfluencias_acusticas", {})
        prosodia = resultados.get("prosodia_global", {})
        vision = resultados.get("vision_global", {})

        wpm = round(ritmo.get("tasa_global_wpm", 0), 1)
        sps = round(ritmo.get("articulation_rate_sps", 0), 2)
        tono_st = round(prosodia.get("tono_std_st", 0), 2)
        
        # --- NUEVAS MÉTRICAS DE VOLUMEN ---
        rms_media_db = round(prosodia.get("rms_media_db", 0), 2)
        rms_std_db = round(prosodia.get("rms_std_db", 0), 2)
        
        perfil_pragmatico = lexico.get("perfil_pragmatico", {})
        densidad_lex = sum(perfil_pragmatico.get("densidades", {}).values()) if perfil_pragmatico else 0
        mul_lexicas = round(densidad_lex * 100, 2)
        mul_acusticas = disf.get("total_muletillas_acusticas", 0)
        
        # =========================================================
        # EXTRACCIÓN Y AUTO-CONTEO DE FRECUENCIAS LÉXICAS
        # =========================================================
        frecuencias = lexico.get("frecuencia_palabras", {})
        if not frecuencias and perfil_pragmatico:
            frecuencias = perfil_pragmatico.get("frecuencia_palabras", {})
            
        muletillas_detalles = perfil_pragmatico.get("detalles", [])
        if not frecuencias and muletillas_detalles:
            frecuencias = {}
            for m in muletillas_detalles:
                palabra = m.get("expresion", "").strip().lower()
                if palabra:
                    frecuencias[palabra] = frecuencias.get(palabra, 0) + 1

        # =========================================================
        # NUEVO DISEÑO VISUAL PARA LAS PALABRAS DE APOYO ("EN FILA")
        # =========================================================
        html_chips_muletillas = ""
        if frecuencias:
            html_chips_muletillas = "<div style='margin-top: 10px;'>"
            for palabra, conteo in sorted(frecuencias.items(), key=lambda x: x[1], reverse=True):
                if conteo > 0:
                    html_chips_muletillas += f"<div style='background-color: #f8f9f9; border: 1px solid #d5dbdb; border-left: 3px solid #3498db; border-radius: 2px; padding: 4px 8px; margin-bottom: 4px; font-size: 10px; color: #2c3e50;'><b>{palabra.upper()}</b>: {conteo} repeticiones</div>"
            html_chips_muletillas += "</div>"
        else:
            html_chips_muletillas = "<div style='margin-top: 6px; font-size: 10px; color: #7f8c8d;'>No se detectaron palabras de relleno específicas.</div>"

        # =========================================================
        # EXTRACCIÓN DE TELEMETRÍA CINÉTICA (Los 4 pTM y Escenario)
        # =========================================================
        mirada_pct = round(vision.get("mirada_y_cabeza", {}).get("OpenOPAF", {}).get("porcentaje_mirada_audiencia", 0), 1)
        
        p_postura = vision.get("postura_pTM", {})
        postura_cerrada = round(p_postura.get("pTM_postura", 0) * 100, 1)
        postura_inactiva = round(p_postura.get("pTM_inactividad", 0) * 100, 1)
        postura_sway = round(p_postura.get("pTM_sway", 0) * 100, 1)        
        postura_perfil = round(p_postura.get("pTM_perfil", 0) * 100, 1)    
        
        cobertura_escenario = round(vision.get("comportamiento", {}).get("total_stage_coverage", 0), 1)
        
        # =========================================================
        # EXTRACCIÓN AVANZADA DE NERVIOSISMO Y EMOCIÓN
        # =========================================================
        nervios = vision.get("nerviosismo", {})
        tics_totales = nervios.get("Total_Toques_Faciales", 0) + nervios.get("Total_Frotamientos_Manos", 0)
        
        tiempo_tics_total = round(nervios.get("Tiempo_Total_Facial_Segundos", 0.0) + nervios.get("Tiempo_Total_Frotamiento_Segundos", 0.0), 1)

        emociones = vision.get("emociones_faciales", {})
        p2n_ratio = round(emociones.get("P2N_Ratio_Metrics", {}).get("mean", 0), 2)
        joyness_mean = round(emociones.get("Joyness_Metrics", {}).get("mean", 0), 3)
        total_emotion_mean = round(emociones.get("Total_Emotion_Metrics", {}).get("mean", 0), 3)

        # Diagnósticos y colores base
        d_wpm, c_wpm = self.diag_fluidez(wpm)
        d_contacto, c_contacto = self.diag_contacto(mirada_pct)
        d_tono, c_tono = self.diag_tono(tono_st)
        d_vol, c_vol = self.diag_volumen(rms_std_db) # Evaluación de volumen
        d_tics, c_tics = self.diag_tics(tics_totales)
        
        c_mul_acu = "#2ecc71" if mul_acusticas < 5 else "#e67e22"
        c_postura = "#2ecc71" if postura_cerrada < 15.0 else "#e67e22"
        c_p2n = "#2ecc71" if p2n_ratio > 0.5 else "#f1c40f"
        
        c_inactiva = "#2ecc71" if postura_inactiva < 15.0 else "#e67e22"
        c_sway = "#2ecc71" if postura_sway < 5.0 else "#e67e22"
        c_perfil = "#2ecc71" if postura_perfil < 10.0 else "#e67e22"

        c_joy = "#2ecc71" if joyness_mean > 0.05 else "#f1c40f"
        d_joy = "Rostro con micro-expresiones de apertura y empatía." if joyness_mean > 0.05 else "Expresión excesivamente seria. Se sugiere sonreír sutilmente para conectar."
        
        c_tot_emo = "#2ecc71" if total_emotion_mean > 0.08 else "#e67e22"
        d_tot_emo = "Dinámica facial activa (Buena transmisión emocional)." if total_emotion_mean > 0.08 else "Rostro inexpresivo. Falta gesticulación facial para acompañar el mensaje."

        # =========================================================
        # FILTRO DE ALERTAS DUPLICADAS
        # =========================================================
        alertas_html = ""
        mensajes_registrados = set()
        
        for alerta in evaluacion.get("alertas_globales", []):
            color_alerta = "#d35400" if alerta.get("gravedad") == "Alta" else "#e67e22"
            mensaje_suave = alerta.get('mensaje', '').replace("Atropellamiento", "Ritmo elevado").replace("Monotonía", "Tonalidad plana").replace("excesiva", "prolongada")
            alertas_html += f"<li style='color: {color_alerta}; margin-bottom: 5px;'><b>{alerta.get('categoria')}:</b> {mensaje_suave}</li>"
            mensajes_registrados.add(mensaje_suave)
            
        feedback_detallado = evaluacion.get("feedback_detallado", {})
        feed_html = ""
        for k, v in feedback_detallado.items():
            if v not in mensajes_registrados:
                feed_html += f"<li><b>{k.capitalize()}:</b> {v}</li>"

        # =========================================================
        # SEPARACIÓN DE TABLAS FORENSES (TRES BITÁCORAS)
        # =========================================================
        eventos_locales = evaluacion.get("eventos_locales", [])
        
        filas_muletillas = ""
        filas_vocales = ""
        filas_kinesicas = ""
        
        for evento in eventos_locales:
            t = evento.get("tiempo", "")
            tipo = evento.get("tipo", "").replace("ALERTA", "Aviso").replace("Incómodo", "Prolongado")
            detalle = evento.get("detalle", "")
            detalle = detalle.replace("ALERTA:", "Sugerencia:").replace("excesiva", "prolongada").replace("excesivo", "elevado")
            detalle = detalle.replace("Atropellada", "Acelerada").replace("Monotonía severa", "Tonalidad plana sostenida")
            
            contexto = evento.get("contexto_texto", "")
            duracion = float(evento.get("duracion", 0.0))
            
            if "Aceleración" in tipo or "Lentitud" in tipo or "Ritmo" in tipo:
                if 0 < duracion < 3.0: continue 
            
            es_muletilla = any(keyword in tipo.lower() for keyword in ["muletilla", "relleno", "repetición", "disfluencia", "apoyo", "pausa sonora"])
            
            if es_muletilla:
                color_evento = "#d35400" 
                filas_muletillas += f"""
                <tr>
                    <td style="font-size: 10px; text-align: center;"><b>{t}</b></td>
                    <td style="font-size: 10px; color: {color_evento};"><b>{tipo}</b></td>
                    <td style="font-size: 10px;">{detalle}</td>
                    <td style="font-size: 9px; font-style: italic; color: #555555;">"{contexto}"</td>
                </tr>
                """
            elif contexto == "Capturado en tiempo real":
                color_evento = "#8e44ad" 
                filas_kinesicas += f"""
                <tr>
                    <td style="font-size: 10px; text-align: center;"><b>{t}</b></td>
                    <td style="font-size: 10px; color: {color_evento};"><b>{tipo}</b></td>
                    <td style="font-size: 10px;">{detalle}</td>
                    <td style="font-size: 9px; font-style: italic; color: #555555;">"{contexto}"</td>
                </tr>
                """
            else:
                color_evento = "#d35400" if ("Aceleración" in tipo or "Silencio" in tipo) else "#2980b9"
                filas_vocales += f"""
                <tr>
                    <td style="font-size: 10px; text-align: center;"><b>{t}</b></td>
                    <td style="font-size: 10px; color: {color_evento};"><b>{tipo}</b></td>
                    <td style="font-size: 10px;">{detalle}</td>
                    <td style="font-size: 9px; font-style: italic; color: #555555;">"{contexto}"</td>
                </tr>
                """
            
        if not filas_vocales:
            filas_vocales = "<tr><td colspan='4' style='text-align: center; padding: 15px;'>Manejo estructural adecuado. No se registraron anomalías significativas de ritmo o pausas.</td></tr>"
            
        if not filas_muletillas:
            filas_muletillas = "<tr><td colspan='4' style='text-align: center; padding: 15px; color: #2ecc71;'>Discurso fluido. No se detectaron vacilaciones sonoras.</td></tr>"
            
        if not filas_kinesicas:
            filas_kinesicas = "<tr><td colspan='4' style='text-align: center; padding: 15px; color: #2ecc71;'>Excelente expresión corporal. Movimientos alineados con el discurso.</td></tr>"

        html_content = f"""
        <html>
        <head>
            <style>
                @page {{ margin: 1.5cm; }}
                body {{ font-family: Helvetica, sans-serif; color: #333333; font-size: 11px; }}
                .header {{ background-color: #001220; color: #FFCC00; padding: 15px; text-align: center; border-radius: 5px; }}
                .header h1 {{ margin: 0; font-size: 20px; text-transform: uppercase; }}
                .header p {{ color: #ffffff; margin: 5px 0 0 0; font-size: 12px; }}
                
                h2 {{ color: #001220; border-bottom: 2px solid #FFCC00; padding-bottom: 3px; margin-top: 20px; font-size: 14px; }}
                
                .rango-info {{ font-size: 10px; background-color: #f4f6f7; border-left: 3px solid #3498db; padding: 6px 10px; margin-top: 5px; margin-bottom: 12px; color: #2c3e50; border-radius: 2px; }}
                
                table {{ width: 100%; border-collapse: collapse; margin-top: 5px; }}
                th, td {{ border: 1px solid #dddddd; padding: 8px; text-align: left; vertical-align: middle; }}
                th {{ background-color: #f2f2f2; color: #001220; font-weight: bold; }}
                
                .tabla-metricas th {{ width: 28%; }}
                .tabla-forense th {{ font-size: 10px; text-align: center; }}
                
                .metric-value {{ font-weight: bold; font-size: 14px; display: block; }}
                .feedback {{ font-style: italic; font-size: 10px; display: block; margin-top: 2px; color: #555555; }}
                
                .alert-box {{ background-color: #fdf2e9; border-left: 4px solid #f39c12; padding: 10px; margin-top: 15px; }}
                .alert-box h3 {{ margin-top: 0; color: #d35400; font-size: 12px; }}
                ul {{ padding-left: 15px; margin: 5px 0; }}
                
                .transcripcion-box {{ background-color: #f9f9f9; border-left: 4px solid #3498db; padding: 12px; font-style: italic; text-align: justify; line-height: 1.4; margin-top: 10px; }}
                
                .footer {{ margin-top: 30px; font-size: 9px; text-align: center; color: #7f8c8d; border-top: 1px solid #dddddd; padding-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reporte Formativo de Oratoria Multimodal</h1>
                <p>Auditoría Automatizada - UNAH Ingeniería</p>
                <p>Sesión: <b>{self.id_orador}</b> | Fecha: {self.fecha}</p>
            </div>

            <div class="alert-box">
                <h3>Resumen y Oportunidades de Mejora</h3>
                <ul>{feed_html}</ul>
                <ul style="margin-top: 10px;">{alertas_html}</ul>
            </div>

            <h2>Transcripción del Discurso (WhisperX)</h2>
            <div class="transcripcion-box">
                "{transcripcion}"
            </div>

            <h2>1. Dimensión Acústica y Fluidez (Voz)</h2>
            <div class="rango-info">
                <b>Rangos Óptimos:</b> Velocidad: 110 - 170 WPM | Expresividad Tonal > 1.5 Semitonos | Léxico de apoyo < 8.0%
            </div>
            <table class="tabla-metricas">
                <tr>
                    <th>Velocidad (WPM) y (SPS)</th>
                    <td>
                        <span class="metric-value" style="color: {c_wpm};">{wpm} Palabras/min | {sps} Sílabas/seg</span>
                        <span class="feedback">{d_wpm}</span>
                    </td>
                </tr>
                <tr>
                    <th>Variabilidad Tonal (Pitch)</th>
                    <td>
                        <span class="metric-value" style="color: {c_tono};">{tono_st} Semitonos (Desviación Estándar)</span>
                        <span class="feedback">{d_tono}</span>
                    </td>
                </tr>
                <tr>
                    <th>Intensidad Vocal (Volumen)</th>
                    <td>
                        <span class="metric-value" style="color: {c_vol};">{rms_media_db} dB (Media) | {rms_std_db} dB (Variabilidad)</span>
                        <span class="feedback">{d_vol}</span>
                    </td>
                </tr>
                <tr>
                    <th>Palabras de Apoyo (Léxicas)</th>
                    <td>
                        <span class="metric-value" style="color: #2980b9;">{mul_lexicas}% del discurso total</span>
                        <span class="feedback">Proporción del discurso ocupada por palabras de transición o apoyo.</span>
                        {html_chips_muletillas}
                    </td>
                </tr>
                <tr>
                    <th>Pausas Sonoras ("Ehhh") - CNN</th>
                    <td>
                        <span class="metric-value" style="color: {c_mul_acu};">{mul_acusticas} pausas detectadas</span>
                        <span class="feedback">Sonidos de vacilación o búsqueda de ideas.</span>
                    </td>
                </tr>
            </table>

            <h2>2. Dimensión Corporal y Uso Escénico (Kinésica)</h2>
            <div class="rango-info">
                <b>Metodología pTM:</b> Porcentaje de Tiempo en Estado de Penalización (Menor % es mejor).
            </div>
            <table class="tabla-metricas">
                <tr>
                    <th>Contacto Visual al Frente</th>
                    <td>
                        <span class="metric-value" style="color: {c_contacto};">{mirada_pct}% del tiempo total</span>
                        <span class="feedback">{d_contacto}</span>
                    </td>
                </tr>
                <tr>
                    <th>Postura Cerrada (Oclusión)</th>
                    <td>
                        <span class="metric-value" style="color: {c_postura};">{postura_cerrada}% (pTM)</span>
                        <span class="feedback">Tiempo con brazos cruzados, escondidos atrás o en los bolsillos.</span>
                    </td>
                </tr>
                <tr>
                    <th>Inactividad Gestual</th>
                    <td>
                        <span class="metric-value" style="color: {c_inactiva};">{postura_inactiva}% (pTM)</span>
                        <span class="feedback">Tiempo que el orador pasó inmóvil sin acompañar el discurso con los brazos.</span>
                    </td>
                </tr>
                <tr>
                    <th>Balanceo Excesivo (Sway)</th>
                    <td>
                        <span class="metric-value" style="color: {c_sway};">{postura_sway}% (pTM)</span>
                        <span class="feedback">Tiempo oscilando como péndulo (Indicador claro de fuga de energía o nervios).</span>
                    </td>
                </tr>
                <tr>
                    <th>Alineación (Cuerpo de Perfil)</th>
                    <td>
                        <span class="metric-value" style="color: {c_perfil};">{postura_perfil}% (pTM)</span>
                        <span class="feedback">Tiempo dando la espalda o en postura lateral prolongada hacia la audiencia.</span>
                    </td>
                </tr>
                <tr>
                    <th>Uso del Espacio</th>
                    <td>
                        <span class="metric-value" style="color: #2980b9;">{cobertura_escenario}% del encuadre</span>
                        <span class="feedback">Mide la apropiación y dinamismo en el espacio físico disponible.</span>
                    </td>
                </tr>
            </table>

            <h2>3. Indicadores de Serenidad y Emoción</h2>
            <div class="rango-info">
                <b>Rangos Óptimos:</b> Toques adaptadores <= 3 interacciones | Ratio P2N > 0.50 | Expresividad Activa
            </div>
            <table class="tabla-metricas">
                <tr>
                    <th>Gestos Adaptadores (Fuga)</th>
                    <td>
                        <span class="metric-value" style="color: {c_tics};">{tics_totales} interacciones ({tiempo_tics_total}s en total)</span>
                        <span class="feedback">{d_tics}</span>
                    </td>
                </tr>
                <tr>
                    <th>Balanza Emocional Facial (P2N)</th>
                    <td>
                        <span class="metric-value" style="color: {c_p2n};">{p2n_ratio} Ratio (Positivas / Negativas)</span>
                        <span class="feedback">Proporción de expresiones afables vs. tensión en el rostro.</span>
                    </td>
                </tr>
                <tr>
                    <th>Índice de Afabilidad (Joyness)</th>
                    <td>
                        <span class="metric-value" style="color: {c_joy};">{joyness_mean} Intensidad Media</span>
                        <span class="feedback">{d_joy}</span>
                    </td>
                </tr>
                <tr>
                    <th>Expresividad Global</th>
                    <td>
                        <span class="metric-value" style="color: {c_tot_emo};">{total_emotion_mean} Intensidad Media</span>
                        <span class="feedback">{d_tot_emo}</span>
                    </td>
                </tr>
            </table>

            <div style="page-break-before: always;"></div>

            <h2>4. Registro de Apoyos y Disfluencias</h2>
            <div class="rango-info">
                <b>Objetivo:</b> Ser consciente de los sonidos de vacilación para lograr mayor fluidez.
            </div>
            <table class="tabla-forense">
                <tr>
                    <th style="width: 15%; background-color: #fbeee6; color: #d35400;">Momento</th>
                    <th style="width: 25%; background-color: #fbeee6; color: #d35400;">Clasificación</th>
                    <th style="width: 25%; background-color: #fbeee6; color: #d35400;">Detalle</th>
                    <th style="width: 35%; background-color: #fbeee6; color: #d35400;">Contexto Transcrito</th>
                </tr>
                {filas_muletillas}
            </table>

            <h2>5. Registro Estructural de Voz (Pausas y Ritmo)</h2>
            <div class="rango-info">
                <b>Objetivo:</b> Utilizar las pausas mayores a 3s a favor del discurso y evitar aceleraciones abruptas.
            </div>
            <table class="tabla-forense">
                <tr>
                    <th style="width: 15%; background-color: #ebf5fb; color: #2980b9;">Momento</th>
                    <th style="width: 25%; background-color: #ebf5fb; color: #2980b9;">Dinámica Local</th>
                    <th style="width: 25%; background-color: #ebf5fb; color: #2980b9;">Detalle</th>
                    <th style="width: 35%; background-color: #ebf5fb; color: #2980b9;">Contexto Transcrito</th>
                </tr>
                {filas_vocales}
            </table>

            <h2>6. Registro de Lenguaje Corporal (Tiempo Real)</h2>
            <div class="rango-info">
                <b>Objetivo:</b> Mantener una postura abierta y conexión visual constante con la audiencia.
            </div>
            <table class="tabla-forense">
                <tr>
                    <th style="width: 15%; background-color: #f5eef8; color: #8e44ad;">Momento</th>
                    <th style="width: 25%; background-color: #f5eef8; color: #8e44ad;">Área Corporal</th>
                    <th style="width: 25%; background-color: #f5eef8; color: #8e44ad;">Observación</th>
                    <th style="width: 35%; background-color: #f5eef8; color: #8e44ad;">Origen del Dato</th>
                </tr>
                {filas_kinesicas}
            </table>

            <div class="footer">
                Investigación Aplicada - UNAH Ingeniería en Sistemas.<br>
                Procesamiento Formativo: WhisperX (ASR), MediaPipe (Pose), DeepFace (Emociones), Librosa/Silero (VAD/DSP).
            </div>
        </body>
        </html>
        """
        return html_content

    def exportar_pdf(self, ruta_salida):
        html_str = self.compilar_html()
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        
        with open(ruta_salida, "w+b") as archivo_pdf:
            pisa_status = pisa.CreatePDF(src=html_str, dest=archivo_pdf)
            
        if pisa_status.err:
            print(f"❌ Error al crear PDF: {pisa_status.err}")
        else:
            print(f"✅ Reporte PDF generado en: {ruta_salida}")

def procesar_carpeta_json():
    carpeta_entrada = "json"
    carpeta_salida = "reportes"

    if not os.path.exists(carpeta_entrada):
        os.makedirs(carpeta_entrada)
        print(f"⚠️ Se creó la carpeta '{carpeta_entrada}'. Pon tus JSON ahí y vuelve a ejecutar.")
        return

    archivos_json = glob.glob(os.path.join(carpeta_entrada, "*.json"))
    
    if not archivos_json:
        print(f"⚠️ No hay archivos JSON en '{carpeta_entrada}'.")
        return

    print(f"🚀 Iniciando generación para {len(archivos_json)} auditorías...")

    for ruta in archivos_json:
        try:
            n_base = os.path.basename(ruta).replace(".json", "")
            r_pdf = os.path.join(carpeta_salida, f"Auditoria_Forense_{n_base}.pdf")
            
            gen = GeneradorReporteElegante(ruta)
            gen.exportar_pdf(r_pdf)
            
        except Exception as e:
            print(f"❌ Falló {ruta}. Error: {e}")

    print("🏁 Proceso de PDFs completado.")

if __name__ == "__main__":
    procesar_carpeta_json()
