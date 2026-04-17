import copy
from typing import Dict, Any, List

# ==========================================
# CONSTANTES DE LA RÚBRICA (GLOBALES)
# ==========================================
RITMO_WPM_MIN = 110.0
RITMO_WPM_MAX = 160.0  
TONO_SD_MIN = 1.5  
TONO_SD_MAX = 6.0  
VAD_MIN_PORCENTAJE = 50.0
VAD_MAX_PORCENTAJE = 92.0 
DENSIDAD_LEXICA_CRITICA = 0.08  

def formato_tiempo(segundos: float) -> str:
    """Convierte segundos brutos a formato MM:SS"""
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"

def generar_veredicto(telemetria_pura: Dict[str, Any]) -> Dict[str, Any]:
    """
    Toma los datos matemáticos puros y aplica la rúbrica de evaluación oratoria
    con un enfoque pedagógico (Oportunidades de Mejora).
    """
    reporte = {
        "id_sesion": telemetria_pura.get("id_sesion", "Desconocido"),
        "crudos": copy.deepcopy(telemetria_pura),
        "evaluacion": {
            "alertas_globales": [],
            "eventos_locales": [],
            "feedback_detallado": {}
        }
    }
    
    resultados = telemetria_pura.get("resultados", {})
    alertas = reporte["evaluacion"]["alertas_globales"]
    feedback = reporte["evaluacion"]["feedback_detallado"]

    # ---------------------------------------------------------
    # 1. EVALUACIÓN DE RITMO Y FLUIDEZ GLOBAL
    # ---------------------------------------------------------
    ritmo = resultados.get("ritmo_y_fluidez", {})
    tasa_global = ritmo.get("tasa_global_wpm", 0)
    
    if tasa_global < RITMO_WPM_MIN:
        msg = f"Ritmo global pausado ({tasa_global:.1f} WPM). Se sugiere dinamizar la entrega para mantener la atención."
        alertas.append({"categoria": "Ritmo", "gravedad": "Media", "mensaje": msg})
        feedback["ritmo"] = msg
    elif tasa_global > RITMO_WPM_MAX:
        msg = f"Ritmo global elevado ({tasa_global:.1f} WPM). Oportunidad para integrar más pausas asimilativas."
        alertas.append({"categoria": "Ritmo", "gravedad": "Media", "mensaje": msg})
        feedback["ritmo"] = msg
    else:
        feedback["ritmo"] = f"Fluidez conversacional excelente ({tasa_global:.1f} WPM)."

    # ---------------------------------------------------------
    # 2. EVALUACIÓN PROSÓDICA (ENTONACIÓN Y VAD)
    # ---------------------------------------------------------
    prosodia = resultados.get("prosodia_global", {})
    tono_sd = prosodia.get("tono_std_st", 0)
    vad_pct = prosodia.get("vuv_porcentaje", 0)

    if tono_sd < TONO_SD_MIN:
        msg = f"Tonalidad plana (SD: {tono_sd:.2f} ST). Se recomienda modular la voz para enfatizar ideas clave."
        alertas.append({"categoria": "Prosodia", "gravedad": "Media", "mensaje": msg})
        feedback["tono"] = msg
    elif tono_sd > TONO_SD_MAX:
        msg = f"Alta fluctuación tonal (SD: {tono_sd:.2f} ST). Sugerencia: estabilizar los cierres de frase."
        alertas.append({"categoria": "Prosodia", "gravedad": "Media", "mensaje": msg})
        feedback["tono"] = msg
    else:
        feedback["tono"] = f"Expresividad vocal óptima y natural (SD: {tono_sd:.2f} ST)."

    if vad_pct < VAD_MIN_PORCENTAJE:
        feedback["densidad_habla"] = "Mínima presencia vocal detectada. Priorizar mayor tiempo de habla en futuras sesiones."
    elif vad_pct > VAD_MAX_PORCENTAJE:
        feedback["densidad_habla"] = "Discurso muy denso. Integrar breves pausas ayudará a la audiencia a reflexionar."
    else:
        feedback["densidad_habla"] = "Equilibrio perfecto entre habla y silencios funcionales."

    # ---------------------------------------------------------
    # 3. EVALUACIÓN CLÍNICA (BIOMETRÍA Y CALIDAD)
    # ---------------------------------------------------------
    shimmer = prosodia.get('shimmer', 0.0)
    hnr = prosodia.get('hnr_db', 0.0)
    
    calidad_audio_pobre = False
    # CORRECCIÓN: Atrapa el 0.0 estricto si Praat falló o el audio tiene ruido excesivo
    if hnr <= 0.0 or (0.0 < hnr < 15.0):
        feedback["calidad_vocal"] = "Señal acústica con ruido/aire. Parámetros micro-vocales bajo revisión."
        calidad_audio_pobre = True
    elif hnr >= 15.0:
        feedback["calidad_vocal"] = "Calidad de fonación y resonancia sólida."

    if shimmer > 0.038 and not calidad_audio_pobre:
        msg_estres = "Leve tensión en pliegues vocales detectada. Ejercicios de respiración diafragmática son recomendados."
        alertas.append({"categoria": "Proyección", "gravedad": "Media", "mensaje": msg_estres})
        feedback["estres_biometrico"] = msg_estres
    else:
        feedback["estres_biometrico"] = "Proyección de amplitud vocal estable y serena."

    # ---------------------------------------------------------
    # 4. EVALUACIÓN LÉXICA
    # ---------------------------------------------------------
    lexico = resultados.get("lexico_y_pragmatica", {})
    perfil_pragmatico = lexico.get("perfil_pragmatico", {})
    
    # Extraer solo las densidades de muletillas para no castigar conectores discursivos válidos
    densidad_total = 0
    if perfil_pragmatico and "densidades" in perfil_pragmatico:
        for cat, val in perfil_pragmatico["densidades"].items():
            if cat != "conector": # Ignoramos conectores lógicos
                densidad_total += val
    
    if densidad_total > DENSIDAD_LEXICA_CRITICA:
        msg = f"Uso frecuente de palabras de apoyo ({(densidad_total*100):.1f}%). Confía en el silencio en lugar de rellenarlo."
        alertas.append({"categoria": "Léxico", "gravedad": "Alta", "mensaje": msg})
        feedback["lexico"] = msg
    else:
        feedback["lexico"] = "Transiciones limpias y léxico preciso."

    # ---------------------------------------------------------
    # 5. INTEGRACIÓN DE EVENTOS PARA LA BITÁCORA FORENSE
    # ---------------------------------------------------------
    eventos_evaluados = reporte["evaluacion"]["eventos_locales"]

    # A. Eventos Estructurales (Z-Score y Monotonía)
    eventos_estructurales = resultados.get("eventos_estructurales", [])
    eventos_evaluados.extend(eventos_estructurales)

    # B. Muletillas Acústicas (CNN)
    disfluencias = resultados.get("disfluencias_acusticas", {}).get("eventos_detallados", [])
    for disfluencia in disfluencias:
        eventos_evaluados.append({
            "tiempo": f"[{formato_tiempo(disfluencia['inicio'])}]",
            "tipo": "Pausa Sonora",
            "detalle": f"Sonido de vacilación detectado ({(disfluencia['duracion_ms']/1000):.1f}s)",
            "contexto_texto": "Ej: 'eh', 'mmm'",
            "start_sec": disfluencia['inicio'] # Agregado para el sort del orquestador
        })

    # C. Muletillas Léxicas (spaCy)
    muletillas_lexicas = lexico.get("perfil_pragmatico", {}).get("detalles", [])
    for m_lex in muletillas_lexicas:
        if m_lex.get("inicio") is not None:
            eventos_evaluados.append({
                "tiempo": f"[{formato_tiempo(m_lex['inicio'])} - {formato_tiempo(m_lex['fin'])}]",
                "tipo": f"Apoyo ({m_lex['categoria'].capitalize()})",
                "detalle": "Palabra de relleno detectada",
                "contexto_texto": f"... {m_lex['expresion']} ...",
                "start_sec": m_lex['inicio'] # Agregado para el sort del orquestador
            })

    return reporte

if __name__ == "__main__":
    print("Motor de rúbrica formativa listo. Formato de minutos MM:SS activado.")