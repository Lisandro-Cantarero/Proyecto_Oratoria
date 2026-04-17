import re
from typing import Dict, Any

def contar_silabas_espanol(palabra: str) -> int:
    """
    Aproximación rigurosa a núcleos silábicos sin usar Praat.
    Cuenta grupos vocálicos como un solo núcleo (diptongos/triptongos),
    emulando los 'picos de intensidad' del script de De Jong.
    """
    palabra = palabra.lower()
    # Encuentra secuencias de vocales (a, e, i, o, u, y acentuadas)
    nucleos = re.findall(r'[aeiouáéíóúü]+', palabra)
    return max(1, len(nucleos)) # Al menos 1 sílaba si tiene letras

def calcular_ritmo_whisper(resultado_asr: Dict[str, Any], duracion_total_audio: float = None) -> Dict[str, Any]:
    """
    Calcula métricas de ritmo (WPM) y tasas de articulación fonética (Sílabas/seg).
    Compatible con el motor de evaluación (WPM) y riguroso para la tesis (SPS).
    """
    palabras_validas = []
    total_silabas = 0
    
    # 1. Extracción Estricta Léxica
    for segment in resultado_asr.get('segments', []):
        for word_info in segment.get('words', []):
            palabra = word_info.get('word', '').strip()
            
            if palabra and re.search(r'[a-záéíóúüñA-Z]', palabra):
                palabras_validas.append({
                    'word': palabra,
                    'start': word_info['start'],
                    'end': word_info['end']
                })
                total_silabas += contar_silabas_espanol(palabra)
                
    if not palabras_validas:
        return {
            "total_palabras": 0,
            "total_silabas": 0,
            "tasa_global_wpm": 0.0,
            "tasa_articulacion_wpm": 0.0,
            "error": "No se encontraron palabras válidas en el resultado."
        }

    total_palabras = len(palabras_validas)

    # 2. Cálculo Estricto de Pausas Silentes (Ajustado a 400ms para rigor cognitivo)
    tiempo_pausas_silentes = 0.0
    eventos_pausas_detallados = [] # <-- AÑADIDO: Lista para guardar las pausas
    
    # UMBRAL AJUSTADO: 400ms evita restar pausas respiratorias cortas,
    # estabilizando la Tasa de Articulación a niveles humanos realistas.
    UMBRAL_PAUSA_SILENTE = 0.400 
    
    for i in range(1, total_palabras):
        fin_palabra_anterior = palabras_validas[i-1]['end']
        inicio_palabra_actual = palabras_validas[i]['start']
        
        brecha = inicio_palabra_actual - fin_palabra_anterior
        
        # Si Whisper da tiempos solapados (brecha < 0), simplemente no es pausa.
        if brecha >= UMBRAL_PAUSA_SILENTE:
            tiempo_pausas_silentes += brecha
            # <-- AÑADIDO: Guardar los detalles de la pausa
            eventos_pausas_detallados.append({
                "inicio": round(fin_palabra_anterior, 3),
                "fin": round(inicio_palabra_actual, 3),
                "duracion": round(brecha, 3)
            })

    # 3. Delimitación Correcta del Tiempo Total
    if duracion_total_audio is None:
        # Si no se envía el audio total, usamos el rango de las palabras detectadas
        tiempo_total_segundos = palabras_validas[-1]['end'] - palabras_validas[0]['start']
        tiempo_total_segundos = max(tiempo_total_segundos, 0.1) # Prevenir división por cero
    else:
        tiempo_total_segundos = duracion_total_audio 
    
    # 4. Tiempo de Fonación (Articulation Time)
    tiempo_fonacion_segundos = tiempo_total_segundos - tiempo_pausas_silentes
    
    if tiempo_total_segundos <= 0 or tiempo_fonacion_segundos <= 0:
        tiempo_total_segundos = 0.1
        tiempo_fonacion_segundos = 0.1

    # 5. Tasas Métricas Puras (Words & Syllables)
    speak_rate_wps = total_palabras / tiempo_total_segundos  # Palabras por segundo
    articulation_rate_sps = total_silabas / tiempo_fonacion_segundos # Sílabas por segundo (rigor fonético)
    
    # 6. Conversión Legacy para el Motor de Rúbrica (WPM)
    tasa_global_wpm = speak_rate_wps * 60.0
    tasa_articulacion_wpm = total_palabras / (tiempo_fonacion_segundos / 60.0)
    
    return {
        "total_palabras": total_palabras,
        "total_silabas": total_silabas,
        "tiempo_total_segundos": round(tiempo_total_segundos, 3),
        "tiempo_fonacion_segundos": round(tiempo_fonacion_segundos, 3),
        "tiempo_pausas_silentes_segundos": round(tiempo_pausas_silentes, 3),
        "speak_rate_wps": round(speak_rate_wps, 2),
        "articulation_rate_sps": round(articulation_rate_sps, 2),
        "tasa_global_wpm": round(tasa_global_wpm, 1),
        "tasa_articulacion_wpm": round(tasa_articulacion_wpm, 1),
        "eventos_pausas_detallados": eventos_pausas_detallados # <-- AÑADIDO: Exportar la lista al JSON
    }

if __name__ == "__main__":
    print("Módulo de ritmo fonético riguroso (calibrado) listo.")