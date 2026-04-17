import numpy as np
import librosa
import parselmouth
import re
from typing import List, Dict, Any, Tuple
import warnings
from collections import deque

# Suprimir advertencias de divisiones por cero
warnings.filterwarnings("ignore")

def formato_tiempo(segundos: float) -> str:
    """Convierte segundos brutos a formato MM:SS"""
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"

def extraer_telemetria_local(
    y: np.ndarray, 
    sr: int, 
    segmentos_whisper: List[Dict[str, Any]], 
    ref_volumen_global: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracción acústica avanzada con Normalización Intra-Sujeto (Z-Score) 
    y Filtros de Persistencia cruzados por segmento.
    """
    telemetria_temporal = []
    eventos_estructurales_pdf = [] 

    if not segmentos_whisper:
        return [], []

    # =================================================================
    # FASE 1: CÁLCULO DE LA HUELLA BASE (BASELINE INTRA-SUJETO)
    # =================================================================
    duracion_base = min(30.0, len(y) / sr)
    muestras_base = int(duracion_base * sr)
    y_base = y[:muestras_base]
    
    # 1. Baseline de Tono (F0)
    try:
        snd_base = parselmouth.Sound(y_base, sr)
        pitch_base = snd_base.to_pitch(pitch_floor=60.0, pitch_ceiling=500.0)
        f0_base_valores = pitch_base.selected_array['frequency']
        f0_base_validos = f0_base_valores[f0_base_valores > 0]
        f0_baseline = float(np.median(f0_base_validos)) if len(f0_base_validos) > 0 else 130.0 # Ajustado a 130Hz
    except Exception:
        f0_baseline = 130.0

    # 2. Baseline de Energía (Dinámico en lugar de fijo -45dB)
    try:
        rms_base = librosa.feature.rms(y=y_base)[0]
        rms_db_base = librosa.amplitude_to_db(rms_base, ref=ref_volumen_global)
        umbral_energia = float(np.percentile(rms_db_base, 15))
    except Exception:
        umbral_energia = -45.0

    # 3. Baseline de Ritmo (WPM)
    wpm_iniciales = []
    for seg in segmentos_whisper:
        if float(seg['end']) <= duracion_base:
            dur = float(seg['end']) - float(seg['start'])
            texto_limpio = re.sub(r'[^\w\s]', '', seg['text'].lower())
            palabras = [p for p in texto_limpio.split() if p.strip()]
            if dur > 0:
                wpm_iniciales.append((len(palabras) / dur) * 60)
        else:
            break
            
    # Cálculo Gaussiano
    mu_wpm = float(np.mean(wpm_iniciales)) if wpm_iniciales else 140.0
    sigma_wpm = float(np.std(wpm_iniciales)) if len(wpm_iniciales) > 1 else 20.0
    if sigma_wpm < 5.0: sigma_wpm = 5.0 # Previene Z-scores explosivos por división por casi cero

    # =================================================================
    # FASE 2: ANÁLISIS POR VENTANAS DESLIZANTES Y PERSISTENCIA
    # =================================================================
    historial_wpm = deque(maxlen=3)
    prev_end = 0.0
    
    # Acumuladores de persistencia multi-segmento
    duracion_monotonia_acumulada = 0.0
    inicio_monotonia = None
    contador_micro_pausas = 0

    for seg in segmentos_whisper:
        inicio_seg = float(seg['start'])
        fin_seg = float(seg['end'])
        duracion = fin_seg - inicio_seg
        texto = seg['text']

        # --- A. DETECCIÓN DE SILENCIOS ---
        gap_silencio = inicio_seg - prev_end
        
        if prev_end > 0 and gap_silencio >= 2.5:
            eventos_estructurales_pdf.append({
                "tiempo": f"[{formato_tiempo(prev_end)} - {formato_tiempo(inicio_seg)}]",
                "tipo": "Silencio Incómodo",
                "detalle": f"Pausa excesiva y antinatural ({round(gap_silencio, 1)}s)",
                "contexto_texto": "[Silencio prolongado]",
                "start_sec": round(prev_end, 2),
                "duracion": round(gap_silencio, 2)
            })
            
        # Acumulador de atropellamiento
        if prev_end > 0 and gap_silencio < 0.35:
            contador_micro_pausas += 1
        elif gap_silencio >= 0.5:
            
            contador_micro_pausas = 0
        
        prev_end = fin_seg

        # --- B. DINÁMICA LINGÜÍSTICA (PPM / WPM) ---
        texto_limpio = re.sub(r'[^\w\s]', '', texto.lower())
        palabras = [p for p in texto_limpio.split() if p.strip()]
        num_palabras = len(palabras)
        wpm_local = (num_palabras / duracion) * 60 if duracion > 0 else 0.0
        
        # EL VERDADERO Z-SCORE
        z_wpm = (wpm_local - mu_wpm) / sigma_wpm

        # --- C. EXTRACCIÓN ACÚSTICA ---
        inicio_muestra = max(0, min(int(inicio_seg * sr), len(y)))
        fin_muestra = max(0, min(int(fin_seg * sr), len(y)))
        y_local = y[inicio_muestra:fin_muestra]

        db_promedio_local = -100.0 
        rango_semitonos = 0.0
        shimmer_local = 0.0        
        hnr_local_db = 0.0         

        if len(y_local) >= int(sr * 0.5):
            rms_local = librosa.feature.rms(y=y_local)[0]
            rms_db_local = librosa.amplitude_to_db(rms_local, ref=ref_volumen_global)
            db_promedio_local = float(np.mean(rms_db_local))

            if db_promedio_local > umbral_energia:
                snd = parselmouth.Sound(y_local, sr)
                pitch = snd.to_pitch(pitch_floor=60.0, pitch_ceiling=500.0)
                f0_valores = pitch.selected_array['frequency']
                f0_validos = f0_valores[f0_valores > 0]

                if len(f0_validos) > 10:
                    semitonos = 12 * np.log2(f0_validos / f0_baseline)
                    # Método robusto anti-outliers de Praat
                    rango_semitonos = float(np.percentile(semitonos, 95) - np.percentile(semitonos, 5))
                    
                # Biometría conservada para dataset (Envuelto en try/except para evitar crasheos si Praat falla)
                try:
                    
                    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 60.0, 500.0)
                    shim = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    shimmer_local = float(shim) if not np.isnan(shim) else 0.0
                    
                    harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
                    hnr_val = parselmouth.praat.call(harm, "Get mean", 0, 0)
                    hnr_local_db = float(hnr_val) if not np.isnan(hnr_val) else 0.0
                except Exception:
                    pass 

            # =================================================================
            # FASE 3: REGLAS COMPUESTAS ANTI-FALSOS POSITIVOS
            # =================================================================
            
            # 1. Habla Atropellada (Regla Compuesta Ajustada)
            if len(historial_wpm) > 0:
                delta_wpm = wpm_local - np.mean(historial_wpm)
                
                # Criterio: Velocidad inusual (Z>1.3) + [Aceleración moderada OR asfixia] en al menos 1.5s
                if (z_wpm > 1.3) and (delta_wpm >= 20.0 or contador_micro_pausas >= 1) and duracion >= 1.5:
                    eventos_estructurales_pdf.append({
                        "tiempo": f"[{formato_tiempo(inicio_seg)} - {formato_tiempo(fin_seg)}]",
                        "tipo": "Aceleración Local",
                        "detalle": f"Habla acelerada (Z-Score: +{round(z_wpm, 1)} | Δ+{round(delta_wpm)} PPM)",
                        "contexto_texto": texto.strip(),
                        "start_sec": round(inicio_seg, 2),
                        "duracion": round(duracion, 2)
                    })
                    contador_micro_pausas = 0 # Reset para evitar spam en cascada
            
            historial_wpm.append(wpm_local)

            # 2. Monotonía Severa (Persistencia Inter-Segmento)
            if rango_semitonos > 0 and rango_semitonos < 4.0 and db_promedio_local > umbral_energia:
                if inicio_monotonia is None:
                    inicio_monotonia = inicio_seg
                duracion_monotonia_acumulada += duracion
                
                if duracion_monotonia_acumulada >= 10.0:
                    eventos_estructurales_pdf.append({
                        "tiempo": f"[{formato_tiempo(inicio_monotonia)} - {formato_tiempo(fin_seg)}]",
                        "tipo": "Caída Tonal",
                        "detalle": f"Monotonía sostenida (Rango P95: {round(rango_semitonos, 1)} ST por {round(duracion_monotonia_acumulada, 1)}s)",
                        "contexto_texto": "Voz plana detectada a lo largo de múltiples enunciados.",
                        "start_sec": round(inicio_monotonia, 2),
                        "duracion": round(duracion_monotonia_acumulada, 2)
                    })
                    duracion_monotonia_acumulada = 0.0 # Reset tras el gatillo
                    inicio_monotonia = None
            else:
                # La varianza superó 4 ST, la monotonía se rompió
                duracion_monotonia_acumulada = 0.0
                inicio_monotonia = None

    
        telemetria_temporal.append({
            "inicio": round(inicio_seg, 2),
            "fin": round(fin_seg, 2),
            "duracion": round(duracion, 2),
            "texto": texto.strip(),
            "palabras": num_palabras,
            "wpm_local": round(wpm_local, 2),
            "z_score_wpm": round(z_wpm, 2),
            "rango_semitonos": round(rango_semitonos, 2),
            "volumen_db_local": round(db_promedio_local, 2),
            "shimmer_local": round(shimmer_local * 100, 2),
            "hnr_local_db": round(hnr_local_db, 2)
        })

    return telemetria_temporal, eventos_estructurales_pdf

if __name__ == "__main__":
    print("Módulo de telemetría local listo. Formato de minutos MM:SS activado.")