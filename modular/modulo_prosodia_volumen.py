import numpy as np
import librosa
import parselmouth
import torch
import warnings
from typing import Dict, Any

# Suprimir advertencias menores de divisiones por cero en silencios
warnings.filterwarnings("ignore")

# Caché global para el modelo VAD. Evita recargar PyTorch en cada llamada.
_vad_model = None
_get_speech_timestamps = None

def _cargar_modelo_silero():
    """Carga el modelo neuronal VAD en memoria solo la primera vez que se requiere."""
    global _vad_model, _get_speech_timestamps
    if _vad_model is None:
        print("   -> Cargando modelo Silero VAD (PyTorch)...")
        _vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
            verbose=False # Mantener consola limpia
        )
        _get_speech_timestamps = utils[0]

def _diccionario_vacio(ref_global: float) -> Dict[str, Any]:
    return {
        "tono_promedio_hz": 0.0, 
        "tono_std_st": 0.0, 
        "rango_pitch_st": 0.0,
        "vuv_porcentaje": 0.0, 
        "ref_volumen_max": ref_global,
        "jitter": 0.0, 
        "shimmer": 0.0, 
        "hnr_db": 0.0, # Añadido para validar calidad clínica
        "rms_media_db": 0.0,
        "rms_std_db": 0.0, 
        "tiempo_hablado_segundos": 0.0
    }

def analizar_prosodia_global(y_crudo: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Función principal requerida por el orquestador (Fase 5).
    Extrae características acústicas usando VAD neuronal y Praat (Two-Pass).
    CORRECCIÓN CLÍNICA: Praat recibe el audio continuo para no quebrar la fase de la onda.
    """
    _cargar_modelo_silero()

    # 1. REFERENCIA GLOBAL PARA RMS
    ref_global = float(np.max(np.abs(y_crudo))) if np.max(np.abs(y_crudo)) > 0 else 1.0

    # 2. VAD NEURONAL (Para aislar energía e identificar porcentaje de habla real)
    wav_tensor = torch.from_numpy(y_crudo).float()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timestamps = _get_speech_timestamps(wav_tensor, _vad_model, sampling_rate=sr)
    
    if not timestamps:
        return _diccionario_vacio(ref_global)

    # El audio recortado SOLO se usará para el RMS (Volumen)
    segmentos_voz = [y_crudo[ts['start']:ts['end']] for ts in timestamps]
    y_solo_voz = np.concatenate(segmentos_voz)
    
    # Cálculo del porcentaje real de habla (VUV)
    vuv_porcentaje = (len(y_solo_voz) / len(y_crudo)) * 100

    # 3. PITCH ADAPTATIVO (Two-Pass con Parselmouth)
    # CORRECCIÓN VITAL: Pasamos 'y_crudo' para mantener la continuidad de fase matemática
    snd = parselmouth.Sound(y_crudo, sr) 
    
    # Pasada 1: Estimación inicial amplia
    pitch_rough = snd.to_pitch(pitch_floor=60.0, pitch_ceiling=500.0)
    f0_rough = pitch_rough.selected_array['frequency']
    # Praat asigna 0 a los silencios, así que al filtrar > 0 nos quedamos solo con la voz natural
    f0_rough_validos = f0_rough[f0_rough > 0]

    if len(f0_rough_validos) > 0:
        # Calcular límites dinámicos basados en la propia voz del usuario
        floor_adaptativo = np.percentile(f0_rough_validos, 5) * 0.8
        ceiling_adaptativo = np.percentile(f0_rough_validos, 95) * 1.2
        pitch_floor = max(60.0, float(floor_adaptativo))
        pitch_ceiling = min(500.0, float(ceiling_adaptativo))
    else:
        pitch_floor, pitch_ceiling = 75.0, 400.0

    # Pasada 2: Extracción refinada con los límites del hablante
    pitch_refinado = snd.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    f0_valores = pitch_refinado.selected_array['frequency']
    f0_validos = f0_valores[f0_valores > 0]

    if len(f0_validos) > 0:
        f0_media = np.mean(f0_validos)
        f0_mediana = np.median(f0_validos)
        
        # Escala logarítmica (Semitonos relativos a la mediana)
        semitonos = 12 * np.log2(f0_validos / f0_mediana)
        std_semitonos = np.std(semitonos)
        
        # Rango en la misma escala relativa que std_semitonos
        rango_pitch_st = float(np.percentile(semitonos, 95)) - float(np.percentile(semitonos, 5))
    else:
        f0_media, std_semitonos, rango_pitch_st = 0.0, 0.0, 0.0

    # 4. MICRO-PERTURBACIONES CLÍNICAS (Jitter, Shimmer y HNR)
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
        
        # FIX APLICADO: El Jitter solo toma el point_process
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        # El Shimmer sí toma la onda y los pulsos
        shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Extracción del HNR (Escudo contra ruido ambiental)
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        
        jitter = jitter if not np.isnan(jitter) else 0.0
        shimmer = shimmer if not np.isnan(shimmer) else 0.0
        hnr = hnr if not np.isnan(hnr) else 0.0
    except Exception as e:
        # AVISO FORENSE: Si falla, ahora sabremos por qué
        print(f"⚠️ [AVISO] Praat no pudo extraer micro-perturbaciones (voz ilegible o ruido extremo): {e}")
        jitter, shimmer, hnr = 0.0, 0.0, 0.0

    # 5. ENERGÍA RMS (Referenciada al global)
    # Aquí sí usamos 'y_solo_voz' para que los silencios largos no hundan el promedio de volumen.
    rms = librosa.feature.rms(y=y_solo_voz)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=ref_global)
    
    rms_media_db = float(np.mean(rms_db)) if len(rms_db) > 0 else 0.0
    rms_std_db = float(np.std(rms_db)) if len(rms_db) > 0 else 0.0

    # 6. MAPEO FINAL ESTRICTO
    return {
        "tono_promedio_hz": float(f0_media),
        "tono_std_st": float(std_semitonos),
        "rango_pitch_st": float(rango_pitch_st),
        "vuv_porcentaje": float(vuv_porcentaje),
        "ref_volumen_max": ref_global,
        "rms_media_db": rms_media_db,
        "rms_std_db": rms_std_db, 
        "jitter": float(jitter),
        "shimmer": float(shimmer),
        "hnr_db": float(hnr), # Métrica clave para la rúbrica
        "tiempo_hablado_segundos": float(len(y_solo_voz) / sr)
    }

if __name__ == "__main__":
    print("Módulo de prosodia (Silero VAD + Two-Pass + Filtros Clínicos) listo para el orquestador.")