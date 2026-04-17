import librosa
import noisereduce as nr
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def _extraer_perfil_ruido_dinamico(y: np.ndarray, sr: int, duracion_ruido: float = 1.0) -> np.ndarray:
    """
    Escanea matemáticamente la señal para encontrar el segmento continuo
    de menor energía (RMS), garantizando un perfil de ruido real y no asumido.
    """
    frame_length = int(sr * 0.05)
    hop_length = int(frame_length / 2)
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    frames_necesarios = int((duracion_ruido * sr) / hop_length)
    
    if frames_necesarios >= len(rms):
        return y
    
    min_rms_sum = float('inf')
    best_start_frame = 0
    
    for i in range(len(rms) - frames_necesarios):
        current_sum = np.sum(rms[i:i + frames_necesarios])
        if current_sum < min_rms_sum:
            min_rms_sum = current_sum
            best_start_frame = i
            
    start_sample = best_start_frame * hop_length
    end_sample = start_sample + int(duracion_ruido * sr)
    
    return y[start_sample:end_sample]

def procesar_audio_oratoria(ruta_audio: str, sr_objetivo: int = 16000):
    """
    Aplica una arquitectura de doble ruta (Dual-Pipeline).
    
    Retorna:
        y_prosodia (np.ndarray): Audio crudo para extracción precisa de f0.
        y_asr (np.ndarray): Audio limpio para transcripción (Whisper) y CNN.
        sr (int): Frecuencia de muestreo (16kHz optimizado).
    """
    print(f"🎙️ Iniciando preprocesamiento dual y escaneo RMS: {ruta_audio}")
    
    try:
        # 1. Carga a 16kHz (Óptimo para Whisper, suficiente para frecuencias vocales)
        y_raw, sr = librosa.load(ruta_audio, sr=sr_objetivo)
        
        # ==========================================
        # RUTA 1: ACÚSTICA PURA (Prosodia)
        # ==========================================
        y_prosodia = y_raw 
        
        # ==========================================
        # RUTA 2: INTELIGIBILIDAD (Whisper y CNN)
        # ==========================================
        print("   -> Extrayendo huella de ruido ambiental (RMS)...")
        y_noise_profile = _extraer_perfil_ruido_dinamico(y_raw, sr, duracion_ruido=1.0)
        
        print("   -> Aplicando sustracción espectral conservadora (60%)...")
        y_asr_ruidoso = nr.reduce_noise(
            y=y_raw, 
            sr=sr, 
            y_noise=y_noise_profile, 
            prop_decrease=0.6,  
            stationary=True,   
            n_jobs=1            
        )
        
        y_asr = librosa.util.normalize(y_asr_ruidoso)
        
        print("✅ Pipeline dual generado con éxito.")
        return y_prosodia, y_asr, sr

    except Exception as e:
        print(f"❌ Error crítico al procesar el audio: {e}")
        return None, None, None

if __name__ == "__main__":
    print("Módulo de preprocesamiento bifásico listo.")