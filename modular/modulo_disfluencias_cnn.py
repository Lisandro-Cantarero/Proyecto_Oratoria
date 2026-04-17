import os
import numpy as np
import librosa
from typing import List, Dict, Any
import warnings

# Suprimir advertencias
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from tensorflow import keras
from tensorflow.keras.models import load_model

N_MELS = 128
MAX_PAD_LEN = 32

def _procesar_segmento_muletilla(audio_segment: np.ndarray, sr: int) -> np.ndarray:
    """Convierte el segmento a un Mel Spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    if log_mel.shape[1] > MAX_PAD_LEN:
        log_mel = log_mel[:, :MAX_PAD_LEN]
    else:
        pad_width = MAX_PAD_LEN - log_mel.shape[1]
        log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return log_mel[np.newaxis, ..., np.newaxis]

def detectar_muletillas_acusticas(
    y_limpio: np.ndarray, 
    sr: int,
    marcas_whisper: List[Dict[str, Any]] = None,
    umbral_confianza: float = 0.75
) -> Dict[str, Any]:
    """
    Inferencia optimizada por Lotes (Batch Processing).
    Filtro de persistencia calibrado según impacto psicolingüístico (>400ms).
    """
    ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo_muletillas_v2.keras')
    
    if not os.path.exists(ruta_modelo):
        return {"error": f"Modelo CNN no encontrado en {ruta_modelo}"}

    print(f"🧠 Ejecutando inferencia acústica pura (Umbral CNN: {umbral_confianza*100}%)...")
    
    try:
        modelo_cnn = load_model(ruta_modelo)
        window_size = sr
        # Paso fino: 100ms
        step_size = sr // 10 
        
        lotes_x = []
        tiempos_lote = []

        # 1. Recolección de Tensors
        for i in range(0, len(y_limpio) - window_size, step_size):
            segmento = y_limpio[i:i + window_size]
            
            # Filtro de silencio más permisivo para voces bajas
            if np.max(np.abs(segmento)) < 0.01: 
                continue
                
            segmento_norm = segmento / np.max(np.abs(segmento))
            x_input = _procesar_segmento_muletilla(segmento_norm, sr)
            
            lotes_x.append(x_input[0])
            tiempos_lote.append(i / sr)

        duracion_minutos = (len(y_limpio) / sr) / 60.0 if sr > 0 else 0.0

        if not lotes_x:
            return {
                "total_muletillas_acusticas": 0, 
                "falsos_positivos_filtrados": 0,
                "frecuencia_por_minuto": 0.0, 
                "eventos_detallados": []
            }

        # 2. INFERENCIA MASIVA 
        tensor_batch = np.array(lotes_x)
        predicciones = modelo_cnn.predict(tensor_batch, verbose=0)
        
        # 3. FILTRADO EXCLUSIVAMENTE ACÚSTICO (Persistencia y Cooldown)
        eventos_filtrados = []
        frames_consecutivos = 0
        tiempo_inicio_racha = None
        confianzas_racha = []
        
        # Umbral heurístico de persistencia basado en (Vera-Ramírez et al.)
        # Muletillas prolongadas y vocálicas > 400ms reducen la competencia percibida.
        # Como cada frame es de 100ms, requerimos al menos 4 frames consecutivos.
        MIN_FRAMES_PERSISTENCIA = 4 

        for idx, prob in enumerate(predicciones):
            prediccion = float(prob[0])
            inicio_seg = tiempos_lote[idx]
            
            # Lógica de Racha (Persistencia)
            if prediccion >= umbral_confianza:
                if frames_consecutivos == 0:
                    tiempo_inicio_racha = inicio_seg
                frames_consecutivos += 1
                confianzas_racha.append(prediccion)
            else:
                # Si se rompe la racha, verificamos si fue lo suficientemente larga (>= 400ms)
                if frames_consecutivos >= MIN_FRAMES_PERSISTENCIA:
                    confianza_promedio = sum(confianzas_racha) / len(confianzas_racha)
                    
                    # Verificamos el cooldown para no registrar la misma muletilla larga varias veces
                    if not eventos_filtrados or (tiempo_inicio_racha - eventos_filtrados[-1]['inicio']) >= 1.5:
                        eventos_filtrados.append({
                            "inicio": round(tiempo_inicio_racha, 2),
                            "confianza": round(confianza_promedio, 3),
                            "duracion_ms": frames_consecutivos * 100 # Exportamos el impacto real en ms
                        })
                
                # Reseteamos los contadores
                frames_consecutivos = 0
                tiempo_inicio_racha = None
                confianzas_racha = []

        # Capturar racha si el audio termina justo en medio de una muletilla prolongada
        if frames_consecutivos >= MIN_FRAMES_PERSISTENCIA:
            confianza_promedio = sum(confianzas_racha) / len(confianzas_racha)
            if not eventos_filtrados or (tiempo_inicio_racha - eventos_filtrados[-1]['inicio']) >= 1.5:
                 eventos_filtrados.append({
                     "inicio": round(tiempo_inicio_racha, 2),
                     "confianza": round(confianza_promedio, 3),
                     "duracion_ms": frames_consecutivos * 100
                 })

        muletillas_detectadas_cnn = len(eventos_filtrados)

        return {
            "total_muletillas_acusticas": muletillas_detectadas_cnn,
            "falsos_positivos_filtrados": 0, # Se mantiene para compatibilidad con el CSV
            "frecuencia_por_minuto": round(muletillas_detectadas_cnn / duracion_minutos, 2) if duracion_minutos > 0 else 0,
            "eventos_detallados": eventos_filtrados
        }

    except Exception as e:
        return {"error": f"Fallo en la inferencia CNN: {str(e)}"}

if __name__ == "__main__":
    print("Módulo de Inferencia Acústica calibrado y listo para el orquestador.")