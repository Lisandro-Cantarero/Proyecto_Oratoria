import os
import warnings

# --- 1. SILENCIO TÉCNICO ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import librosa
import numpy as np
import keras  # Importación directa segura
import config

# --- 2. CARGA DEL MODELO NATIVO ---
try:
    # ¡Mira qué limpio! Ya no hay custom_objects ni parches raros
    modelo_ia = keras.models.load_model('modelo_muletillas_final.h5')
    print("✅ IA de Muletillas: Modelo nativo cargado con éxito.")
except Exception as e:
    modelo_ia = None
    print(f"⚠️ IA de Muletillas: Error al cargar el modelo. Detalles: {e}")

# --- 3. PREPROCESAMIENTO IDÉNTICO AL ENTRENAMIENTO ---
def preprocesar_audio_tesis(audio_segmento, sr=16000):
    """
    Aplica la misma transformación que usamos para entrenar la red:
    Normalización -> Ajuste de tiempo -> Mel Spectrogram -> Z-Score.
    """
    # 1. Normalización de volumen
    if np.max(np.abs(audio_segmento)) > 0:
        audio_segmento = audio_segmento / np.max(np.abs(audio_segmento))
        
    # 2. Cazador de energía (Ignorar silencios profundos)
    intervalos = librosa.effects.split(audio_segmento, top_db=25)
    if len(intervalos) > 0:
        audio_limpio = audio_segmento[intervalos[0][0]:intervalos[-1][1]]
    else:
        audio_limpio = audio_segmento

    # 3. Forzar 1 segundo exacto (16000 muestras)
    target = int(sr * 1.0)
    if len(audio_limpio) > target:
        audio_limpio = audio_limpio[:target]
    else:
        pad = target - len(audio_limpio)
        audio_limpio = np.pad(audio_limpio, (pad//2, pad - pad//2))
        
    # 4. Espectrograma Mel (128x32)
    melspec = librosa.feature.melspectrogram(y=audio_limpio, sr=sr, n_mels=128, hop_length=512)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # 5. Estandarización Z-Score (Crítico)
    if melspec_db.std() != 0:
        melspec_norm = (melspec_db - melspec_db.mean()) / melspec_db.std()
    else:
        melspec_norm = melspec_db
        
    return melspec_norm[np.newaxis, ..., np.newaxis]

# --- 4. MOTOR DE DETECCIÓN ---
def detectar_muletillas_nn(ruta_audio):
    alertas = []
    if modelo_ia is None: return alertas
    
    try:
        # Cargamos el audio completo a 16000 Hz
        y, sr = librosa.load(ruta_audio, sr=16000)
        muestras_ventana = int(sr * 1.0) # Ventanas de 1 segundo
        
        # Escaneamos el audio saltando segundo a segundo
        for i in range(0, len(y) - muestras_ventana, muestras_ventana):
            segmento = y[i : i + muestras_ventana]
            
            # Filtro rápido para no procesar silencio absoluto y ahorrar CPU
            if np.max(np.abs(segmento)) < 0.05:
                continue
                
            # Extraer características y predecir
            feat = preprocesar_audio_tesis(segmento, sr)
            prediccion = modelo_ia.predict(feat, verbose=0)[0][0]
            
            # Umbral de decisión (Ajustable, 0.70 es un buen punto de partida)
            if prediccion > 0.70: 
                tiempo_segundos = i / sr
                minutos = int(tiempo_segundos // 60)
                segundos = int(tiempo_segundos % 60)
                
                alerta = f"[{minutos:02d}:{segundos:02d}] 🧠 Muletilla detectada ({prediccion*100:.1f}%)"
                alertas.append(alerta)
                
    except Exception as e:
        print(f"Error procesando el escaneo de audio: {e}")
        
    return alertas

# --- 5. EJECUCIÓN DE PRUEBA ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎙️ INICIANDO ESCANEO DE MULETILLAS")
    print("="*50)
    
    # Hemos cambiado config.ARCHIVO_AUDIO por tu archivo específico
    resultados = detectar_muletillas_nn("emeli.wav")
    
    for r in resultados: 
        print(r)
        
    if not resultados: 
        print("✅ Habla fluida. No se detectaron muletillas fonéticas.")
    print("="*50 + "\n")