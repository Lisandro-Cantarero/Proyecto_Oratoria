import cv2
import os
import glob
import numpy as np
import pandas as pd
import warnings

# Suprimir advertencias molestas de TensorFlow/DeepFace
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from deepface import DeepFace

def procesar_emociones_video(ruta_video, frames_salto=30):
    """
    Escanea un video extrayendo 1 frame cada X frames (por defecto 30, aprox 1 fps)
    y calcula E_total y P2N Ratio según el Marco Teórico.
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return None

    totales_e_total = []
    totales_p2n = []
    totales_alegria = []

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Procesar solo 1 de cada 30 frames para hacerlo súper rápido
        if frame_count % frames_salto == 0:
            try:
                # Convertir a RGB para DeepFace
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, silent=True)
                
                emociones = resultado[0]['emotion']
                
                # Extraer las 7 emociones base
                joy = emociones.get('happy', 0.0)
                sad = emociones.get('sad', 0.0)
                angry = emociones.get('angry', 0.0)
                fear = emociones.get('fear', 0.0)
                disgust = emociones.get('disgust', 0.0)
                surprise = emociones.get('surprise', 0.0)
                neutral = emociones.get('neutral', 0.0)
                
                # Aplicar Ecuaciones 5.1 y 5.2 del Marco Teórico
                e_pos = joy
                e_neg = (sad + angry + fear + disgust + surprise + neutral) / 6.0
                
                e_total = (e_pos + e_neg) / 2.0
                p2n_ratio = e_pos / e_neg if e_neg > 0 else e_pos
                
                totales_e_total.append(e_total)
                totales_p2n.append(p2n_ratio)
                totales_alegria.append(joy)
                
            except Exception as e:
                pass # Si DeepFace falla en un frame borroso, ignorar

        frame_count += 1
        
    cap.release()

    # Si no se detectó ningún rostro en todo el video, devolver ceros
    if not totales_e_total:
        return {'Emocion_Total': 0.0, 'Ratio_P2N': 0.0, 'Alegria_Media': 0.0}

    # Devolver promedios del video completo
    return {
        'Emocion_Total': np.mean(totales_e_total),
        'Ratio_P2N': np.mean(totales_p2n),
        'Alegria_Media': np.mean(totales_alegria)
    }

def inyectar_emociones_al_dataset():
    # Ruta absoluta al archivo maestro
    ruta_csv = r'C:\proyecto_oratoria\datos_exportados\dataset_multimodal_maestro.csv'
    carpeta_videos = 'videos_entrada'
    
    print(f"📥 Cargando dataset: {ruta_csv}...")
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el dataset en la ruta: {ruta_csv}")
        return

    # 1. FILTRO METODOLÓGICO: Solo videos > 110 segundos
    df_filtrado = df[df['Duracion_Segundos'] > 110].copy()
    ids_validos = df_filtrado['ID_Orador'].values
    
    print(f"✅ Filtro aplicado: Procesando solo videos > 110s ({len(ids_validos)} videos válidos en el CSV).")

    # Asegurar que existan las columnas de emoción en el dataframe
    for col in ['Emocion_Total', 'Ratio_P2N', 'Alegria_Media']:
        if col not in df.columns:
            df[col] = np.nan

    videos = glob.glob(os.path.join(carpeta_videos, '*.mp4'))
    total_videos = len(videos)
    
    print(f"🎥 Se encontraron {total_videos} videos físicos en la carpeta '{carpeta_videos}'...\n")

    for i, ruta_video in enumerate(videos, 1):
        nombre_archivo = os.path.basename(ruta_video)
        id_orador = os.path.splitext(nombre_archivo)[0]
        
        # 2. VERIFICACIÓN DOBLE: ¿Está en el CSV y cumple los 110s?
        if id_orador in ids_validos:
            print(f"🚀 [{i}/{total_videos}] Analizando emociones de: {id_orador}...")
            
            metricas = procesar_emociones_video(ruta_video, frames_salto=30)
            
            if metricas:
                # Inyectar datos en la fila correspondiente
                idx = df.index[df['ID_Orador'] == id_orador].tolist()[0]
                df.at[idx, 'Emocion_Total'] = metricas['Emocion_Total']
                df.at[idx, 'Ratio_P2N'] = metricas['Ratio_P2N']
                df.at[idx, 'Alegria_Media'] = metricas['Alegria_Media']
        else:
            # Informar por qué se salta (ya sea por duración o por no estar en el CSV)
            print(f"⚠️ [OMITIDO] {id_orador} (Dura menos de 110s o no pertenece a la muestra).")

    # Guardar el resultado final
    ruta_salida = 'dataset_multimodal_maestro_FINAL_EMOCIONES.csv'
    df.to_csv(ruta_salida, index=False)
    print(f"\n✨ ¡Listo! Se generó el dataset final con emociones para la tesis: {ruta_salida}")

if __name__ == "__main__":
    inyectar_emociones_al_dataset()