import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
import logging
from scipy.stats import kurtosis, skew

# Configurar logging para ver el progreso en consola de forma limpia
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analizar_emociones_oratoria(video_path, fps_deseado=1):
    """
    Analiza un video de oratoria para extraer intensidades emocionales, 
    alineado estrictamente a la metodología de Chen et al. (2015).
    Se descarta el análisis individual de emociones para enfocarse en 
    métricas de variación gruesa (P2N Ratio, Joyness, Total Emotion) 
    y sus estadísticos descriptivos (SD, Kurtosis, Skewness).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video en la ruta: {video_path}")

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original == 0:
        fps_original = 30 # Respaldo de seguridad
        
    frames_a_saltar = max(1, int(fps_original / fps_deseado))
    
    frame_count = 0
    datos_emociones = []

    logging.info(f"Iniciando análisis offline del video. Procesando 1 frame cada {frames_a_saltar} frames originales.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video

        # Muestreo a los FPS deseados (por defecto 1 frame por segundo para no saturar)
        if frame_count % frames_a_saltar == 0:
            try:
                # enforce_detection=True: Si el orador voltea completamente, falla y lo ignoramos (como en el paper)
                resultado = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, silent=True)
                
                # Manejo seguro si detecta múltiples rostros (tomamos el principal)
                if isinstance(resultado, list):
                    resultado = resultado[0]

                emociones = resultado['emotion']
                
                # Guardamos el instante temporal y los datos en crudo (escala original 0-100)
                fila = {
                    'segundo': frame_count / fps_original,
                    'angry': emociones.get('angry', 0),
                    'disgust': emociones.get('disgust', 0),
                    'fear': emociones.get('fear', 0),
                    'happy': emociones.get('happy', 0), # "Joyness" en el paper
                    'sad': emociones.get('sad', 0),
                    'surprise': emociones.get('surprise', 0),
                    'neutral': emociones.get('neutral', 0)
                }
                datos_emociones.append(fila)

            except ValueError:
                # Ocurre si el sujeto se voltea y no se detecta el rostro.
                # Chen et al. (2015) marcan estos datos como perdidos/ignorados.
                pass 
                
        frame_count += 1

    cap.release()
    logging.info("Extracción de frames finalizada. Generando métricas estadísticas...")
    
    # --- PROCESAMIENTO ESTADÍSTICO ACADÉMICO (Chen et al., 2015) ---
    df = pd.DataFrame(datos_emociones)
    
    if df.empty:
        raise ValueError("No se detectaron rostros en todo el video. Revisa la iluminación o el encuadre.")

    # 1. Normalización: de 0-100 a escala 0.0 - 1.0 (intensidad)
    emociones_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    df[emociones_cols] = df[emociones_cols] / 100.0

    # 2. Métricas compuestas: Emociones Negativas (excluyendo neutral) y Joyness
    emociones_negativas = ['angry', 'disgust', 'fear', 'sad', 'surprise']
    df['emocion_negativa_media'] = df[emociones_negativas].mean(axis=1)
    
    # 3. Métrica A: Total Emotion (Media entre Joyness y las emociones negativas)
    df['total_emotion'] = (df['happy'] + df['emocion_negativa_media']) / 2.0

    # 4. Métrica B: P2N Ratio (LOG-TRANSFORMED para control de ruido)
    # Usamos np.log1p (logaritmo natural de 1 + x) para suavizar los valores atípicos
    ratio_crudo = df['happy'] / (df['emocion_negativa_media'] + 1e-6)
    df['P2N_ratio'] = np.log1p(ratio_crudo)

    # 5. Extracción de Descriptores (SD, Kurtosis, Skewness) para las 3 métricas de interés
    # Se utiliza dropna() por seguridad estadística
    resumen = {
        'Joyness_Metrics': {
            'mean': float(df['happy'].mean()),
            'sd': float(df['happy'].std()),
            'kurtosis': float(kurtosis(df['happy'].dropna())),
            'skewness': float(skew(df['happy'].dropna()))
        },
        'Total_Emotion_Metrics': {
            'mean': float(df['total_emotion'].mean()),
            'sd': float(df['total_emotion'].std()),
            'kurtosis': float(kurtosis(df['total_emotion'].dropna())),
            'skewness': float(skew(df['total_emotion'].dropna()))
        },
        'P2N_Ratio_Metrics': {
            'mean': float(df['P2N_ratio'].mean()),
            'sd': float(df['P2N_ratio'].std()),
            'kurtosis': float(kurtosis(df['P2N_ratio'].dropna())),
            'skewness': float(skew(df['P2N_ratio'].dropna()))
        },
        'estadisticas_generales': {
            'total_frames_analizados': len(df),
            'duracion_analizada_segundos': round(len(df) * (1.0 / fps_deseado), 2)
        },
        # Se retorna el dataframe completo por si necesitas gráficas temporales en tu tesis
        'dataframe_crudo': df 
    }
    
    return resumen

# ================= USO DE PRUEBA =================
if __name__ == "__main__":
    
    ruta_video = "C:/ruta/a/tu/video_prueba.mp4" 
    
    try:
        resultados = analizar_emociones_oratoria(ruta_video, fps_deseado=1)
        
        print("\n" + "="*50)
        print("REPORTE DE EXPRESIVIDAD FACIAL (Chen et al., 2015)")
        print("="*50)
        
        print("\n--- RATIO POSITIVO/NEGATIVO (P2N) ---")
        print(f"  > P2N Mean:     {resultados['P2N_Ratio_Metrics']['mean']:.4f}")
        print(f"  > P2N SD:       {resultados['P2N_Ratio_Metrics']['sd']:.4f}")
        print(f"  > P2N Kurtosis: {resultados['P2N_Ratio_Metrics']['kurtosis']:.4f}")
        print(f"  > P2N Skewness: {resultados['P2N_Ratio_Metrics']['skewness']:.4f}")

        print("\n--- ALEGRÍA / APERTURA (Joyness) ---")
        print(f"  > Joyness Mean:     {resultados['Joyness_Metrics']['mean']:.4f}")
        print(f"  > Joyness SD:       {resultados['Joyness_Metrics']['sd']:.4f}")
        print(f"  > Joyness Kurtosis: {resultados['Joyness_Metrics']['kurtosis']:.4f}")
        print(f"  > Joyness Skewness: {resultados['Joyness_Metrics']['skewness']:.4f}")
        
        print("\n--- TOTAL EMOTION (Variación Expresiva Global) ---")
        print(f"  > Total Mean:     {resultados['Total_Emotion_Metrics']['mean']:.4f}")
        print(f"  > Total SD:       {resultados['Total_Emotion_Metrics']['sd']:.4f}")
        print(f"  > Total Kurtosis: {resultados['Total_Emotion_Metrics']['kurtosis']:.4f}")
        print(f"  > Total Skewness: {resultados['Total_Emotion_Metrics']['skewness']:.4f}")
        
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")