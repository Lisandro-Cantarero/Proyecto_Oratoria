import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
from moviepy.editor import VideoFileClip

# Importar tu módulo
from modulo_disfluencias_cnn import detectar_muletillas_acusticas

warnings.filterwarnings("ignore")

CARPETA_VIDEOS = "videos_validacion" # Crea esta carpeta y mete tus 5-10 videos ahí

def extraer_audio(ruta_video, ruta_audio_temp="temp_audio.wav"):
    """Extrae el audio del video mp4 para pasarlo a librosa"""
    try:
        video = VideoFileClip(ruta_video)
        video.audio.write_audiofile(ruta_audio_temp, logger=None)
        y, sr = librosa.load(ruta_audio_temp, sr=16000)
        os.remove(ruta_audio_temp)
        return y, sr
    except Exception as e:
        print(f"Error procesando {ruta_video}: {e}")
        return None, None

def formatear_tiempo(segundos):
    m, s = divmod(segundos, 60)
    return f"{int(m):02d}:{int(s):02d}"

def auditar_sistema():
    if not os.path.exists(CARPETA_VIDEOS):
        os.makedirs(CARPETA_VIDEOS)
        print(f"📁 Carpeta '{CARPETA_VIDEOS}' creada. Mete tus videos de prueba ahí y vuelve a correr el script.")
        return

    archivos = [f for f in os.listdir(CARPETA_VIDEOS) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not archivos:
        print(f"⚠️ No hay videos en la carpeta '{CARPETA_VIDEOS}'.")
        return

    total_tp = 0 # Verdaderos Positivos (CNN acertó)
    total_fp = 0 # Falsos Positivos (CNN se equivocó, no era muletilla)
    total_fn = 0 # Falsos Negativos (CNN se quedó callada y el orador dijo "ehh")
    total_segmentos_analizados = 0

    print("="*50)
    print(" INICIANDO AUDITORÍA MANUAL DE LA CNN ".center(50))
    print("="*50)

    for archivo in archivos:
        ruta = os.path.join(CARPETA_VIDEOS, archivo)
        print(f"\n🎥 Procesando: {archivo}...")
        
        y, sr = extraer_audio(ruta)
        if y is None: continue

        duracion_seg = len(y) / sr
        # Estimamos la cantidad de "segmentos de decisión" de 1.5s en el video
        total_segmentos_analizados += int(duracion_seg / 1.5)

        # Usar tu modelo
        resultados = detectar_muletillas_acusticas(y, sr)
        
        if "error" in resultados:
            print(f"Error de CNN: {resultados['error']}")
            continue

        eventos = resultados["eventos_detallados"]
        cnn_count = resultados["total_muletillas_acusticas"]

        print(f"🤖 La CNN detectó: {cnn_count} muletillas.")
        if cnn_count > 0:
            tiempos = [formatear_tiempo(e['inicio']) for e in eventos]
            print(f"   Tiempos: {', '.join(tiempos)}")

        # Interacción con el usuario (Ground Truth)
        try:
            tp = int(input(f"➤ Escucha el video. De las {cnn_count} detectadas, ¿cuántas fueron REALES (Aciertos / TP)?: "))
            fn = int(input(f"➤ ¿Cuántas muletillas evidentes OMITIÓ la CNN (Falsos Negativos / FN)?: "))
            
            # Los Falsos positivos son las que la CNN detectó, menos las que realmente eran
            fp = cnn_count - tp
            if fp < 0: fp = 0

            total_tp += tp
            total_fp += fp
            total_fn += fn

        except ValueError:
            print("Entrada inválida. Saltando video...")
            continue

    # Calcular Verdaderos Negativos (Estimación de los silencios/habla fluida donde la CNN acertó en no decir nada)
    total_tn = total_segmentos_analizados - (total_tp + total_fp + total_fn)

    # ---------------------------------------------------------
    # GENERACIÓN DE LA MATRIZ Y MÉTRICAS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" RESULTADOS FINALES DE LA AUDITORÍA ".center(50))
    print("="*50)

    # Evitar división por cero
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"✅ Verdaderos Positivos (Aciertos): {total_tp}")
    print(f"❌ Falsos Positivos (Falsas Alarmas): {total_fp}")
    print(f"⚠️ Falsos Negativos (Omisiones): {total_fn}")
    print("-"*50)
    print(f"🎯 Precisión (Precision):  {precision:.2%}")
    print(f"🔎 Exhaustividad (Recall): {recall:.2%}")
    print(f"🏆 F1-Score:               {f1:.2%}")

    # Reconstruir arrays para sklearn
    y_true = [1]*total_tp + [0]*total_fp + [1]*total_fn + [0]*total_tn
    y_pred = [1]*total_tp + [1]*total_fp + [0]*total_fn + [0]*total_tn

    # Graficar Matriz
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Habla Fluida/Silencio', 'Muletilla'],
                yticklabels=['Habla Fluida/Silencio', 'Muletilla'])
    
    plt.xlabel('Predicción de la CNN (Sistema)', fontsize=12, fontweight='bold')
    plt.ylabel('Realidad (Auditor Humano)', fontsize=12, fontweight='bold')
    plt.title(f'Matriz de Confusión: CNN Acústica\n(F1-Score: {f1:.2f})', fontsize=14)
    plt.tight_layout()
    plt.savefig('matriz_confusion_cnn.png', dpi=300)
    print("\n📸 Gráfica 'matriz_confusion_cnn.png' guardada con éxito.")

if __name__ == "__main__":
    auditar_sistema()