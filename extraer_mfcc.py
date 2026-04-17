import os
import librosa
import numpy as np
from tqdm import tqdm # Para ver la barra de progreso

# --- CONFIGURACIÓN ---
RUTA_DATASET = "Dataset_Aumentado" # Cambia esto por tu carpeta real
MAX_PAD_LEN = 32 # Longitud fija para 1 segundo de audio a 16kHz
N_MFCC = 40 # Número de filtros (40 es el estándar de alta resolución para voz)

def extraer_caracteristicas():
    etiquetas = ["0_Normal", "1_Muletillas"]
    X = [] # Aquí guardaremos las "imágenes" (MFCCs)
    y = [] # Aquí guardaremos las respuestas (0 o 1)

    print("🎙️ Iniciando extracción de características MFCC...")

    for indice_etiqueta, etiqueta in enumerate(etiquetas):
        ruta_carpeta = os.path.join(RUTA_DATASET, etiqueta)
        
        if not os.path.exists(ruta_carpeta):
            print(f"⚠️ No encuentro la carpeta: {ruta_carpeta}")
            continue

        archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.wav')]
        print(f"\nProcesando clase '{etiqueta}' ({len(archivos)} audios):")

        # tqdm nos da una barra de carga bonita en la terminal
        for archivo in tqdm(archivos):
            ruta_audio = os.path.join(ruta_carpeta, archivo)
            
            try:
                # 1. Cargar el audio
                audio, sr = librosa.load(ruta_audio, sr=16000)
                
                # 2. Extraer el MFCC
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
                
                # 3. Normalizar el tamaño (Padding/Truncating)
                # Si el audio duró un poquito menos o más del segundo exacto, lo cuadramos.
                if mfcc.shape[1] > MAX_PAD_LEN:
                    mfcc = mfcc[:, :MAX_PAD_LEN]
                else:
                    pad_width = MAX_PAD_LEN - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                
                # Guardar en las listas
                X.append(mfcc)
                y.append(indice_etiqueta) # 0 para Normal, 1 para Muletillas
                
            except Exception as e:
                print(f"❌ Error leyendo {archivo}: {e}")

    # Convertir a matrices de NumPy para TensorFlow
    X = np.array(X)
    y = np.array(y)

    # Las CNN de Keras necesitan que la imagen tenga "profundidad" (como los canales RGB)
    # Así que pasamos de (N, 40, 32) a (N, 40, 32, 1)
    X = X[..., np.newaxis]

    # Guardar los datos procesados
    print("\n💾 Guardando matrices en el disco...")
    np.save("X_datos.npy", X)
    np.save("y_etiquetas.npy", y)

    print(f"✅ ¡Éxito! Matriz X creada con forma: {X.shape}")
    print(f"✅ Matriz Y creada con forma: {y.shape}")

extraer_caracteristicas()