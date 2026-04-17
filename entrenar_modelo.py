import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DE PARÁMETROS
# ==========================================
DATASET_PATH = "Dataset_Aumentado"
SR = 16000           # Frecuencia de muestreo estandarizada
DURATION = 1.0       # 1 segundo exacto
N_MELS = 128         # Resolución de frecuencias en el espectrograma
MAX_PAD_LEN = 32     # Aproximadamente 1 seg con hop_length=512

# ==========================================
# 2. EXTRACCIÓN DE CARACTERÍSTICAS (LIBROSA)
# ==========================================
def extract_features(file_path):
    try:
        # Cargar el audio
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        
        # Rellenar con ceros (silencio) si el audio dura menos de 1 segundo
        target_length = int(SR * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
        # Generar Espectrograma Mel
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Asegurar un tamaño exacto (128, 32)
        if log_mel.shape[1] > MAX_PAD_LEN:
            log_mel = log_mel[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - log_mel.shape[1]
            log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return log_mel
    except Exception as e:
        print(f"❌ Error en el archivo {file_path}: {e}")
        return None

# ==========================================
# 3. CARGA DE DATOS Y ETIQUETADO
# ==========================================
print("🔄 Extrayendo Espectrogramas Mel. Esto tomará un par de minutos...")
X = []
y = []

# Mapeo de carpetas
clases = {"0_Normal": 0, "1_Muletillas": 1}

for carpeta, etiqueta in clases.items():
    ruta_carpeta = os.path.join(DATASET_PATH, carpeta)
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(('.wav', '.mp4', '.m4a', '.ogg')):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            features = extract_features(ruta_archivo)
            
            if features is not None:
                X.append(features)
                y.append(etiqueta)

# Convertir a tensores de Numpy
X = np.array(X)
y = np.array(y)

# Añadir la dimensión del "canal" para que la CNN lo vea como una imagen en escala de grises
# El shape pasará de (N, 128, 32) a (N, 128, 32, 1)
X = X[..., np.newaxis] 

print(f"✅ Datos cargados: {X.shape[0]} muestras. Shape del tensor: {X.shape}")

# Dividir en Entrenamiento (80%) y Validación (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 4. ARQUITECTURA DE LA CNN
# ==========================================
model = models.Sequential([
    # Capa de entrada y primera extracción de bordes acústicos
    layers.Input(shape=(N_MELS, MAX_PAD_LEN, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    
    # Segunda capa convolucional (Patrones más complejos como formantes)
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    
    # Tercera capa convolucional
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    
    # Aplanado y toma de decisiones
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid') # Sigmoid porque es clasificación binaria (0 o 1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 5. ENTRENAMIENTO CON SALVAVIDAS (CALLBACKS)
# ==========================================
# Guardar siempre el mejor modelo, no el de la última época
checkpoint = callbacks.ModelCheckpoint('modelo_muletillas_v2.keras', save_best_only=True, monitor='val_accuracy', mode='max')
# Detenerse si la red ya no aprende (evita Overfitting)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n🚀 Iniciando el entrenamiento de la Red Neuronal...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

print("\n🎉 Entrenamiento Finalizado. Modelo guardado como 'modelo_muletillas_v2.keras'")