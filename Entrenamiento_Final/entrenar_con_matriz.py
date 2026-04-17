import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
print("📦 Cargando matrices de datos...")
try:
    X = np.load("X_datos.npy")
    y = np.load("y_etiquetas.npy")
except FileNotFoundError:
    print("❌ Error: No se encuentran los archivos X_datos.npy o y_etiquetas.npy")
    exit()

# 2. Dividir los datos (80% Entrenamiento, 20% Prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🧠 Entrenando con: {X_train.shape[0]} audios")
print(f"🧪 Probando con: {X_test.shape[0]} audios")

# 3. Construir la CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 32, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 4. Entrenar el modelo
print("\n🚀 Iniciando entrenamiento...")
history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1)

# 5. Evaluar
print("\n📊 Evaluando con datos nunca vistos...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Precisión Final: {test_acc * 100:.2f}%")

# ==========================================
# 6. GENERAR MATRIZ DE CONFUSIÓN (PDF VECTORIAL)
# ==========================================
print("\n🖼️ Generando Matriz de Confusión en formato PDF...")

y_pred_probs = model.predict(X_test)
y_pred_classes = (y_pred_probs >= 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.set_theme(font_scale=1.2)

ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                 xticklabels=['Habla Normal (0)', 'Muletilla (1)'],
                 yticklabels=['Habla Normal (0)', 'Muletilla (1)'])

plt.title('Matriz de Confusión del Modelo CNN\n', fontsize=16, fontweight='bold')
plt.xlabel('\nPredicción del Modelo', fontsize=14)
plt.ylabel('Valor Real (Verdad Terrestre)', fontsize=14)

# CAMBIO CLAVE: Guardar como PDF para calidad infinita en LaTeX
nombre_pdf = "matriz_confusion_tesis.pdf"
plt.savefig(nombre_pdf, format='pdf', bbox_inches='tight')
print(f"✅ ¡Gráfico vectorial guardado con éxito como '{nombre_pdf}'!")

# Guardar el modelo
model.save("modelo_muletillas_v2.keras")
print("💾 Modelo guardado como 'modelo_muletillas_v2.keras'")

# Reporte detallado
print("\n📑 Reporte de Clasificación:")
print(classification_report(y_test, y_pred_classes, target_names=['Normal', 'Muletillas']))