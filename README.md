# Auditor Multimodal de Oratoria con IA (Proyecto de Grado)

Este sistema es una herramienta de **Investigación Aplicada** diseñada para evaluar de forma objetiva las habilidades de oratoria mediante Inteligencia Artificial. Utiliza una arquitectura híbrida que combina procesamiento en tiempo real con una auditoría posterior para generar diagnósticos detallados sobre el lenguaje corporal, la mirada, las emociones y la fluidez verbal.

## Requisitos Mínimos del Sistema

Para garantizar la ejecución fluida de los modelos de Visión y Audio (WhisperX, DeepFace, MediaPipe), se recomienda:

* **Sistema Operativo**: Windows 10/11 o Linux.
* **Procesador**: Intel Core i5 de 10ma Gen o superior (o equivalente AMD).
* **Memoria RAM**: Mínimo 8 GB (16 GB recomendados para procesamiento de video largo).
* **GPU (Opcional)**: Compatible con CUDA para acelerar la transcripción con WhisperX.
* **Python**: Versión 3.10 o superior.

## Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/Lisandro-Cantarero/Proyecto_Oratoria.git
cd Proyecto_Oratoria
```

2. **Crear y activar el entorno virtual**:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # En Windows
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```
*Nota: La primera ejecución descargará automáticamente los modelos pre-entrenados de MediaPipe, DeepFace y WhisperX.*

## Modo de Uso

El sistema opera mediante una orquestación en etapas:

### Fase de Captura (Tiempo Real)
Ejecuta `main.py` para iniciar la grabación y el análisis de gestos adaptadores y postura.
```bash
python main.py
```

### Fase de Auditoría (Post-Hoc)
Una vez finalizada la grabación, el `orquestador_maestro.py` procesa los datos (pose 3D, cinemática y emociones).
```bash
python orquestador_maestro.py
```

### Generación de Reporte
El sistema exportará automáticamente un informe en PDF dentro de la carpeta `/reportes` con las métricas finales.

## Métricas Evaluadas

* **Kinésica**: Porcentaje de tiempo en postura cerrada (oclusión), inactividad gestual y balanceo excesivo (Sway).
* **Contacto Visual**: Estimación de la mirada mediante pose 3D (SolvePnP) con compensación de ángulo para laptops.
* **Biometría Emocional**: Análisis de afabilidad (Joyness), tensión facial (Ratio P2N) y expresividad global.
* **Análisis Acústico**: Detección de muletillas y fluidez del discurso.

## Condiciones y Recomendaciones

* **Iluminación**: Se requiere una fuente de luz frontal clara para que MediaPipe detecte los landmarks faciales y corporales correctamente.
* **Encuadre**: El orador debe estar centrado, preferiblemente de la cintura hacia arriba (plano medio), para permitir el seguimiento de las manos y el torso.
* **Posición de Cámara**: El sistema incluye un offset compensatorio de -12° para cámaras de laptops en escritorio.
* **Ambiente**: Se recomienda un fondo neutro para evitar ruidos visuales en el cálculo de la energía cinética.

---

**Desarrollado por**: Lisandro Cantarero

**Institución**: Universidad Nacional Autónoma de Honduras (UNAH)

**Carrera**: Ingeniería en Sistemas
