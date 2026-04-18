# Auditor Multimodal de Oratoria con IA (Proyecto de Grado)

Este sistema es una herramienta de **Investigación Aplicada** diseñada para evaluar de forma objetiva las habilidades de oratoria mediante Inteligencia Artificial. Utiliza una arquitectura híbrida que integra el procesamiento en tiempo real con una auditoría posterior automatizada para generar diagnósticos detallados sobre el lenguaje corporal, la mirada, las emociones y la fluidez verbal.

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

El sistema opera de forma automatizada mediante un único punto de entrada. Al ejecutar el archivo principal, el sistema gestiona secuencialmente todas las etapas del análisis:

```bash
python main.py
```

### Flujo Automático de Procesamiento:
1. **Captura en Tiempo Real**: Inicia la grabación de video/audio y realiza el seguimiento de gestos adaptadores y postura.
2. **Auditoría Posterior (Post-Hoc)**: Al finalizar la grabación, el sistema activa automáticamente la orquestación para el cálculo de pose 3D, cinemática y emociones.
3. **Generación de Reporte**: Se exporta de forma inmediata un informe en PDF dentro de la carpeta `/reportes` con los resultados finales.

## Métricas Evaluadas

* **Kinésica**: Porcentaje de tiempo en postura cerrada (oclusión), inactividad gestual y balanceo excesivo (Sway).
* **Contacto Visual**: Estimación de la mirada mediante pose 3D (SolvePnP) basada en la proporción geométrica facial real (ojos, nariz, orejas y boca).
* **Biometría Emocional**: Análisis de afabilidad (Joyness), tensión facial (Ratio P2N) y expresividad global.
* **Análisis Acústico**: Detección de muletillas y fluidez del discurso.

## Condiciones y Recomendaciones



* **Set de Grabación**: La toma debe realizarse a una distancia aproximada de 2 a 3 metros de la computadora. Grabar a distancias cortas (ej. 1 metro) altera el comportamiento de los datos y afecta la precisión del seguimiento corporal.
* **Calidad de Audio**: El audio debe ser grabado con un dispositivo cercano, como un micrófono de solapa. El uso de micrófonos integrados a larga distancia degradará la calidad de los datos acústicos necesarios para el análisis.
* **Iluminación**: Se requiere luz frontal clara para la detección correcta de landmarks faciales y corporales.
* **Encuadre**: El orador debe estar centrado en un plano medio (cintura hacia arriba) para permitir el rastreo simultáneo de las manos, el torso y el rostro.
* **Posición de Cámara (Crítico para la Mirada)**: La cámara debe situarse **estrictamente a la altura de los ojos** del orador y en posición completamente frontal. El sistema utiliza un cálculo geométrico tridimensional puro; por lo tanto, grabar en ángulos de contrapicado (ej. desde la cámara inferior de una laptop en un escritorio) invalidará las proporciones faciales y generará falsos positivos de "mirada al techo" o "mirada al suelo".

---

**Desarrollado por**: Lisandro Cantarero

**Institución**: Universidad Nacional Autónoma de Honduras (UNAH)

**Carrera**: Ingeniería en Sistemas
