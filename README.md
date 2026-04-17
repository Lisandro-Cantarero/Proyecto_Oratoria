````markdown
# Auditor Multimodal de Oratoria con IA (Proyecto de Grado)

Este sistema es una herramienta de investigación aplicada diseñada para evaluar de forma objetiva las habilidades de oratoria mediante inteligencia artificial. Utiliza una arquitectura híbrida que combina procesamiento en tiempo real con un análisis posterior para generar diagnósticos detallados sobre el lenguaje corporal, la mirada, las emociones y la fluidez verbal.

## Requisitos mínimos del sistema

Para garantizar la ejecución fluida de los modelos de visión y audio (WhisperX, DeepFace, MediaPipe), se recomienda:

* **Sistema operativo**: Windows 10/11 o Linux.  
* **Procesador**: Intel Core i5 de 10.ª generación o superior (o equivalente AMD).  
* **Memoria RAM**: Mínimo 8 GB (16 GB recomendados para procesamiento de video largo).  
* **GPU (opcional)**: Compatible con CUDA para acelerar la transcripción con WhisperX.  
* **Python**: Versión 3.10 o superior.  

## Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/Lisandro-Cantarero/Proyecto_Oratoria.git
cd Proyecto_Oratoria
````

2. **Crear y activar el entorno virtual**:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # En Windows
```

3. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

*Nota: La primera ejecución descargará automáticamente los modelos preentrenados de MediaPipe, DeepFace y WhisperX.*

## Modo de uso

El sistema opera mediante una ejecución en etapas:

### Fase de captura (tiempo real)

Ejecuta `main.py` para iniciar la grabación y el análisis de gestos y postura.

```bash
python main.py
```

### Fase de análisis posterior

Una vez finalizada la grabación, el `orquestador_maestro.py` procesa los datos generados (pose 3D, movimiento corporal y emociones).

```bash
python orquestador_maestro.py
```

### Generación de reporte

El sistema exportará automáticamente un informe en PDF dentro de la carpeta `/reportes` con las métricas finales.

## Métricas evaluadas

* **Kinésica**: Porcentaje de tiempo en postura cerrada (oclusión), inactividad gestual y balanceo excesivo (sway).
* **Contacto visual**: Estimación de la mirada mediante pose 3D (SolvePnP) con compensación de ángulo para laptops.
* **Emociones**: Análisis de expresividad, nivel de agrado (joy) y tensión facial.
* **Análisis acústico**: Detección de muletillas y evaluación de la fluidez del discurso.

## Condiciones y recomendaciones

* **Iluminación**: Se requiere una fuente de luz frontal clara para que MediaPipe detecte correctamente los puntos faciales y corporales.
* **Encuadre**: El orador debe estar centrado, preferiblemente de la cintura hacia arriba (plano medio), para permitir el seguimiento de manos y torso.
* **Posición de cámara**: El sistema incluye un ajuste aproximado de -12° para cámaras de laptops en escritorio.
* **Ambiente**: Se recomienda un fondo neutro para evitar interferencias en el análisis visual.

---

**Desarrollado por**: Lisandro Cantarero
**Institución**: Universidad Nacional Autónoma de Honduras (UNAH)
**Carrera**: Ingeniería en Sistemas

```
```
