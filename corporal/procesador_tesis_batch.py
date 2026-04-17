import cv2
import json
import time
import os
import csv
import pandas as pd
import mediapipe as mp
import logging

# Importar tus módulos de grado 
from analizador_cinetico import AnalizadorCinetico
from analizador_behavioral import AnalizadorBehavioral
from analizador_cabeza import AnalizadorMirada
from self_adaptors import AnalizadorSelfAdaptors
from gestos import analizar_emociones_oratoria

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def exportar_archivos_vision(id_video, reporte_maestro, incidencias, carpeta_salida):
    """
    Exporta la 'Trinidad' de archivos: JSON maestro, CSV de eventos temporales 
    y actualiza el CSV global de características (Machine Learning).
    """
    # Definir rutas de salida
    json_maestro = os.path.join(carpeta_salida, f"{id_video}_maestro_vision.json")
    csv_eventos = os.path.join(carpeta_salida, f"{id_video}_eventos_vision.csv")
    csv_global = os.path.join(carpeta_salida, "dataset_global_vision.csv")

    # 1. Exportar JSON MAESTRO (Detalle Profundo)
    with open(json_maestro, "w", encoding='utf-8') as f:
        json.dump(reporte_maestro, f, indent=4, ensure_ascii=False)

    # 2. Exportar CSV de EVENTOS (Línea de tiempo para fusionar con Audio)
    with open(csv_eventos, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["segundo_exacto", "modalidad", "incidencia"])
        for evt in incidencias:
            writer.writerow([evt["segundo_exacto"], evt["modalidad"], evt["incidencia"]])

    # 3. Actualizar DATASET GLOBAL (Extracción segura con .get)
    metricas = reporte_maestro.get("metricas_globales", {})
    comp = metricas.get("comportamiento", {})
    mirada = metricas.get("mirada_y_cabeza", {})
    nervios = metricas.get("nerviosismo", {})
    postura = metricas.get("postura_pTM", {})
    emos = metricas.get("emociones_faciales", {}).get("P2N_Ratio_Metrics", {})

    nueva_fila = {
        "id_orador": id_video,
        "duracion_seg": reporte_maestro["metadata_orador"]["duracion_video_segundos"],
        # Métricas Comportamentales
        "energy_mean": comp.get("motion_energy_mean", 0),
        "stage_coverage": comp.get("total_stage_coverage", 0),
        "pct_zona_valida": comp.get("porcentaje_tiempo_zona_valida", 0),
        # Métricas de Contacto Visual
        "gaze_audiencia_pct": mirada.get("OpenOPAF", {}).get("porcentaje_mirada_audiencia", 0),
        "gaze_real_pct": mirada.get("Rastreo_Ocular_Secundario", {}).get("porcentaje_contacto_visual_real", 0),
        "head_yaw_sd": mirada.get("Chen_Metrics", {}).get("Face_Left_Right_Sd", 0),
        "head_pitch_sd": mirada.get("Chen_Metrics", {}).get("Face_Up_Down_Sd", 0),
        # Métricas de Nerviosismo Motriz
        "total_frotamiento_manos": nervios.get("Total_Frotamientos_Manos", 0),
        "total_toques_faciales": nervios.get("Total_Toques_Faciales", 0),
        "ptm_postura_cerrada": postura.get("pTM_postura", 0),
        # Emociones (Extraído de DeepFace)
        "emocion_p2n_mean": emos.get("mean", 0) 
    }

    # Añadir al CSV global (crea el archivo y encabezados si no existe)
    file_exists = os.path.isfile(csv_global)
    df = pd.DataFrame([nueva_fila])
    df.to_csv(csv_global, mode='a', index=False, header=not file_exists)

    logging.info(f"✓ Archivos exportados: {json_maestro} | {csv_eventos}")


def procesar_video_oratoria(ruta_video, nombre_orador, carpeta_salida):
    """
    Procesador Batch Headless. Extrae métricas de MediaPipe frame a frame,
    registra la línea de tiempo de incidencias y delega el análisis de emociones a DeepFace.
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        logging.error(f"No se pudo abrir el video: {ruta_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_total = total_frames / fps if fps > 0 else 0
    
    logging.info(f"\n=== INICIANDO ANÁLISIS MULTIMODAL: {nombre_orador} ===")
    logging.info(f"Resolución/FPS: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {fps} FPS")
    logging.info(f"Duración: {duracion_total:.2f} segundos")

    # --- INICIALIZAR MOTORES CON RIGOR METODOLÓGICO ---
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=2, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True # <--- VITAL PARA EL RASTREO DEL IRIS
    )
    mp_pose = mp.solutions.pose

    # Instanciar analizadores
    cinetico = AnalizadorCinetico()
    behavioral = AnalizadorBehavioral()
    mirada = AnalizadorMirada()
    adaptors = AnalizadorSelfAdaptors()

    # --- TIMELINE DE INCIDENCIAS (Para cruzar con Audio/Whisper) ---
    log_incidencias = []
    
    # Memorias de estado para el Detector de Flancos (evitar spam en el log)
    prev_estado_cinetico = {
        "postura_cerrada": False, 
        "balanceo_excesivo": False, 
        "inactividad_gestual": False, 
        "cuerpo_de_perfil": False
    }
    prev_alerta_facial = None
    prev_alerta_manos = None

    inicio_procesamiento = time.time()
    frames_procesados = 0

    logging.info("1/2: Extrayendo cinemática, postura y mirada (MediaPipe)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        timestamp_actual = frames_procesados / fps
        frame_shape = frame.shape[:2] # (alto, ancho)

        # Inferencia MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        if results.pose_landmarks:
            # 1. Cuerpo (Cinético)
            estado_cinetico = cinetico.procesar_frame(results.pose_landmarks, mp_pose, timestamp_actual)
            
            # Registrar SOLO cuando un error inicia (Edge Detection)
            for anomalia, activo in estado_cinetico.items():
                if activo and not prev_estado_cinetico[anomalia]:
                    log_incidencias.append({
                        "segundo_exacto": round(timestamp_actual, 2),
                        "modalidad": "Postura",
                        "incidencia": anomalia
                    })
            prev_estado_cinetico = estado_cinetico.copy()

            # 2. Comportamiento (Energía y Cobertura)
            behavioral.procesar_frame_behavioral(results.pose_landmarks, mp_pose)

            # 3. Cabeza y Mirada (GGDE - OpenOPAF)
            mirada.procesar_frame_cabeza(results.face_landmarks, frame_shape)

            # 4. Nerviosismo (Self-Adaptors y Fidgeting)
            alerta_facial, alerta_manos = adaptors.procesar_frame_adaptors(
                results.face_landmarks, 
                results.right_hand_landmarks, 
                results.left_hand_landmarks, 
                timestamp_actual, 
                frame_shape
            )

            # Registro de Toques en el rostro
            if alerta_facial and not prev_alerta_facial:
                log_incidencias.append({
                    "segundo_exacto": round(timestamp_actual, 2),
                    "modalidad": "Nerviosismo",
                    "incidencia": f"toque_facial_{adaptors.zona_contacto}" if adaptors.zona_contacto else "toque_facial"
                })
            prev_alerta_facial = alerta_facial

            # Registro de Frotamiento de manos
            if alerta_manos and not prev_alerta_manos:
                log_incidencias.append({
                    "segundo_exacto": round(timestamp_actual, 2),
                    "modalidad": "Nerviosismo",
                    "incidencia": "frotamiento_manos"
                })
            prev_alerta_manos = alerta_manos

        frames_procesados += 1
        
        # Indicador de progreso cada 10%
        if total_frames > 0 and frames_procesados % max(1, (total_frames // 10)) == 0:
            porcentaje = (frames_procesados / total_frames) * 100
            logging.info(f"Progreso MediaPipe: {porcentaje:.0f}%")

    cap.release()
    holistic.close()

    # --- ANÁLISIS DE EMOCIONES OFFLINE (DeepFace) ---
    logging.info("2/2: Extrayendo estadísticas emocionales (DeepFace)...")
    try:
        reporte_emociones = analizar_emociones_oratoria(ruta_video, fps_deseado=1)
        if 'dataframe_crudo' in reporte_emociones:
            del reporte_emociones['dataframe_crudo']
    except Exception as e:
        logging.error(f"Fallo en DeepFace: {e}")
        reporte_emociones = {"error": str(e)}

    # --- ENSAMBLAJE DEL REPORTE MAESTRO ---
    logging.info("Ensamblando y Exportando Reporte Multimodal...")
    reporte_final = {
        "metadata_orador": {
            "id": nombre_orador,
            "duracion_video_segundos": round(duracion_total, 2),
            "tiempo_procesamiento_segundos": round(time.time() - inicio_procesamiento, 2)
        },
        "metricas_globales": {
            "postura_pTM": cinetico.finalizar_sesion(duracion_total),
            "comportamiento": behavioral.generar_reporte_post_hoc(),
            "mirada_y_cabeza": mirada.generar_reporte_post_hoc(),
            "nerviosismo": adaptors.generar_reporte_post_hoc(),
            "emociones_faciales": reporte_emociones
        }
    }

    # --- LLAMADA A LA EXPORTACIÓN ---
    exportar_archivos_vision(nombre_orador, reporte_final, log_incidencias, carpeta_salida)

if __name__ == "__main__":
    # 1. Configurar directorios
    carpeta_videos = "./videos"
    carpeta_reportes = "./reportes_generados"
    
    # Crear las carpetas si no existen en tu sistema
    if not os.path.exists(carpeta_videos):
        os.makedirs(carpeta_videos)
        logging.info(f"Se creó la carpeta '{carpeta_videos}'. Por favor, coloca tus videos ahí y vuelve a ejecutar.")
        exit()

    if not os.path.exists(carpeta_reportes):
        os.makedirs(carpeta_reportes)

    # 2. Extensiones de video permitidas
    extensiones = (".mp4", ".mkv", ".avi", ".mov")

    # 3. Listar y procesar
    lista_videos = [f for f in os.listdir(carpeta_videos) if f.lower().endswith(extensiones)]
    
    if not lista_videos:
        logging.warning(f"No se encontraron videos en la carpeta '{carpeta_videos}'.")
    else:
        logging.info(f"--- INICIANDO PROCESAMIENTO MASIVO DE {len(lista_videos)} VIDEOS ---")

        for video_archivo in lista_videos:
            ruta_completa = os.path.join(carpeta_videos, video_archivo)
            sujeto_id = os.path.splitext(video_archivo)[0]
            
            try:
                procesar_video_oratoria(ruta_completa, sujeto_id, carpeta_reportes)
            except Exception as e:
                logging.error(f"X Error fatal procesando {sujeto_id}: {e}")

        logging.info("\n--- TODOS LOS ORADORES HAN SIDO PROCESADOS ---")