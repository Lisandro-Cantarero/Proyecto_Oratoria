import os
import time
import glob
import traceback
import librosa
import numpy as np
import pandas as pd
import csv
import torch
import whisperx

# ==========================================
# IMPORTACIONES DE AUDIO (Carpeta 'modular/')
# ==========================================
from modular.preprocesamiento_acustico import procesar_audio_oratoria
from modular.modulo_ritmo_transcripcion import calcular_ritmo_whisper
from modular.modulo_telemetria_local import extraer_telemetria_local
from modular.modulo_prosodia_volumen import analizar_prosodia_global
from modular.modulo_muletillas_lexicas import analizar_lexico_y_taxonomia
from modular.modulo_disfluencias_cnn import detectar_muletillas_acusticas
from modular.motor_rubrica_evaluacion import generar_veredicto
from modular import exportador_datos_crudos as exportador

# ==========================================
# IMPORTACIONES DE VISIÓN
# ==========================================
try:
    import cv2
    import mediapipe as mp
    
    try:
        from analizador_cinetico import AnalizadorCinetico
        from analizador_behavioral import AnalizadorBehavioral
        from analizador_cabeza import AnalizadorMirada
        from self_adaptors import AnalizadorSelfAdaptors
        from gestos import analizar_emociones_oratoria
    except ImportError:
        from corporal.analizador_cinetico import AnalizadorCinetico
        from corporal.analizador_behavioral import AnalizadorBehavioral
        from corporal.analizador_cabeza import AnalizadorMirada
        from corporal.self_adaptors import AnalizadorSelfAdaptors
        from corporal.gestos import analizar_emociones_oratoria

    VISION_ACTIVA = True
except ImportError as e:
    VISION_ACTIVA = False
    print(f"⚠️ Módulos de visión no encontrados. Operando en Solo Audio. Detalle: {e}")

# ==========================================
# MOTOR DE VISIÓN
# ==========================================
def extraer_datos_vision(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_total = total_frames / fps if fps > 0 else 0

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=2, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )
    mp_pose = mp.solutions.pose

    cinetico = AnalizadorCinetico()
    behavioral = AnalizadorBehavioral()
    mirada = AnalizadorMirada()
    adaptors = AnalizadorSelfAdaptors()

    frames_procesados = 0
    incidencias_vision = []
    
    prev_cinetico = {"postura_cerrada": False, "balanceo_excesivo": False, "inactividad_gestual": False, "cuerpo_de_perfil": False}
    prev_alerta_facial = None
    prev_alerta_manos = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        timestamp_actual = frames_procesados / fps
        frame_shape = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        if results.pose_landmarks:
            est_cinetico = cinetico.procesar_frame(results.pose_landmarks, mp_pose, timestamp_actual)
            for anomalia, activo in est_cinetico.items():
                if activo and not prev_cinetico[anomalia]:
                    incidencias_vision.append({"tiempo": timestamp_actual, "tipo": "Postura", "detalle": anomalia})
            prev_cinetico = est_cinetico.copy()

            behavioral.procesar_frame_behavioral(results.pose_landmarks, mp_pose)
            mirada.procesar_frame_cabeza(results.face_landmarks, frame_shape)
            
            alerta_facial, alerta_manos = adaptors.procesar_frame_adaptors(
                results.face_landmarks, results.right_hand_landmarks, results.left_hand_landmarks, timestamp_actual, frame_shape
            )
            
            if alerta_facial and not prev_alerta_facial:
                incidencias_vision.append({"tiempo": timestamp_actual, "tipo": "Nerviosismo", "detalle": "toque_facial"})
            prev_alerta_facial = alerta_facial

            if alerta_manos and not prev_alerta_manos:
                incidencias_vision.append({"tiempo": timestamp_actual, "tipo": "Nerviosismo", "detalle": "frotamiento_manos"})
            prev_alerta_manos = alerta_manos

        frames_procesados += 1
        if total_frames > 0 and frames_procesados % max(1, (total_frames // 10)) == 0:
            print(f"      -> Progreso MediaPipe: {(frames_procesados / total_frames) * 100:.0f}%")

    cap.release()
    holistic.close()

    try:
        reporte_emociones = analizar_emociones_oratoria(ruta_video, fps_deseado=1)
        if 'dataframe_crudo' in reporte_emociones:
            del reporte_emociones['dataframe_crudo']
    except Exception as e:
        print(f"      -> Fallo en DeepFace: {e}")
        reporte_emociones = {"error": str(e)}

    return {
        "metricas": {
            "postura_pTM": cinetico.finalizar_sesion(duracion_total),
            "comportamiento": behavioral.generar_reporte_post_hoc(),
            "mirada_y_cabeza": mirada.generar_reporte_post_hoc(),
            "nerviosismo": adaptors.generar_reporte_post_hoc(),
            "emociones_faciales": reporte_emociones
        },
        "incidencias_temporales": incidencias_vision
    }

# ==========================================
# EXPORTADOR ML (Dataset Global)
# ==========================================
def actualizar_dataset_estadistico(id_video, reporte_maestro):
    csv_global = "datos_exportados/dataset_multimodal_maestro.csv"
    os.makedirs("datos_exportados", exist_ok=True)

    res = reporte_maestro.get("crudos", {}).get("resultados", {})
    ritmo = res.get("ritmo_y_fluidez", {})
    prosodia = res.get("prosodia_global", {})
    lex = res.get("lexico_y_pragmatica", {})
    vis = res.get("vision_global", {})
    
    comp = vis.get("comportamiento", {})
    mirada = vis.get("mirada_y_cabeza", {})
    nervios = vis.get("nerviosismo", {})
    postura = vis.get("postura_pTM", {})

    fila = {
        "ID_Orador": id_video,
        "Duracion_Segundos": reporte_maestro.get("crudos", {}).get("metadata", {}).get("duracion_segundos", 0),
        "WPM_Tasa_Habla": ritmo.get("tasa_global_wpm", 0),
        "SPS_Articulacion": ritmo.get("articulation_rate_sps", 0),
        "Tono_Media_Hz": prosodia.get("tono_promedio_hz", 0),
        "Tono_Variabilidad_ST": prosodia.get("tono_std_st", 0),
        "Jitter_Pct": prosodia.get("jitter", 0),
        "Shimmer_Pct": prosodia.get("shimmer", 0),
        "Calidad_Vocal_HNR": prosodia.get("hnr_db", 0),
        "Total_Muletillas_Lexicas": lex.get("perfil_pragmatico", {}).get("total_muletillas_lexicas", 0),
        "Densidad_Muletillas_Pct": sum(lex.get("perfil_pragmatico", {}).get("densidades", {}).values()) if lex.get("perfil_pragmatico") else 0,
        "Muletillas_Acusticas_CNN": res.get("disfluencias_acusticas", {}).get("total_muletillas_acusticas", 0),
        "Energia_Cinetica_Media": comp.get("motion_energy_mean", 0),
        "Cobertura_Escenario": comp.get("total_stage_coverage", 0),
        "Contacto_Visual_Pct": mirada.get("OpenOPAF", {}).get("porcentaje_mirada_audiencia", 0),
        "Cabeza_Estabilidad_Up_Down": mirada.get("Chen_Metrics", {}).get("Face_Up_Down_Sd", 0),
        "Total_Frotamiento_Manos": nervios.get("Total_Frotamientos_Manos", 0),
        "Postura_Cerrada_pTM": postura.get("pTM_postura", 0)
    }

    df = pd.DataFrame([fila])
    df.to_csv(csv_global, mode='a', index=False, header=not os.path.isfile(csv_global))

# ==========================================
# LÓGICA DE PROCESAMIENTO
# ==========================================
# NOTA: Recibe los modelos de WhisperX por parámetro para no re-cargarlos.
def procesar_video_multimodal(ruta_video: str, model_w, model_a, metadata_a, device) -> bool:
    id_sesion = os.path.splitext(os.path.basename(ruta_video))[0]
    print(f"\n{'='*50}\n🎥 INICIANDO ANÁLISIS: {id_sesion}\n{'='*50}")
    inicio_total = time.time()
    
    try:
        print("1. Cargando señal (Dual-Pipeline)...")
        y_crudo, y_limpio, sr = procesar_audio_oratoria(ruta_video, sr_objetivo=16000)
        duracion_audio = librosa.get_duration(y=y_crudo, sr=sr)
        
        if duracion_audio < 110.0:
            print(f"⚠️ [OMITIDO] Video corto ({duracion_audio:.2f}s). Se requieren 120s.")
            return False
        
        print("2. Transcribiendo y alineando (WhisperX)...")
        # Ya no se carga el modelo aquí. Se usa el que vino por parámetro.
        result = model_w.transcribe(whisperx.load_audio(ruta_video), batch_size=16)
        result_aligned = whisperx.align(result["segments"], model_a, metadata_a, whisperx.load_audio(ruta_video), device)

        texto_crudo = ""
        marcas_whisper_global = []
        for seg in result_aligned['segments']:
            texto_crudo += seg['text'] + " "
            if 'words' in seg: marcas_whisper_global.extend(seg['words'])

        print("3. Calculando métricas acústicas y léxicas...")
        datos_ritmo = calcular_ritmo_whisper({'text': texto_crudo, 'segments': result_aligned['segments']}, duracion_total_audio=duracion_audio)
        datos_prosodia = analizar_prosodia_global(y_crudo, sr)
        
        # --- MODIFICACIÓN CLAVE: RECIBE LA TUPLA CON EVENTOS ESTRUCTURALES ---
        telemetria_temporal, eventos_estructurales = extraer_telemetria_local(y_crudo, sr, result_aligned.get("segments", []), datos_prosodia.get("ref_volumen_max", 1.0))
        
        datos_lexicos = analizar_lexico_y_taxonomia(texto_crudo, marcas_whisper=marcas_whisper_global)
        datos_cnn = detectar_muletillas_acusticas(y_limpio, sr, marcas_whisper=marcas_whisper_global)

        # --- VISIÓN ---
        res_vision_global = {}
        incidencias_vision = []
        if VISION_ACTIVA:
            print("4. 👁️ Procesando Visión Corporal (MediaPipe + DeepFace)...")
            datos_vision = extraer_datos_vision(ruta_video)
            res_vision_global = datos_vision["metricas"]
            incidencias_vision = datos_vision["incidencias_temporales"]
            
            # ------------------------------------------------------------------
            # FUSIÓN MULTIMODAL REAL (Sincronía Temporal)
            # Inyectamos los gestos del cuerpo adentro de los segmentos de habla
            # ------------------------------------------------------------------
            print("   -> Cruzando métricas de visión con segmentos de voz...")
            for segmento in telemetria_temporal:
                inicio_seg = segmento.get('inicio', 0.0)
                fin_seg = segmento.get('fin', 0.0)
                
                # Buscamos qué hizo el orador con el cuerpo en este lapso de segundos
                gestos_sincronizados = [
                    inc for inc in incidencias_vision 
                    if inicio_seg <= inc['tiempo'] <= fin_seg
                ]
                segmento['incidencias_corporales'] = gestos_sincronizados
        else:
            print("4. ⏭️ Saltando visión (Módulos no detectados).")

        telemetria_pura = {
            "id_sesion": id_sesion,
            "metadata": {"duracion_segundos": duracion_audio, "sr": sr, "transcripcion": texto_crudo.strip()},
            "resultados": {
                "ritmo_y_fluidez": datos_ritmo,
                "lexico_y_pragmatica": datos_lexicos,
                "disfluencias_acusticas": datos_cnn,
                "prosodia_global": datos_prosodia,
                "vision_global": res_vision_global,
                "incidencias_vision": incidencias_vision, # Lista global cruda
                "telemetria_temporal": telemetria_temporal, # Lista FUSIONADA MULTIMODAL
                "eventos_estructurales": eventos_estructurales # --- AHORA SE EXPORTAN LOS EVENTOS ---
            }
        }

        print("5. Generando reporte evaluado y exportando datasets...")
        reporte_maestro = generar_veredicto(telemetria_pura)
        exportador.exportar_todo(reporte_maestro)
        actualizar_dataset_estadistico(id_sesion, reporte_maestro)

        print(f"✅ Procesado exitosamente en {time.time() - inicio_total:.2f}s.")
        return True

    except Exception as e:
        print(f"❌ ERROR CRÍTICO en {id_sesion}: {e}")
        traceback.print_exc()
        return False

# ==========================================
# BUCLE BATCH (MOTOR PRINCIPAL)
# ==========================================
def main():
    carpeta = "videos_entrada"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        return

    videos = glob.glob(os.path.join(carpeta, "*.mp4"))
    if not videos:
        print("⚠️ No hay videos en 'videos_entrada/'.")
        return

    print(f"🚀 Iniciando orquestador para {len(videos)} videos en BATCH...")
    
    # ---------------------------------------------------------------------
    # OPTIMIZACIÓN DE MEMORIA (Se carga WhisperX 1 sola vez para todo el lote)
    # ---------------------------------------------------------------------
    print("⏳ Cargando modelos de Inteligencia Artificial en RAM/VRAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model_w = whisperx.load_model("small", device, compute_type="int8")
        # Asumimos idioma español ('es') para tesis UNAH
        model_a, metadata_a = whisperx.load_align_model(language_code="es", device=device)
        print("✔️ Modelos cargados exitosamente.")
    except Exception as e:
        print(f"❌ Fallo al cargar WhisperX: {e}")
        return

    for v in videos:
        procesar_video_multimodal(v, model_w, model_a, metadata_a, device)

if __name__ == "__main__":
    main()