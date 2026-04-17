import cv2
import threading
import numpy as np
import mediapipe as mp
import time
import os
import queue
import sounddevice as sd
import soundfile as sf

# --- Importaciones de Visión (Captura Kinésica en Tiempo Real) ---
from video.body_metrics import GestureAnalyzer, PostureSwayAnalyzer, BodyOrientationAnalyzer, PostureAmplitudeAnalyzer
from video.face_metrics import HeadPoseAnalyzer
from video.self_adaptors import SelfAdaptorAnalyzer 
from video.logger import SessionLogger 

# --- HILO DE GRABACIÓN DE AUDIO DE ALTA FIDELIDAD (Para WhisperX) ---
q_audio = queue.Queue()

def callback_audio(indata, frames, time_info, status):
    if status: print(status, flush=True)
    q_audio.put(indata.copy())

def hilo_de_grabacion_audio(stop_event, filename):
    """Graba el micrófono en alta calidad (48kHz, 16-bit PCM) para la auditoría científica."""
    samplerate = 48000 
    channels = 1
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Manejo de archivo temporal en caso de sobreescritura accidental
    if os.path.exists(filename):
        os.remove(filename)
        
    # Forzamos formato PCM_16 para máxima compatibilidad con modelos acústicos
    with sf.SoundFile(filename, mode='x', samplerate=samplerate, channels=channels, subtype='PCM_16') as file:
        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback_audio, blocksize=4096):
            while not stop_event.is_set():
                file.write(q_audio.get())

# --- INTERFAZ VISUAL ---
def crear_panel_telemetria(alto_video, metricas_vision):
    """Panel ajustado a 550px exclusivo para telemetría kinésica."""
    ancho_panel = 550
    panel = np.full((alto_video, ancho_panel, 3), (30, 30, 30), dtype=np.uint8)

    fuente_titulo = cv2.FONT_HERSHEY_COMPLEX
    fuente_texto = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "SISTEMA DE AUDITORIA MULTIMODAL", (15, 30), fuente_titulo, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (15, 45), (ancho_panel - 15, 45), (100, 100, 100), 1, cv2.LINE_AA)

    cv2.putText(panel, "[ METRICAS VISUALES - MediaPipe ]", (15, 75), fuente_titulo, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    y_pos = 105
    for nombre, (texto_descriptivo, color) in metricas_vision.items():
        cv2.circle(panel, (25, y_pos - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(panel, f"{nombre}:", (40, y_pos), fuente_texto, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, texto_descriptivo, (140, y_pos), fuente_texto, 0.45, color, 1, cv2.LINE_AA)
        y_pos += 35 

    
    y_estado = alto_video - 50
    cv2.rectangle(panel, (15, y_estado), (ancho_panel - 15, alto_video - 15), (15, 15, 15), -1)
    cv2.rectangle(panel, (15, y_estado), (ancho_panel - 15, alto_video - 15), (100, 100, 100), 1)
    cv2.putText(panel, "ESTADO: GRABANDO AUDIO HQ Y VIDEO", (30, y_estado + 22), fuente_texto, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    return panel

def main():
    print("="*50)
    print("🚀 SISTEMA DE CAPTURA MULTIMODAL (Fase 1)")
    print("="*50)

    nombre_orador = input("\n👤 Por favor, ingresa tu nombre o identificador: ").strip()
    if not nombre_orador:
        nombre_orador = "Orador" 

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    ancho_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if ancho_real <= 0 or alto_real <= 0:
        ancho_real, alto_real = 640, 480

    display_w = ancho_real
    display_h = alto_real
    
    fps_grabacion = 15.0 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    nombre_ventana = "Tesis - Captura Multimodal"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(nombre_ventana, cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty(nombre_ventana, cv2.WND_PROP_TOPMOST, 0)

    print(f"\n⏳ Hola {nombre_orador}. Encuadra tu cámara y presiona 's' para iniciar.")
    
    # ==========================================================
    # 1. BUCLE DE ESPERA (STANDBY)
    # ==========================================================
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        frame_reducido = cv2.resize(frame, (display_w, display_h))

        cv2.rectangle(frame_reducido, (0, display_h - 70), (display_w, display_h), (30, 30, 30), -1)
        cv2.line(frame_reducido, (0, display_h - 70), (display_w, display_h - 70), (100, 100, 100), 1)
        
        cv2.putText(frame_reducido, "PRESIONA 'S' PARA INICIAR LA SESION", (20, display_h - 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_reducido, "El audio y video se sincronizaran automaticamente.", (20, display_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(nombre_ventana, frame_reducido)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('s') or tecla == ord('S'):
            print("\n▶️ Iniciando captura sincronizada...")
            break
        elif tecla == ord('q') or tecla == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # ==========================================================
    # 2. INICIALIZACIÓN INMEDIATA Y ARRANQUE DE MOTORES
    # ==========================================================
    print("⏳ Preparando entorno de captura...")
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        mp_drawing = mp.solutions.drawing_utils
    except Exception as e:
        print(f"\n❌ CRASH FATAL: Falló la carga de módulos visuales: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return

    gesture_analyzer = GestureAnalyzer()
    sway_analyzer = PostureSwayAnalyzer()
    orientation_analyzer = BodyOrientationAnalyzer()
    amplitude_analyzer = PostureAmplitudeAnalyzer()
    head_pose_analyzer = HeadPoseAnalyzer()
    adaptor_analyzer = SelfAdaptorAnalyzer()  

    # Preparar archivos y crear subcarpeta de sesión
    logger = SessionLogger(nombre_orador)
    carpeta_sesion = f"reportes/{logger.session_id}"
    os.makedirs(carpeta_sesion, exist_ok=True)
    
    audio_crudo_filename = f"{carpeta_sesion}/audio_crudo_{logger.session_id}.wav"
    video_crudo_filename = f"{carpeta_sesion}/video_crudo_{logger.session_id}.mp4"

    # Encendido Sincronizado
    stop_event = threading.Event()
    audio_thread = threading.Thread(target=hilo_de_grabacion_audio, args=(stop_event, audio_crudo_filename))
    audio_thread.start()

    video_out = cv2.VideoWriter(video_crudo_filename, fourcc, fps_grabacion, (display_w, display_h))
    
    
    tiempo_inicio_grabacion = time.time()
    frames_escritos = 0 
    frame_contador = 0
    
    last_pose_lms = None
    last_world_lms = None
    last_face_landmarks = None
    last_left_hand = None
    last_right_hand = None

    print("\n✅ Sistema en Vivo (Presiona 'Q' en la ventana de video para finalizar la sesión)")

    # ==========================================================
    # 3. BUCLE PRINCIPAL DE INTERFAZ Y GRABACIÓN
    # ==========================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue
        
        frame_contador += 1 
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        if frame_contador % 3 == 0:
            frame_ia = cv2.resize(rgb_frame, (640, 360))
            pose_results = pose.process(frame_ia)
            face_results = face_mesh.process(frame_ia)
            hands_results = hands.process(frame_ia) 

            last_pose_lms = pose_results.pose_landmarks if pose_results.pose_landmarks else None
            last_world_lms = pose_results.pose_world_landmarks if pose_results.pose_world_landmarks else None
            last_face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
            
            last_left_hand = None
            last_right_hand = None
            if hands_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                    if handedness.classification[0].label == 'Left':
                        last_left_hand = hand_landmarks
                    else:
                        last_right_hand = hand_landmarks

        pose_lms = last_pose_lms
        world_lms = last_world_lms
        face_landmarks = last_face_landmarks
        left_hand_lms = last_left_hand
        right_hand_lms = last_right_hand

        # --- ANÁLISIS KINÉSICO EN TIEMPO REAL ---
        g_metrics = gesture_analyzer.process_landmarks(pose_lms, world_lms, mp_pose)
        
        is_swaying = False
        adaptor_metrics = {"pocket": {"active": False, "type": None}, "face": {"instant_active": False, "report_active": False, "type": None, "instant_type": None}} 
        body_orientation = {"is_facing": True, "error": None}
        body_amplitude = {"is_closed": False, "error": None}

        if pose_lms:
            is_swaying = sway_analyzer.check_sway(pose_lms, mp_pose)
            adaptor_metrics = adaptor_analyzer.check_adaptors(pose_lms, left_hand_lms, right_hand_lms, mp_pose)
            body_orientation = orientation_analyzer.check_orientation(pose_lms, mp_pose)
            body_amplitude = amplitude_analyzer.check_amplitude(pose_lms, mp_pose)

        head_metrics = head_pose_analyzer.process_head_pose(face_landmarks, pose_lms, mp_pose, (360, 640, 3))

        if logger is not None and video_out is not None:
            # Registrar incidencias en RAM
            logger.update_metric("Gesticulacion", g_metrics["inactivity_alert"], "Falta gesto válido (>6s)")
            logger.update_metric("Amplitud Corporal", body_amplitude["is_closed"], body_amplitude.get("error", ""))
            logger.update_metric("Estabilidad", is_swaying, "Balanceo lateral excesivo (Péndulo)")
            logger.update_metric("Alineacion", not body_orientation["is_facing"], body_orientation.get("error", ""))
            logger.update_metric("Mirada", not head_metrics["is_looking"], head_metrics.get("error_type", ""))
            logger.update_metric("Bolsillos/Ocultas", adaptor_metrics["pocket"]["active"], adaptor_metrics["pocket"]["type"])
            logger.update_metric("Toques Rostro", adaptor_metrics["face"]["report_active"], adaptor_metrics["face"]["type"])

            # Sincronización precisa de escritura de video
            tiempo_transcurrido_real = time.time() - tiempo_inicio_grabacion
            frames_esperados = int(tiempo_transcurrido_real * fps_grabacion)
            frame_a_guardar = cv2.resize(frame, (display_w, display_h))
            
            while frames_escritos < frames_esperados:
                video_out.write(frame_a_guardar)
                frames_escritos += 1

        # --- DIBUJO DE UI ---
        color_gesto = (0, 0, 255) if g_metrics["inactivity_alert"] else (0, 255, 0)
        txt_inactividad = "ALERTA: FALTA GESTO VALIDO (>8s)" if g_metrics["inactivity_alert"] else "Gesticulacion Visible"

        if body_amplitude["is_closed"]: 
            txt_amplitud, color_amplitud = f"ALERTA: {body_amplitude.get('error', 'CONTRAIDA')}", (0, 0, 255)
        else: 
            txt_amplitud, color_amplitud = "Amplitud: OK (Postura Abierta)", (0, 255, 0)

        txt_sway, color_sway = ("ALERTA: BALANCEO (BAILE)!", (0, 0, 255)) if is_swaying else ("Postura Estable", (0, 255, 0))

        if body_orientation["is_facing"]: 
            txt_cuerpo, color_cuerpo = "Alineacion: FRENTE AL PUBLICO", (0, 255, 0)
        else: 
            txt_cuerpo, color_cuerpo = f"ALERTA: {body_orientation.get('error', 'PERFIL')}", (0, 0, 255)

        if head_metrics["is_looking"]: 
            txt_mirada, color_mirada = "Contacto Visual: OK (Paneo Activo)", (0, 255, 0)
        else: 
            txt_mirada, color_mirada = f"ALERTA: {head_metrics.get('error_type', 'MIRADA PERDIDA')}", (0, 0, 255)

        if adaptor_metrics["face"]["instant_active"]: 
            txt_toques_rostro, color_toques_rostro = f"Toques: ALERTA ({adaptor_metrics['face']['instant_type']})", (0, 165, 255) 
        else: 
            txt_toques_rostro, color_toques_rostro = "Toques Rostro: Ninguno", (0, 255, 0)

        if adaptor_metrics["pocket"]["active"]: 
            txt_manos, color_manos = f"Manos: ALERTA ({adaptor_metrics['pocket']['type']})", (0, 0, 255) 
        else: 
            txt_manos, color_manos = "Manos: Visibles", (0, 255, 0)

        frame_display = cv2.resize(frame, (display_w, display_h))
        
        # 🌟 EL RENDERIZADO DE LAS MALLAS 🌟
        if pose_lms: 
            mp_drawing.draw_landmarks(frame_display, pose_lms, mp_pose.POSE_CONNECTIONS)
        if left_hand_lms: 
            mp_drawing.draw_landmarks(frame_display, left_hand_lms, mp_hands.HAND_CONNECTIONS)
        if right_hand_lms: 
            mp_drawing.draw_landmarks(frame_display, right_hand_lms, mp_hands.HAND_CONNECTIONS)

        estado_ui = {
            "Gesticula": (txt_inactividad, color_gesto), "Amplitud": (txt_amplitud, color_amplitud),
            "Estabil.": (txt_sway, color_sway), "Alineac.": (txt_cuerpo, color_cuerpo),
            "Mirada": (txt_mirada, color_mirada), "Rostro": (txt_toques_rostro, color_toques_rostro),
            "Manos": (txt_manos, color_manos)
        }

        panel_dashboard = crear_panel_telemetria(display_h, estado_ui)
        vista_completa = cv2.hconcat([frame_display, panel_dashboard])

        # Banner Superior
        cv2.rectangle(vista_completa, (0, 0), (display_w + 550, 40), (0, 255, 0), -1)
        cv2.putText(vista_completa, "REC - FASE 1: CAPTURA EN TIEMPO REAL", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(vista_completa, (420, 20), 8, (0, 0, 255), -1)

        cv2.imshow(nombre_ventana, vista_completa)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹️ Señal de detención recibida ('q'). Cerrando sesión...")
            break

    # ==========================================================
    # --- CIERRE Y TRANSICIÓN A LA FASE CIENTÍFICA ---
    # ==========================================================
    if logger is not None:
        logger.update_metric("Gesticulacion", False, "")
        logger.update_metric("Amplitud Corporal", False, "")
        logger.update_metric("Estabilidad", False, "")
        logger.update_metric("Alineacion", False, "")
        logger.update_metric("Mirada", False, "")
        logger.update_metric("Bolsillos/Ocultas", False, "")
        logger.update_metric("Toques Rostro", False, "")

    stop_event.set()
    cap.release()
    if video_out is not None: video_out.release()
    cv2.destroyAllWindows()

    print("⏳ Guardando video kinésico y pista de audio HQ...")
    audio_thread.join()

    print("\n" + "="*50)
    print("🧠 FASE 2: ARCHIVOS LISTOS PARA EXTRACCIÓN CIENTÍFICA")
    print("="*50)
    
    # Exportar el JSON del Logger de tiempo real hacia la nueva subcarpeta
    ruta_json_vivo = f"{carpeta_sesion}/{logger.session_id}_tiempo_real.json"
    logger.exportar_json_tiempo_real(ruta_json_vivo)
    print(f"✅ Telemetría en vivo lista. Los archivos aguardan en: {carpeta_sesion}/")
    
    # === LA LLAMADA AUTOMÁTICA AL ORQUESTADOR CIENTÍFICO ===
    print("🚀 Invocando Orquestador Maestro (WhisperX + DeepFace)...")
    from orquestador_maestro import ejecutar_pipeline_hibrido
    ejecutar_pipeline_hibrido(video_crudo_filename, audio_crudo_filename, ruta_json_vivo, logger.session_id)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nEjecución interrumpida manualmente.")