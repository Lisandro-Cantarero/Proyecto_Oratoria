import cv2
import time
import mediapipe as mp
import math

# Importamos tus módulos de grado tesis
from analizador_cinetico import AnalizadorCinetico
from analizador_behavioral import AnalizadorBehavioral
from analizador_cabeza import AnalizadorMirada 
from self_adaptors import AnalizadorSelfAdaptors

def main():
    print("Iniciando entorno de pruebas visual MULTIMODAL (UNAH - Ingeniería en Sistemas)...")
    print("Presiona 'q' para cerrar la ventana y generar el reporte final.")

    # 1. CÁMARA WEB EN VIVO
    cap = cv2.VideoCapture(0)
    
    # --- MEDIAPIPE HOLISTIC ---
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=1,           # Complejidad alta para estabilidad de hombros/cuerpo
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        refine_face_landmarks=True    # <--- CORREGIDO: Nombre correcto para habilitar el iris en Holistic
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Instanciar Analizadores
    analizador_cinetico = AnalizadorCinetico()
    analizador_behavioral = AnalizadorBehavioral()
    analizador_mirada = AnalizadorMirada()
    analizador_adaptors = AnalizadorSelfAdaptors()
    
    # Cronómetro absoluto para la sesión en vivo
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara web.")
            break

        # Modo espejo
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inferencia
        results = holistic.process(rgb_frame)
        
        # Tiempo real
        timestamp_actual = time.time() - start_time

        if results.pose_landmarks:
            # Dibujar cuerpo para referencia
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # --- EJECUCIÓN MULTIMODAL SINCRÓNICA ---
            # 1. Cuerpo
            estado_cinetico = analizador_cinetico.procesar_frame(results.pose_landmarks, mp_pose, timestamp_actual)
            analizador_behavioral.procesar_frame_behavioral(results.pose_landmarks, mp_pose)
            
            # 2. Cabeza (Captura datos en crudo para el Post-Hoc)
            analizador_mirada.procesar_frame_cabeza(results.face_landmarks, frame.shape[:2])
            
            # 3. Nerviosismo (Self-Adaptors)
            alerta_facial, alerta_manos = analizador_adaptors.procesar_frame_adaptors(
                results.face_landmarks,
                results.right_hand_landmarks,
                results.left_hand_landmarks,
                timestamp_actual,
                frame.shape[:2] 
            )

            # --- INTERFAZ VISUAL (UI) ---
            y_pos = 40
            
            # Alertas Cinéticas completas (OK / ALERTA)
            for incidencia, activo in estado_cinetico.items():
                if activo:
                    texto = f"ALERTA: {incidencia.replace('_', ' ').upper()}"
                    color = (0, 0, 255) # Rojo
                else:
                    texto = f"OK: {incidencia.replace('_', ' ')}"
                    color = (0, 255, 0) # Verde
                
                # Borde negro para que se lea siempre
                cv2.putText(frame, texto, (22, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, texto, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 30
                    
            # Alertas Multimodales de Nerviosismo
            y_pos += 10 # Pequeño margen
            if alerta_facial:
                cv2.putText(frame, alerta_facial, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                y_pos += 30
            if alerta_manos:
                cv2.putText(frame, alerta_manos, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            # --- Métricas en vivo (Derecha) ---
            reporte_parcial = analizador_behavioral.generar_reporte_post_hoc()
            reporte_mirada_vivo = analizador_mirada.generar_reporte_post_hoc()
            
            if "error" not in reporte_mirada_vivo:
                ojos_vivo = reporte_mirada_vivo.get('Rastreo_Ocular_Secundario', {}).get('porcentaje_contacto_visual_real', 0.0)
                pitch_vivo = reporte_mirada_vivo['Chen_Metrics']['Face_Up_Down_Sd']
            else:
                ojos_vivo = pitch_vivo = 0.0

            textos_ui = [
                "ESTADISTICAS (Chen & OPAF):",
                f"Motion Energy: {reporte_parcial['motion_energy_mean']:.4f}",
                f"Gesticulacion: {reporte_parcial['porcentaje_tiempo_zona_valida']:.1f}%",
                f"Gaze (OpenOPAF): {ojos_vivo:.1f}%",
                f"Head SD (Chen): {pitch_vivo:.2f}"
            ]
            
            y_pos_der = 40
            alto, ancho, _ = frame.shape
            
            for linea in textos_ui:
                cv2.putText(frame, linea, (ancho - 320 + 2, y_pos_der + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, linea, (ancho - 320, y_pos_der), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos_der += 30

        else:
            cv2.putText(frame, "Cuerpo no detectado", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow('Laboratorio Multimodal - UNAH', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # Cerrar recursos
    cap.release()
    cv2.destroyAllWindows()

    # --- REPORTE ACADÉMICO FINAL EN CONSOLA ---
    print("\n" + "="*60)
    print("REPORTE DE EVALUACIÓN MULTIMODAL DE ORATORIA")
    print("="*60)
    
    tiempo_total = time.time() - start_time
    
    # 1. Errores de Postura
    reporte_cinetico = analizador_cinetico.finalizar_sesion(tiempo_total)
    print("\n--- ERRORES DE POSTURA ---")
    for metrica, valor in reporte_cinetico.items():
        print(f"  > {metrica}: {valor*100:.2f}% del tiempo")

    # 2. Comportamiento (Wörtwein et al., 2015)
    reporte_behavioral = analizador_behavioral.generar_reporte_post_hoc()
    print("\n--- COMPORTAMIENTO CINÉTICO (Wörtwein et al.) ---")
    print(f"  > Energía Media de Movimiento: {reporte_behavioral['motion_energy_mean']:.4f}")
    print(f"  > Desplazamiento en Escenario: {reporte_behavioral.get('total_stage_coverage', 0.0):.2f} anchos de hombro")
    print(f"  > Tiempo en Zona Válida:       {reporte_behavioral['porcentaje_tiempo_zona_valida']:.2f}%")

    # 3. Nerviosismo (Adaptadores)
    reporte_adaptors = analizador_adaptors.generar_reporte_post_hoc()
    print("\n--- ADAPTORES DE NERVIOSISMO ---")
    print(f"  > Toques Faciales:      {reporte_adaptors['Total_Toques_Faciales']} veces ({reporte_adaptors['Tiempo_Total_Facial_Segundos']} seg)")
    print(f"  > Frotamiento de Manos: {reporte_adaptors['Total_Frotamientos_Manos']} veces ({reporte_adaptors['Tiempo_Total_Frotamiento_Segundos']} seg)")

    # 4. Mirada y Cabeza (OpenOPAF & Chen et al.)
    reporte_mirada = analizador_mirada.generar_reporte_post_hoc()
    print("\n--- CONTACTO VISUAL Y CABEZA (OpenOPAF & Chen et al.) ---")
    if "error" in reporte_mirada:
        print(f"  > Error: {reporte_mirada['error']}")
    else:
        # Lógica OpenOPAF: Zonas y cumplimiento
        print(f"  [OpenOPAF] Mirada a la Audiencia (Cabeza): {reporte_mirada['OpenOPAF']['porcentaje_mirada_audiencia']}%")
        print(f"  [OpenOPAF] Habilidad Dominada:             {'SÍ' if reporte_mirada['OpenOPAF']['habilidad_dominada'] else 'NO'}")
        
        # Lógica de Rastreo Ocular (Métrica Secundaria)
        reporte_ojos = reporte_mirada.get('Rastreo_Ocular_Secundario', {})
        print(f"  [Gaze-Lock] Contacto Visual Real (Ojos):   {reporte_ojos.get('porcentaje_contacto_visual_real', 0)}%")
        print(f"  [Gaze-Lock] Estabilidad Ocular (SD):       {reporte_ojos.get('sd_movimiento_ocular', 0)}")
        
        # Lógica Chen et al.: Descriptores Estadísticos para ML
        print(f"  [Chen et al.] SD Pitch (Vertical):         {reporte_mirada['Chen_Metrics']['Face_Up_Down_Sd']:.4f}°")
        print(f"  [Chen et al.] SD Yaw (Horizontal):         {reporte_mirada['Chen_Metrics']['Face_Left_Right_Sd']:.4f}°")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()