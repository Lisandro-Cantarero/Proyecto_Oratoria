import os
import time
import json
import traceback
import librosa
import numpy as np
import pandas as pd
import torch
import whisperx


from modular.preprocesamiento_acustico import procesar_audio_oratoria
from modular.modulo_ritmo_transcripcion import calcular_ritmo_whisper
from modular.modulo_telemetria_local import extraer_telemetria_local
from modular.modulo_prosodia_volumen import analizar_prosodia_global
from modular.modulo_muletillas_lexicas import analizar_lexico_y_taxonomia
from modular.modulo_disfluencias_cnn import detectar_muletillas_acusticas
from modular.motor_rubrica_evaluacion import generar_veredicto

# Importación de Visión y Generador PDF
from orquestador_principal import extraer_datos_vision, actualizar_dataset_estadistico
from generador_reportes_pdf import GeneradorReporteElegante

def ejecutar_pipeline_hibrido(ruta_video, ruta_audio, ruta_json_vivo, id_sesion):
    print(f"\n{'='*50}\n🧠 INICIANDO FASE 2: AUDITORÍA CIENTÍFICA ({id_sesion})\n{'='*50}")
    inicio_total = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("⏳ Cargando modelos pesados (WhisperX) en RAM/VRAM...")
    try:
        model_w = whisperx.load_model("small", device, compute_type="int8")
        model_a, metadata_a = whisperx.load_align_model(language_code="es", device=device)
    except Exception as e:
        print(f"❌ Fallo al cargar WhisperX: {e}")
        return

    try:
        print("1. Cargando señal acústica HQ...")
        # AHORA CARGAMOS DESDE EL ARCHIVO .WAV INDEPENDIENTE
        y_crudo, y_limpio, sr = procesar_audio_oratoria(ruta_audio, sr_objetivo=16000)
        duracion_audio = librosa.get_duration(y=y_crudo, sr=sr)
        
        # ELIMINADO EL BLOQUEO DE 110 SEGUNDOS PARA PERMITIR PRUEBAS CORTAS
        if duracion_audio < 5.0:
            print(f"⚠️ Video demasiado corto ({duracion_audio:.2f}s). Mínimo 5s.")
            return False
        
        print("2. Transcribiendo y alineando (WhisperX)...")
        result = model_w.transcribe(whisperx.load_audio(ruta_audio), batch_size=16)
        result_aligned = whisperx.align(result["segments"], model_a, metadata_a, whisperx.load_audio(ruta_audio), device)

        texto_crudo = ""
        marcas_whisper_global = []
        for seg in result_aligned['segments']:
            texto_crudo += seg['text'] + " "
            if 'words' in seg: marcas_whisper_global.extend(seg['words'])

        print("3. Calculando métricas acústicas y léxicas...")
        datos_ritmo = calcular_ritmo_whisper({'text': texto_crudo, 'segments': result_aligned['segments']}, duracion_total_audio=duracion_audio)
        datos_prosodia = analizar_prosodia_global(y_crudo, sr)
        
        # --- CAMBIO 1: SE RECIBEN LOS EVENTOS ESTRUCTURALES ---
        telemetria_temporal, eventos_estructurales = extraer_telemetria_local(y_crudo, sr, result_aligned.get("segments", []), datos_prosodia.get("ref_volumen_max", 1.0))
        
        datos_lexicos = analizar_lexico_y_taxonomia(texto_crudo, marcas_whisper=marcas_whisper_global)
        datos_cnn = detectar_muletillas_acusticas(y_limpio, sr, marcas_whisper=marcas_whisper_global)

        print("4. 👁️ Procesando Visión Corporal (MediaPipe + DeepFace) en modo Batch...")
        datos_vision = extraer_datos_vision(ruta_video)
        res_vision_global = datos_vision["metricas"]
        incidencias_vision = datos_vision["incidencias_temporales"]
        
        # EL DICCIONARIO APLANADO (CORREGIDO)
        telemetria_pura = {
            "id_sesion": id_sesion,
            "metadata": {
                "duracion_segundos": duracion_audio, 
                "sr": sr, 
                "transcripcion": texto_crudo.strip()
            },
            "resultados": {
                "ritmo_y_fluidez": datos_ritmo,
                "lexico_y_pragmatica": datos_lexicos,
                "disfluencias_acusticas": datos_cnn,
                "prosodia_global": datos_prosodia,
                "vision_global": res_vision_global,
                "incidencias_vision": incidencias_vision,
                "telemetria_temporal": telemetria_temporal,
                "eventos_estructurales": eventos_estructurales # INYECTADO PARA QUE LO LEA LA RÚBRICA
            }
        }

        print("5. Generando Veredicto Científico...")
        reporte_maestro = generar_veredicto(telemetria_pura)
        
        # ====================================================================
        # 6. FUSIÓN MULTIMODAL: INYECCIÓN DEL JSON DE TIEMPO REAL Y ESTRUCTURALES
        # ====================================================================
        print("6. 🔗 Fusionando Bitácora Kinésica y Eventos Locales...")
        if os.path.exists(ruta_json_vivo):
            with open(ruta_json_vivo, 'r', encoding='utf-8') as f:
                datos_vivo = json.load(f)
                eventos_kinesicos = datos_vivo.get("eventos_tiempo_real", [])
            
            # Asegurarnos de que exista la llave 'evaluacion' y 'eventos_locales'
            if "evaluacion" not in reporte_maestro:
                reporte_maestro["evaluacion"] = {}
            if "eventos_locales" not in reporte_maestro["evaluacion"]:
                reporte_maestro["evaluacion"]["eventos_locales"] = []
                
            # Agregar los eventos visuales a la lista de eventos acústicos
            reporte_maestro["evaluacion"]["eventos_locales"].extend(eventos_kinesicos)
            
            # --- CAMBIO 2: AGREGAR EVENTOS ESTRUCTURALES (Z-Score y Monotonía) ---
            # La rúbrica ya los inyectó en el paso 5, pero esto previene fallos si se llama a este script directamente.
            if "eventos_estructurales" in telemetria_pura["resultados"]:
                 pass # Ya están dentro del JSON maestro generado por la rúbrica
            
            # Ordenar toda la línea de tiempo fusionada por segundo exacto
            reporte_maestro["evaluacion"]["eventos_locales"].sort(key=lambda x: x.get("start_sec", 0))
        else:
            print("⚠️ No se encontró el JSON de tiempo real. Se generará el reporte solo con audio.")

        # --- CAMBIO 3: GUARDADO EN SUBCARPETA DE SESIÓN ---
        # Asegurar que el directorio exista
        directorio_sesion = f"reportes/{id_sesion}"
        os.makedirs(directorio_sesion, exist_ok=True)
        
        # Guardar el JSON Maestro Fusionado
        ruta_json_final = f"{directorio_sesion}/{id_sesion}_maestro_fusionado.json"
        with open(ruta_json_final, "w", encoding='utf-8') as f:
            json.dump(reporte_maestro, f, indent=4, ensure_ascii=False)
            
        # Actualizar Dataset para ML
        actualizar_dataset_estadistico(id_sesion, reporte_maestro)

        # ====================================================================
        # 7. GENERACIÓN DEL PDF FINAL
        # ====================================================================
        print("7. 📄 Generando Reporte PDF Diagnóstico...")
        ruta_pdf = f"{directorio_sesion}/Auditoria_Forense_{id_sesion}.pdf"
        generador = GeneradorReporteElegante(ruta_json_final)
        generador.exportar_pdf(ruta_pdf)

        print(f"\n✅ ¡PIPELINE HÍBRIDO COMPLETADO EN {time.time() - inicio_total:.2f}s!")
        print(f"El reporte final ha sido guardado en: {ruta_pdf}")
        return True

    except Exception as e:
        print(f"❌ ERROR CRÍTICO en la Fase Científica: {e}")
        traceback.print_exc()
        return False

# ================= USO DIRECTO PARA PRUEBAS =================
if __name__ == "__main__":
    # Remplaza 'liko_20260415_233555' con el ID de la prueba que acabas de grabar
    ID_PRUEBA = "liko_20260415_233555" 
    
    video_path = f"reportes/{ID_PRUEBA}/video_crudo_{ID_PRUEBA}.mp4"
    audio_path = f"reportes/{ID_PRUEBA}/audio_crudo_{ID_PRUEBA}.wav"
    json_path = f"reportes/{ID_PRUEBA}/{ID_PRUEBA}_tiempo_real.json"
    
    ejecutar_pipeline_hibrido(video_path, audio_path, json_path, ID_PRUEBA)
