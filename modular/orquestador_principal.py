import os
import time
import librosa
import numpy as np
from typing import Dict, Any

# ==========================================
# IMPORTACIÓN DE MÓDULOS DE EXTRACCIÓN (SENSORES)
# ==========================================
import preprocesamiento_acustico as preprocesador  # El módulo dual-pipeline
import modulo_ritmo_transcripcion as ritmo_asr
import modulo_muletillas_lexicas as muletillas_nlp
import modulo_disfluencias_cnn as disfluencias_cnn
import modulo_telemetria_local as telemetria_time_series
import modulo_prosodia_volumen as prosodia_global

# ==========================================
# IMPORTACIÓN DE MÓDULOS DE EVALUACIÓN Y EXPORTACIÓN
# ==========================================
import motor_rubrica_evaluacion as evaluador
import exportador_datos_crudos as exportador

def ejecutar_pipeline_audio(ruta_audio: str, id_sesion: str = "test_01") -> Dict[str, Any]:
    """
    Orquesta el análisis acústico y lingüístico completo.
    Implementa una arquitectura Dual-Pipeline para proteger la integridad de los datos clínicos.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] INICIANDO PIPELINE MULTIMODAL (FASE AUDIO) - Sesión: {id_sesion}")
    print("="*60)
    
    tiempo_inicio = time.time()
    telemetria_pura = {"id_sesion": id_sesion, "metadata": {}, "resultados": {}}

    # ---------------------------------------------------------
    # FASE 1: PREPROCESAMIENTO ACÚSTICO BIFURCADO (DUAL-PIPELINE)
    # ---------------------------------------------------------
    print("⏳ [1/6] Cargando señal y generando bifurcación cruda/limpia...")
    try:
        y_crudo, y_limpio, sr = preprocesador.procesar_audio_oratoria(ruta_audio, sr_objetivo=16000)
        
        if y_crudo is None or y_limpio is None:
            return {"error": "Fallo en la extracción y preprocesamiento de audio."}
            
        duracion_audio = librosa.get_duration(y=y_crudo, sr=sr)
        telemetria_pura["metadata"]["duracion_segundos"] = duracion_audio
        telemetria_pura["metadata"]["sr"] = sr
    except Exception as e:
        print(f"❌ Error crítico en preprocesamiento: {e}")
        return {"error": "Fallo en limpieza de audio"}

    # ---------------------------------------------------------
    # FASE 2: MODELO DE LENGUAJE (WHISPERX) Y ALINEACIÓN FORZADA
    # ---------------------------------------------------------
    print("🤖 [2/6] Ejecutando transcripción y alineación milimétrica (WhisperX)...")
    try:
        import whisperx
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Cargar modelo y transcribir
        model = whisperx.load_model("small", device, compute_type="int8")
        audio_whisperx = whisperx.load_audio(ruta_audio)
        result = model.transcribe(audio_whisperx, batch_size=16)

        # 2. ALINEACIÓN FORZADA
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_whisperx, device, return_char_alignments=False)

        # 3. RECONSTRUCCIÓN DE DATOS
        texto_crudo = ""
        marcas_whisper_global = []
        
        for seg in result_aligned['segments']:
            texto_crudo += seg['text'] + " "
            if 'words' in seg:
                marcas_whisper_global.extend(seg['words'])

        print(f"\n🔍 [DEBUG] Cantidad de palabras alineadas por WhisperX: {len(marcas_whisper_global)}")
        if len(marcas_whisper_global) == 0:
            print("⚠️ [DEBUG] ¡ALERTA! La lista marcas_whisper_global está VACÍA.")

        telemetria_pura["metadata"]["transcripcion"] = texto_crudo.strip()

        # Adaptación RIGUROSA de Ritmo (Contiene filtro cognitivo de 400ms)
        datos_ritmo = ritmo_asr.calcular_ritmo_whisper(
            {'text': texto_crudo, 'segments': result_aligned['segments']}, 
            duracion_total_audio=duracion_audio
        )
        
        telemetria_pura["resultados"]["ritmo_y_fluidez"] = datos_ritmo
        
    except Exception as e:
        print(f"❌ Error en procesamiento de lenguaje o alineación (WhisperX): {e}")
        return {"error": "Fallo en WhisperX o Análisis de Pausas"}
        
    # ---------------------------------------------------------
    # FASE 3: ANÁLISIS LÉXICO-PRAGMÁTICO (SPACY)
    # ---------------------------------------------------------
    print("📝 [3/6] Analizando complejidad léxica y palabras de relleno (spaCy)...")
    try:
        datos_lexicos = muletillas_nlp.analizar_lexico_y_taxonomia(texto_crudo, marcas_whisper=marcas_whisper_global)
        telemetria_pura["resultados"]["lexico_y_pragmatica"] = datos_lexicos
    except Exception as e:
        print(f"❌ Error en análisis NLP: {e}")

    # ---------------------------------------------------------
    # FASE 4: INFERENCIA ACÚSTICA PROFUNDA (CNN)
    # ---------------------------------------------------------
    print("🧠 [4/6] Escaneando disfluencias sonoras con Red Neuronal (TensorFlow)...")
    try:
        datos_cnn = disfluencias_cnn.detectar_muletillas_acusticas(y_limpio, sr, marcas_whisper=marcas_whisper_global)
        telemetria_pura["resultados"]["disfluencias_acusticas"] = datos_cnn
    except Exception as e:
        print(f"❌ Error en inferencia CNN: {e}")

    # ---------------------------------------------------------
    # FASE 5: PROSODIA Y ESTRÉS VOCAL (LIBROSA / PARSELMOUTH)
    # ---------------------------------------------------------
    print("🎵 [5/6] Extrayendo descriptores biométricos y prosodia global...")
    try:
        # Pasa y_crudo garantizando la integridad de fase para Jitter/Shimmer
        datos_globales = prosodia_global.analizar_prosodia_global(y_crudo, sr)
        telemetria_pura["resultados"]["prosodia_global"] = datos_globales
        ref_volumen = datos_globales.get("ref_volumen_max", np.max(np.abs(y_crudo)))
    except Exception as e:
        print(f"❌ Error en prosodia global: {e}")
        ref_volumen = np.max(np.abs(y_crudo))

    # ---------------------------------------------------------
    # FASE 6: TELEMETRÍA LOCAL MULTIMODAL (TIME-SERIES)
    # ---------------------------------------------------------
    print("⏱️ [6/6] Generando línea de tiempo de estabilidad vocal...")
    try:
        datos_locales = telemetria_time_series.extraer_telemetria_local(
            y_crudo, sr, result_aligned.get("segments", []), ref_volumen
        )
        telemetria_pura["resultados"]["telemetria_temporal"] = datos_locales
    except Exception as e:
        print(f"❌ Error en telemetría local: {e}")

    tiempo_total = round(time.time() - tiempo_inicio, 2)
    print(f"\n✅ PIPELINE COMPLETADO EN {tiempo_total} SEGUNDOS.")
    print("="*60)
    
    # =========================================================
    # PUNTO DE INFLEXIÓN: EVALUACIÓN Y EXPORTACIÓN
    # =========================================================
    print("⚙️ [Evaluación] Aplicando rúbrica clínico-pedagógica...")
    # Ahora el evaluador leerá correctamente el Shimmer y el HNR
    reporte_evaluado = evaluador.generar_veredicto(telemetria_pura)
    
    # Exportación automática a JSON y CSV
    exportador.exportar_todo(reporte_evaluado)
    
    return reporte_evaluado

if __name__ == "__main__":
    # Prueba del orquestador unificado
    archivo_prueba = "chico.wav" # <-- Puedes poner aquí el audio de pánico cuando lo grabes
    
    if os.path.exists(archivo_prueba):
        reporte_final = ejecutar_pipeline_audio(archivo_prueba, "PRUEBA_02_INTEGRADA")
    else:
        print(f"⚠️ Archivo '{archivo_prueba}' no encontrado en el directorio.")