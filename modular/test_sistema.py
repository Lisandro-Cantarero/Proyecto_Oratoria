from orquestador_principal import ejecutar_pipeline_audio

# Reemplaza con el nombre de tu archivo de prueba
ruta_test = "lol.wav" 

print("🚀 Iniciando Test de Integración Acústica...")
resultado = ejecutar_pipeline_audio(ruta_test, id_sesion="PRUEBA_01_LISANDRO")

print("\n--- RESUMEN DE EJECUCIÓN ---")
if "error" in resultado:
    print(f"❌ Fallo: {resultado['error']}")
else:
    print("✅ Pipeline ejecutado sin errores técnicos.")