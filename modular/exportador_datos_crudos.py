import os
import json
import csv
from datetime import datetime
from typing import Dict, Any

def _preparar_directorio(subcarpeta: str = "datos_exportados") -> str:
    """Crea el directorio de exportación si no existe."""
    base_dir = os.path.join(os.getcwd(), subcarpeta)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def guardar_json_maestro(reporte_completo: Dict[str, Any], id_sesion: str) -> str:
    """
    Guarda el reporte íntegro (datos crudos + evaluación) en formato JSON.
    Garantiza la reproducibilidad exacta del experimento.
    """
    directorio = _preparar_directorio("datos_exportados/json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{id_sesion}_{timestamp}_maestro.json"
    ruta_completa = os.path.join(directorio, nombre_archivo)
    
    with open(ruta_completa, 'w', encoding='utf-8') as f:
        json.dump(reporte_completo, f, indent=4, ensure_ascii=False)
        
    return ruta_completa

def guardar_csv_globales(reporte_completo: Dict[str, Any], id_sesion: str) -> str:
    """
    Extrae las métricas globales del expositor, calcula el pTM clínico
    y añade todo a un CSV acumulativo para análisis estadístico.
    """
    directorio = _preparar_directorio("datos_exportados/estadistica")
    ruta_completa = os.path.join(directorio, "dataset_global_expositores.csv")
    
    crudos = reporte_completo.get("crudos", {}).get("resultados", {})
    ritmo = crudos.get("ritmo_y_fluidez", {})
    prosodia = crudos.get("prosodia_global", {})
    lexico = crudos.get("lexico_y_pragmatica", {})
    
    # Extraer densidad léxica total
    perfil_pragmatico = lexico.get("perfil_pragmatico", {})
    densidad_muletillas = sum(perfil_pragmatico.get("densidades", {}).values()) if perfil_pragmatico else 0.0
    
    # Extraer conteo acústico (CNN)
    cnn = crudos.get("disfluencias_acusticas", {})
    conteo_acusticas = cnn.get("total_muletillas_acusticas", 0)

    # -------------------------------------------------------------
    # NUEVO CÁLCULO ESTADÍSTICO: pTM (Proportion of Time Making a mistake)
    # -------------------------------------------------------------
    telemetria_temporal = crudos.get("telemetria_temporal", [])
    tiempo_total_hablado = 0.0
    tiempo_shimmer_alto = 0.0
    tiempo_hnr_bajo = 0.0
    
    for evento in telemetria_temporal:
        duracion = evento.get("duracion", 0.0)
        tiempo_total_hablado += duracion
        
        # Si la frase superó el umbral patológico de Shimmer (3.8%)
        if evento.get("shimmer_local", 0.0) > 3.8:
            tiempo_shimmer_alto += duracion
            
        # Si la frase cayó por debajo de la claridad vocal (15 dB), ignorando fallos (0.0)
        if 0.0 < evento.get("hnr_local_db", 100.0) < 15.0:
            tiempo_hnr_bajo += duracion

    ptm_shimmer = (tiempo_shimmer_alto / tiempo_total_hablado * 100) if tiempo_total_hablado > 0 else 0.0
    ptm_hnr = (tiempo_hnr_bajo / tiempo_total_hablado * 100) if tiempo_total_hablado > 0 else 0.0
    
    # -------------------------------------------------------------
    # Aplanar los datos en un diccionario de 1 solo nivel
    # -------------------------------------------------------------
    fila_datos = {
        "id_sesion": id_sesion,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasa_global_wpm": ritmo.get("tasa_global_wpm", 0),
        "tasa_articulacion_wpm": ritmo.get("tasa_articulacion_wpm", 0),
        "tono_sd_semitonos": prosodia.get("tono_std_st", 0),
        "vad_habla_porcentaje": prosodia.get("vuv_porcentaje", 0),
        "densidad_muletillas_lexicas": round(densidad_muletillas, 4),
        "conteo_muletillas_acusticas": conteo_acusticas,
        "shimmer_porcentaje_global": round(prosodia.get("shimmer", 0) * 100, 2), 
        "hnr_db_global": round(prosodia.get("hnr_db", 0), 2),
        "pTM_shimmer_pct": round(ptm_shimmer, 2),
        "pTM_hnr_pct": round(ptm_hnr, 2)
    }
    
    archivo_existe = os.path.isfile(ruta_completa)
    
    with open(ruta_completa, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fila_datos.keys())
        # Si el archivo no existe O si queremos forzar las nuevas cabeceras
        if not archivo_existe:
            writer.writeheader()
        writer.writerow(fila_datos)
        
    return ruta_completa

def guardar_csv_eventos_temporales(reporte_completo: Dict[str, Any], id_sesion: str) -> str:
    """
    Convierte la matriz de telemetría local en un formato tabular de series de tiempo.
    Se ha ajustado para incluir el id_sesion, blindando la integridad referencial de los datos.
    """
    directorio = _preparar_directorio("datos_exportados/series_tiempo")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{id_sesion}_{timestamp}_eventos.csv"
    ruta_completa = os.path.join(directorio, nombre_archivo)
    
    telemetria_temporal = reporte_completo.get("crudos", {}).get("resultados", {}).get("telemetria_temporal", [])
    
    if not telemetria_temporal:
        return "Sin datos temporales"
        
    # INYECCIÓN DEL ID: Transformamos la lista original para que cada diccionario comience con el id_sesion
    datos_corregidos = []
    for evento in telemetria_temporal:
        evento_con_id = {"id_sesion": id_sesion}
        evento_con_id.update(evento)
        datos_corregidos.append(evento_con_id)
        
    # Las cabeceras ahora detectarán automáticamente la nueva clave 'id_sesion'
    cabeceras = datos_corregidos[0].keys()
    
    with open(ruta_completa, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cabeceras)
        writer.writeheader()
        writer.writerows(datos_corregidos)
        
    return ruta_completa

def exportar_todo(reporte_evaluado: Dict[str, Any]) -> Dict[str, str]:
    """Función envoltura para llamar desde el orquestador principal."""
    id_sesion = reporte_evaluado.get("id_sesion", "anonimo")
    rutas = {}
    
    print("\n💾 [Exportador] Guardando telemetría y resultados...")
    try:
        rutas["json"] = guardar_json_maestro(reporte_evaluado, id_sesion)
        rutas["csv_global"] = guardar_csv_globales(reporte_evaluado, id_sesion)
        rutas["csv_eventos"] = guardar_csv_eventos_temporales(reporte_evaluado, id_sesion)
        print(f"✅ Datos exportados exitosamente en: {os.getcwd()}/datos_exportados/")
    except Exception as e:
        print(f"❌ Error al exportar datos: {e}")
        
    return rutas

if __name__ == "__main__":
    print("Módulo de exportación listo, relacionalmente seguro y actualizado con métricas clínicas.")