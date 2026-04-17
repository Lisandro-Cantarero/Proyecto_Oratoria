import time
import os
import json

class SessionLogger:
    """
    Motor de registro de eventos (Event-Driven Logger) EXCLUSIVO PARA KINÉSICA EN TIEMPO REAL.
    Captura las anomalías visuales al instante sin afectar el rendimiento y exporta 
    un JSON unificado para la Fase Científica.
    """
    def __init__(self, nombre_usuario="Orador"):
        self.start_time = time.time()
        self.active_events = {}
        
        # --- LISTA ÚNICA PARA EVENTOS VISUALES ---
        self.visual_events = [] 
        
        if not os.path.exists("reportes"):
            os.makedirs("reportes")
            
        self.nombre_usuario = nombre_usuario
        nombre_limpio = nombre_usuario.replace(" ", "_")
        self.session_id = f"{nombre_limpio}_{time.strftime('%Y%m%d_%H%M%S')}"

    def get_elapsed_time(self):
        return time.time() - self.start_time

    def format_time(self, seconds):
        """Convierte segundos brutos a formato MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def update_metric(self, metric_name, is_error, error_type=None):
        """
        Máquina de estados: Abre un evento cuando is_error es True, 
        y lo cierra (calculando su duración) cuando vuelve a False o cambia de tipo.
        """
        current_time = self.get_elapsed_time()

        if is_error:
            if metric_name not in self.active_events:
                
                self.active_events[metric_name] = {"start": current_time, "type": error_type}
            elif self.active_events[metric_name]["type"] != error_type:
                
                self._close_event(metric_name, current_time)
                self.active_events[metric_name] = {"start": current_time, "type": error_type}
        else:
            if metric_name in self.active_events:
                
                self._close_event(metric_name, current_time)

    def _close_event(self, metric_name, end_time):
        """Cierra el evento, calcula la duración y lo empaqueta para el reporte final."""
        event = self.active_events.pop(metric_name)
        start_time = event["start"]
        duration = end_time - start_time
        
        if duration >= 1.5: 
            self.visual_events.append({
                "tiempo": f"[{self.format_time(start_time)} - {self.format_time(end_time)}]",  # Formato elegante [MM:SS - MM:SS]
                "tipo": metric_name,                                       # Categoría (ej. "Mirada")
                "detalle": f"{event['type']} ({round(duration, 1)}s)",     # Detalle técnico + Duración
                "start_sec": round(start_time, 2),                         # Para ordenamiento matemático posterior
                "duracion_sec": round(duration, 2),
                "contexto_texto": "Capturado en tiempo real"               # Placeholder para el PDF
            })

    def exportar_json_tiempo_real(self, ruta_salida):
        """
        Cierra los eventos pendientes y exporta la bitácora a un archivo JSON
        compatible con la arquitectura de la Fase Científica.
        """
        current_time = self.get_elapsed_time()
        
        # 1. Cerrar cualquier evento que siguiera activo al presionar 'Q'
        for metric in list(self.active_events.keys()):
            self._close_event(metric, current_time)

        # 2. Ordenar toda la bitácora cronológicamente
        self.visual_events.sort(key=lambda x: x["start_sec"])

        # 3. Estructurar el diccionario de exportación
        datos_exportacion = {
            "metadata_sesion": {
                "id_sesion": self.session_id,
                "orador": self.nombre_usuario,
                "duracion_grabacion_segundos": round(current_time, 2),
            },
            "eventos_tiempo_real": self.visual_events
        }

        # 4. Volcar a disco
        with open(ruta_salida, "w", encoding='utf-8') as f:
            json.dump(datos_exportacion, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Bitácora kinésica guardada exitosamente en: {ruta_salida}")