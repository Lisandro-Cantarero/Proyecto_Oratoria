import spacy
from spacy.matcher import PhraseMatcher
from typing import Dict, List, Any

# Carga del modelo lingüístico determinista (requiere: python -m spacy download es_core_news_sm)
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    raise OSError("Falta el modelo de spaCy. Ejecuta: python -m spacy download es_core_news_sm")

# Taxonomía funcional de disfluencias léxicas en español (Vera-Ramírez et al., 2025)
TAXONOMIA_DISFLUENCIAS_ESPANOL = {
    "transicion": ["entonces", "bueno", "pues", "luego", "este", "ya"],
    "revision": ["o sea", "es decir", "mejor dicho", "digo"],
    "apoyo": ["básicamente", "digamos", "literalmente", "tipo", "como que", "verdad", "claro"],
    "apelacion": ["me explico", "cierto", "entiendes"]
}

def analizar_lexico_y_taxonomia(
    texto_transcrito: str, 
    taxonomia_disfluencias: Dict[str, List[str]] = TAXONOMIA_DISFLUENCIAS_ESPANOL,
    marcas_whisper: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Procesa el texto para extraer métricas morfosintácticas estándar y contabilizar 
    muletillas clasificadas por su función pragmática (Vera-Ramírez et al., 2025).
    
    Alinea temporalmente las coincidencias exactas del PhraseMatcher con los 
    timestamps generados por el modelo ASR (Whisper), manejando desincronizaciones 
    y puntuación residual.
    """
    if not texto_transcrito or not texto_transcrito.strip():
        return {"error": "El texto proporcionado está vacío."}

    # 1. Procesamiento sintáctico base
    doc = nlp(texto_transcrito)
    
    # 2. Inicialización del PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for categoria, lista_muletillas in taxonomia_disfluencias.items():
        patrones = [nlp.make_doc(muletilla.lower()) for muletilla in lista_muletillas]
        matcher.add(categoria, patrones)

    # 3. Métricas de Normalización y Riqueza Léxica (Lingüística de Corpus Estándar)
    total_palabras_w = 0
    total_oraciones_s = len(list(doc.sents))
    conteo_adjetivos = 0
    conteo_adverbios = 0
    
    for token in doc:
        if token.is_punct or token.is_space:
            continue
            
        total_palabras_w += 1
        
        if token.pos_ == "ADJ":
            conteo_adjetivos += 1
        elif token.pos_ == "ADV":
            conteo_adverbios += 1

    # 4. Extracción de Taxonomía y Alineación Temporal (Vera-Ramírez et al., 2025)
    conteo_por_categoria = {categoria: 0 for categoria in taxonomia_disfluencias.keys()}
    detalles_encontrados = []
    
    matches = matcher(doc)
    indice_whisper_actual = 0 # Cursor de optimización
    
    # Caracteres de puntuación exhaustivos del español (se agregó guiones para limpieza profunda)
    caracteres_basura = ".,!?;:¿¡\"'()[] -_"

    for match_id, start_idx, end_idx in matches:
        regla_categoria = nlp.vocab.strings[match_id]
        span_encontrado = doc[start_idx:end_idx]
        
        conteo_por_categoria[regla_categoria] += 1
        
        inicio_ts = None
        fin_ts = None
        
        # Cruce con las marcas de tiempo de WhisperX
        if marcas_whisper:
            palabras_span = [t.text.lower() for t in span_encontrado if not t.is_punct and not t.is_space]
            
            # BUG CORREGIDO: Eliminamos el límite de +50. Ahora escanea todo el remanente del arreglo.
            for i in range(indice_whisper_actual, len(marcas_whisper)):
                w_word = marcas_whisper[i].get("word", "").lower().strip(caracteres_basura)
                
                if palabras_span and w_word == palabras_span[0]:
                    coincidencia_completa = True
                    
                    for j in range(1, len(palabras_span)):
                        if i + j >= len(marcas_whisper):
                            coincidencia_completa = False
                            break
                        w_next = marcas_whisper[i+j].get("word", "").lower().strip(caracteres_basura)
                        if w_next != palabras_span[j]:
                            coincidencia_completa = False
                            break
                    
                    if coincidencia_completa:
                        # Extraemos tiempos de forma segura (por si WhisperX retorna un token sin tiempo)
                        inicio_raw = marcas_whisper[i].get("start")
                        fin_raw = marcas_whisper[i + len(palabras_span) - 1].get("end")
                        
                        inicio_ts = round(inicio_raw, 2) if inicio_raw is not None else None
                        fin_ts = round(fin_raw, 2) if fin_raw is not None else None
                        
                        # Avanzar solo 1 para no perder solapamientos
                        indice_whisper_actual = i + 1 
                        break

        detalles_encontrados.append({
            "categoria": regla_categoria,
            "expresion": span_encontrado.text,
            "inicio": inicio_ts,
            "fin": fin_ts
        })

    total_muletillas_lexicas = sum(conteo_por_categoria.values())

    # 5. Cálculo de Densidades Relativas (Vera-Ramírez et al., 2025)
    if total_palabras_w == 0:
        return {"error": "No se detectaron palabras léxicas válidas en el texto."}

    densidades_pragmaticas = {
        f"densidad_{cat}": round(conteo / total_palabras_w, 4) 
        for cat, conteo in conteo_por_categoria.items()
    }

    palabras_por_oracion = total_palabras_w / total_oraciones_s if total_oraciones_s > 0 else float(total_palabras_w)

    return {
        "W_total_palabras": total_palabras_w,
        "S_total_oraciones": total_oraciones_s,
        "palabras_por_oracion_promedio": round(palabras_por_oracion, 2),
        "conteo_adjetivos": conteo_adjetivos,
        "conteo_adverbios": conteo_adverbios,
        "densidad_adjetivos": round(conteo_adjetivos / total_palabras_w, 4),
        "densidad_adverbios": round(conteo_adverbios / total_palabras_w, 4),
        "perfil_pragmatico": {
            "total_muletillas_lexicas": total_muletillas_lexicas,
            "conteo_por_categoria": conteo_por_categoria,
            "densidades": densidades_pragmaticas,
            "detalles": detalles_encontrados
        }
    }