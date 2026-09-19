"""Demo de KB local con Chroma y Sentence-Transformers.

Provee funciones de ingesta y búsqueda (kb_search) usadas por el servidor MCP demo.
"""

from __future__ import annotations

from typing import Any
import os
import re

import chromadb
from sentence_transformers import SentenceTransformer


_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Persistencia opcional mediante variable de entorno (útil en Docker)
_chroma_path = os.getenv("CHROMA_PATH", "/data/chroma")
try:
    _chroma = chromadb.PersistentClient(path=_chroma_path)
except Exception:
    # Fallback a cliente en memoria si la ruta no es válida (entorno local)
    _chroma = chromadb.Client()
_collection = _chroma.get_or_create_collection("kb_tech")


def get_all_documents() -> list[dict[str, Any]]:
    """Obtiene todos los documentos del KB para análisis de taxonomía."""
    try:
        # Obtener todos los documentos de la colección (sin incluir 'ids' explícitamente)
        results = _collection.get(include=["documents", "metadatas"])
        
        documents = []
        for i, doc_id in enumerate(results["ids"]):
            documents.append({
                "id": doc_id,
                "text": results["documents"][i] if i < len(results["documents"]) else "",
                "metadata": results["metadatas"][i] if i < len(results["metadatas"]) else {}
            })
        
        return documents
        
    except Exception as e:
        print(f"Error obteniendo documentos del KB: {e}")
        return []


def ingest_docs(docs: list[dict[str, Any]]) -> None:
    texts = [d["text"] for d in docs]
    embeddings = _model.encode(texts, normalize_embeddings=True).tolist()
    # Chroma requiere metadatas no vacíos; forzamos un valor por defecto
    metadatas = []
    for d in docs:
        md = d.get("metadata") or {"source": "unspecified"}
        # asegurar al menos un atributo
        if isinstance(md, dict) and len(md) == 0:
            md = {"source": "unspecified"}
        metadatas.append(md)

    # Usar upsert para permitir actualizar documentos existentes
    _collection.upsert(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


def _find_best_match_position(query: str, full_text: str, window_size: int = 100) -> int:
    """Encuentra la mejor posición del match usando similitud de embeddings.
    
    Args:
        query: Consulta de búsqueda
        full_text: Texto completo del documento
        window_size: Tamaño de ventana para buscar matches
        
    Returns:
        Posición aproximada del mejor match en el texto
    """
    if not full_text:
        return 0
    
    # Para documentos cortos, retornar inicio
    if len(full_text) <= window_size * 2:
        return 0
    
    # Dividir texto en ventanas solapadas
    step = window_size // 2
    windows = []
    positions = []
    
    for i in range(0, len(full_text) - window_size, step):
        windows.append(full_text[i:i + window_size])
        positions.append(i)
    
    if not windows:
        return 0
    
    # Encontrar ventana más similar a la query
    try:
        query_emb = _model.encode([query], normalize_embeddings=True)[0]
        windows_emb = _model.encode(windows, normalize_embeddings=True)
        
        # Calcular similitud coseno
        import numpy as np
        similarities = np.dot(windows_emb, query_emb)
        best_idx = int(np.argmax(similarities))
        
        return positions[best_idx]
    except Exception:
        # Fallback: buscar primer término de query en texto
        query_terms = query.lower().split()[:3]  # Primeros 3 términos
        for term in query_terms:
            pos = full_text.lower().find(term)
            if pos != -1:
                return max(0, pos - window_size // 2)
        return 0


def _extract_context_window(
    text: str, 
    center_pos: int, 
    context_chars: int
) -> tuple[str, int, int]:
    """Extrae ventana de contexto alrededor de una posición.
    
    Args:
        text: Texto completo
        center_pos: Posición central del contexto
        context_chars: Número de caracteres de contexto deseados
        
    Returns:
        Tupla de (contexto, start_pos, end_pos)
    """
    if not text:
        return ("", 0, 0)
    
    # Calcular límites de la ventana
    half_window = context_chars // 2
    start = max(0, center_pos - half_window)
    end = min(len(text), center_pos + half_window)
    
    # Ajustar para no cortar palabras (buscar espacios cercanos)
    if start > 0:
        # Buscar espacio hacia atrás (máximo 50 chars)
        for i in range(start, max(0, start - 50), -1):
            if text[i].isspace():
                start = i + 1
                break
    
    if end < len(text):
        # Buscar espacio hacia adelante (máximo 50 chars)
        for i in range(end, min(len(text), end + 50)):
            if text[i].isspace():
                end = i
                break
    
    context = text[start:end].strip()
    
    # Agregar indicadores si hay texto antes/después
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."
    
    return (context, start, end)


def kb_search(
    query: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Búsqueda semántica en KB (versión original, mantiene compatibilidad).
    
    Args:
        query: Consulta de búsqueda
        top_k: Número de resultados a retornar
        where: Filtros de metadata opcionales
        
    Returns:
        Lista de hits con doc_id, score, snippet (500 chars) y metadata
    """
    q_emb = _model.encode([query], normalize_embeddings=True).tolist()[0]
    # Filtro por metadatos (opcional)
    kwargs: dict[str, Any] = {"query_embeddings": [q_emb], "n_results": top_k}
    if where and isinstance(where, dict) and len(where) > 0:
        kwargs["where"] = where
    if where_document and isinstance(where_document, dict) and len(where_document) > 0:
        kwargs["where_document"] = where_document
    res = _collection.query(**kwargs)
    hits: list[dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        hits.append(
            {
                "doc_id": res["ids"][0][i],
                "score": float(res["distances"][0][i]),
                "snippet": res["documents"][0][i][:500],
                "metadata": res["metadatas"][0][i],
            }
        )
    return hits


def extract_key_terms(query: str) -> list[str]:
    """Extrae términos clave de una consulta para highlighting.
    
    Args:
        query: Consulta de búsqueda
        
    Returns:
        Lista de términos clave (normalizados a minúsculas)
    """
    import re
    
    # Remover palabras comunes (stop words en español)
    stop_words = {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'de', 'del', 'en', 'con', 'por', 'para', 'sin', 'sobre',
        'y', 'o', 'pero', 'que', 'si', 'como', 'cuando', 'donde',
        'este', 'esta', 'ese', 'esa', 'aquel', 'aquella',
        'no', 'es', 'son', 'está', 'están', 'ser', 'estar'
    }
    
    # Tokenizar y limpiar
    tokens = re.findall(r'\b\w+\b', query.lower())
    
    # Filtrar stop words y tokens muy cortos
    key_terms = [t for t in tokens if t not in stop_words and len(t) >= 3]
    
    return key_terms


def highlight_text(text: str, terms: list[str], marker: str = "**") -> str:
    """Resalta términos clave en un texto.
    
    Args:
        text: Texto donde resaltar
        terms: Lista de términos a resaltar
        marker: Marcador para el highlighting (default: ** para markdown bold)
        
    Returns:
        Texto con términos resaltados
    """
    import re
    
    if not terms:
        return text
    
    highlighted = text
    
    # Resaltar cada término (case-insensitive)
    for term in terms:
        # Usar word boundary para evitar matches parciales
        pattern = rf'\b({re.escape(term)})\b'
        highlighted = re.sub(
            pattern,
            f'{marker}\\1{marker}',
            highlighted,
            flags=re.IGNORECASE
        )
    
    return highlighted


def generate_document_url(doc_id: str, metadata: dict[str, Any] | None = None) -> str:
    """Genera URL navegable para un documento con soporte multi-formato.
    
    Construye URLs navegables basadas en:
    - doc_id: identificador del documento/chunk
    - metadata.source: ruta o URL del documento original
    - metadata.page: número de página (si existe)
    
    Formatos soportados para PDFs:
    1. URLs S3/HTTP con #page=N (Adobe Reader, Chrome)
    2. URLs S3/HTTP con #p=N (alternativa)
    3. Google Docs Viewer: https://docs.google.com/viewer?url=...&page=N
    4. Rutas locales: endpoint /view-document/ con ?page=N
    
    Args:
        doc_id: ID del documento o chunk
        metadata: Metadata del documento (debe incluir 'source' y opcionalmente 'page')
        
    Returns:
        URL navegable al documento (con página si está disponible)
    """
    if metadata is None:
        metadata = {}
    
    source = metadata.get("source", "")
    page = metadata.get("page")
    
    # Caso 1: Source es URL (HTTP/HTTPS/S3)
    if source and source.startswith(("http://", "https://", "s3://")):
        base_url = source
        
        # Limpiar fragmento existente si hay
        if "#" in base_url:
            base_url = base_url.split("#")[0]
        if "?" in base_url:
            base_url = base_url.split("?")[0]
        
        # Agregar página si existe - usar múltiples formatos para compatibilidad
        if page is not None:
            # Formato 1: Estándar #page=N (funciona en Chrome, Firefox, Adobe Reader)
            url_standard = f"{base_url}#page={page}"
            
            # Formato 2: Alternativo con query param (algunos visores)
            # url_query = f"{base_url}?page={page}"
            
            # Formato 3: Google Docs Viewer (más compatible pero requiere URL pública)
            # url_google = f"https://docs.google.com/viewer?url={urllib.parse.quote(base_url)}&embedded=true#page={page}"
            
            # Por defecto usar formato estándar
            # El cliente puede modificar según su visor
            return url_standard
        else:
            return base_url
    
    # Caso 2: doc_id tiene formato legacy con #c (chunk) y posible URL
    if "#c" in doc_id:
        base_id = doc_id.split("#")[0]
        
        # Si el base_id es una URL, usarla
        if base_id.startswith(("http://", "https://", "s3://")):
            if page is not None:
                return f"{base_id}#page={page}"
            else:
                return base_id
        
        # Si source está disponible, usarlo
        if source:
            if page is not None:
                return f"{source}#page={page}"
            else:
                return source
        
        # Si hay página pero no source, usar formato legacy mejorado
        if page is not None:
            return f"/view-document/{base_id}?page={page}"
        else:
            return f"/view-document/{doc_id}"
    
    # Caso 3: Source con formato "default.services", "default.activities", etc.
    # Estos son datos importados de BD - agregar metadata descriptivo
    if source and source.startswith("default."):
        source_type = source.split(".")[-1]  # "services", "activities", etc.
        if page is not None:
            return f"/view-document/{doc_id}?page={page}&source={source_type}"
        else:
            return f"/view-document/{doc_id}?source={source_type}"
    
    # Caso 4: Ruta local o sin source específico
    # Usar endpoint de visualización del servidor MCP
    if page is not None:
        return f"/view-document/{doc_id}?page={page}"
    else:
        return f"/view-document/{doc_id}"


def kb_search_extended(
    query: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    context_chars: int = 2000,
    include_full_text: bool = False,
    highlight_terms: bool = True
) -> list[dict[str, Any]]:
    """Búsqueda semántica con contexto ampliado y metadata enriquecida.
    
    Esta versión extendida proporciona:
    - Contexto ampliado configurable (vs 500 chars fijos)
    - Ventana de contexto centrada en el match
    - Metadata enriquecida con posiciones
    - Opción de incluir texto completo
    - URLs navegables a documentos (Fase 2)
    - Highlighting de términos clave (Fase 3)
    
    Args:
        query: Consulta de búsqueda
        top_k: Número de resultados a retornar
        where: Filtros de metadata opcionales
        context_chars: Número de caracteres de contexto (default: 2000)
        include_full_text: Si incluir texto completo en respuesta (default: False)
        highlight_terms: Si resaltar términos clave en contexto (default: True)
        
    Returns:
        Lista de hits con:
        - doc_id: ID del documento
        - score: Score de relevancia (distancia)
        - snippet: Primeros 500 chars (compatibilidad)
        - context: Ventana de contexto ampliada
        - context_highlighted: Contexto con términos resaltados (si highlight_terms=True)
        - full_text: Texto completo (si include_full_text=True)
        - metadata: Metadata enriquecida con:
            - match_position: Posición del match en el texto
            - context_start: Inicio de la ventana de contexto
            - context_end: Fin de la ventana de contexto
            - [metadata original del documento]
        - document_url: URL navegable al documento
        - highlighted_terms: Lista de términos resaltados (si highlight_terms=True)
    """
    # Realizar búsqueda semántica base
    q_emb = _model.encode([query], normalize_embeddings=True).tolist()[0]
    kwargs: dict[str, Any] = {"query_embeddings": [q_emb], "n_results": top_k}
    if where and isinstance(where, dict) and len(where) > 0:
        kwargs["where"] = where
    if where_document and isinstance(where_document, dict) and len(where_document) > 0:
        kwargs["where_document"] = where_document
    
    res = _collection.query(**kwargs)
    
    # Extraer términos clave para highlighting
    key_terms = extract_key_terms(query) if highlight_terms else []
    
    hits: list[dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        doc_id = res["ids"][0][i]
        full_text = res["documents"][0][i]
        metadata = res["metadatas"][0][i] or {}
        score = float(res["distances"][0][i])
        
        # Encontrar mejor posición del match en el texto
        match_pos = _find_best_match_position(query, full_text)
        
        # Extraer ventana de contexto ampliada
        context, context_start, context_end = _extract_context_window(
            full_text, match_pos, context_chars
        )
        
        # Construir metadata enriquecida
        enriched_metadata = {
            **metadata,  # Metadata original
            "match_position": match_pos,
            "context_start": context_start,
            "context_end": context_end,
            "text_length": len(full_text),
        }
        
        # Generar URL navegable al documento
        document_url = generate_document_url(doc_id, metadata)
        
        # Construir hit con información extendida
        hit = {
            "doc_id": doc_id,
            "score": score,
            "snippet": full_text[:500],  # Mantener para compatibilidad
            "context": context,  # Contexto ampliado
            "metadata": enriched_metadata,
            "document_url": document_url,  # URL navegable (NUEVO en Fase 2)
        }
        
        # Agregar highlighting si está habilitado
        if highlight_terms and key_terms:
            hit["context_highlighted"] = highlight_text(context, key_terms, marker="**")
            hit["highlighted_terms"] = key_terms
        
        # Incluir texto completo si se solicita
        if include_full_text:
            hit["full_text"] = full_text
        
        hits.append(hit)
    
    return hits


def _detect_error_codes(query: str) -> list[str]:
    """Detecta códigos de error en la query.
    
    Busca patrones como:
    - service 25, servicio 25
    - error 25
    - S_25, S25
    - código 25
    
    Returns:
        Lista de códigos detectados (e.g., ["25"])
    """
    patterns = [
        r'\bservice\s+(\d+)',
        r'\bservicio\s+(\d+)',
        r'\berror\s+(\d+)',
        r'\bS_?(\d+)',
        r'\bcódigo\s+(\d+)',
        r'\bcode\s+(\d+)',
    ]
    
    codes = set()
    query_lower = query.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, query_lower, re.IGNORECASE)
        codes.update(matches)
    
    return sorted(list(codes))


def _keyword_boost_search(
    query: str,
    error_codes: list[str],
    top_k: int = 20,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Búsqueda por keyword con scoring para códigos de error.
    
    Args:
        query: Query original
        error_codes: Códigos de error detectados
        top_k: Número de resultados
        where: Filtros de metadata
        
    Returns:
        Dict de {doc_id: keyword_score}
    """
    if not error_codes:
        return {}
    
    # Construir patrones regex tolerantes a saltos de línea y guiones de PDF.
    search_patterns: list[tuple[re.Pattern[str], float]] = []
    for code in error_codes:
        escaped = re.escape(code)
        search_patterns.extend([
            (re.compile(rf"\bservicio\s*{escaped}\b", re.I), 3.0),
            (re.compile(rf"\bservice\s*{escaped}\b", re.I), 3.0),
            (re.compile(rf"\bs[_\-\s]?{escaped}\b", re.I), 3.0),
            (re.compile(rf"\berror\s*{escaped}\b", re.I), 2.0),
            (re.compile(rf"\b{escaped}\b", re.I), 0.5),
        ])

    # Obtener todos los documentos (o filtrados). Chroma no soporta `$contains`
    # sobre metadata `where`; para texto debe usarse `where_document`.
    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if where:
        get_kwargs["where"] = where
    if where_document:
        get_kwargs["where_document"] = where_document

    try:
        results = _collection.get(**get_kwargs)
    except Exception:
        return {}

    # Scoring por matches de keywords.
    keyword_scores = {}
    for i, doc_id in enumerate(results["ids"]):
        text = results["documents"][i] if i < len(results["documents"]) else ""
        score = 0.0

        for pattern, pattern_weight in search_patterns:
            count = len(pattern.findall(text))
            if count > 0:
                score += count * pattern_weight

        if score > 0:
            keyword_scores[doc_id] = score

    return keyword_scores


def kb_search_hybrid(
    query: str,
    top_k: int = 10,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    semantic_weight: float = 0.5,
    keyword_weight: float = 0.5,
    context_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Búsqueda híbrida: combina búsqueda semántica con keyword matching.
    
    Especialmente útil para códigos de error técnicos donde la búsqueda semántica
    puede fallar en encontrar matches exactos.
    
    Args:
        query: Consulta de búsqueda
        top_k: Número de resultados finales
        where: Filtros de metadata opcionales
        semantic_weight: Peso para score semántico (0-1)
        keyword_weight: Peso para score de keywords (0-1)
        context_chars: Caracteres de contexto
        
    Returns:
        Lista de hits con scores híbridos, ordenados por relevancia
    """
    # 1. Detectar códigos de error en la query
    error_codes = _detect_error_codes(query)
    
    # 2. Si no hay códigos de error, usar búsqueda semántica normal
    if not error_codes:
        return kb_search_extended(
            query=query,
            top_k=top_k,
            where=where,
            where_document=where_document,
            context_chars=context_chars,
            highlight_terms=False
        )
    
    # 3. Búsqueda semántica (top_k * 2 para tener más candidatos)
    semantic_results = kb_search_extended(
        query=query,
        top_k=top_k * 3,  # Obtener más candidatos
        where=where,
        where_document=where_document,
        context_chars=context_chars,
        highlight_terms=False
    )
    
    # 4. Búsqueda por keywords
    keyword_scores = _keyword_boost_search(
        query=query,
        error_codes=error_codes,
        top_k=top_k * 3,
        where=where,
        where_document=where_document,
    )
    
    # 5. Normalizar scores semánticos a [0-1]
    semantic_scores = {}
    if semantic_results:
        max_sem_score = max(r["score"] for r in semantic_results)
        min_sem_score = min(r["score"] for r in semantic_results)
        score_range = max_sem_score - min_sem_score if max_sem_score > min_sem_score else 1.0
        
        for r in semantic_results:
            # Chroma devuelve distancias: menor distancia implica mayor relevancia.
            normalized = 1.0 - ((r["score"] - min_sem_score) / score_range)
            semantic_scores[r["doc_id"]] = normalized
    
    # 6. Normalizar keyword scores a [0-1]
    if keyword_scores:
        max_kw_score = max(keyword_scores.values())
        if max_kw_score > 0:
            keyword_scores = {k: v/max_kw_score for k, v in keyword_scores.items()}
    
    # 7. Combinar scores híbridos
    combined_scores = {}
    all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    for doc_id in all_doc_ids:
        sem_score = semantic_scores.get(doc_id, 0.0)
        kw_score = keyword_scores.get(doc_id, 0.0)
        
        # Score híbrido: dar más peso a keyword matches para códigos de error
        combined_scores[doc_id] = (
            semantic_weight * sem_score + 
            keyword_weight * kw_score
        )
    
    # 8. Reordenar resultados por score híbrido
    semantic_results_dict = {r["doc_id"]: r for r in semantic_results}
    
    # Obtener documentos adicionales que solo aparecieron en keyword search
    for doc_id in keyword_scores:
        if doc_id not in semantic_results_dict:
            # Recuperar el documento completo
            try:
                doc_result = _collection.get(ids=[doc_id], include=["documents", "metadatas"])
                if doc_result and doc_result["ids"]:
                    semantic_results_dict[doc_id] = {
                        "doc_id": doc_id,
                        "score": 0.0,  # No tuvo score semántico
                        "snippet": doc_result["documents"][0][:500] if doc_result["documents"] else "",
                        "context": doc_result["documents"][0][:context_chars] if doc_result["documents"] else "",
                        "metadata": doc_result["metadatas"][0] if doc_result["metadatas"] else {},
                        "document_url": generate_document_url(
                            doc_id,
                            doc_result["metadatas"][0] if doc_result["metadatas"] else {},
                        ),
                    }
            except Exception:
                continue
    
    # Crear lista final con scores híbridos
    hybrid_results = []
    for doc_id, hybrid_score in sorted(combined_scores.items(), key=lambda x: -x[1])[:top_k]:
        if doc_id in semantic_results_dict:
            result = semantic_results_dict[doc_id].copy()
            result["score"] = hybrid_score
            result["semantic_score"] = semantic_scores.get(doc_id, 0.0)
            result["keyword_score"] = keyword_scores.get(doc_id, 0.0)
            result["error_codes_found"] = error_codes
            hybrid_results.append(result)
    
    return hybrid_results


if __name__ == "__main__":
    ingest_docs(
        [
            {"id": "m1", "text": "Manual modelo X: revisar filtro y bomba"},
            {"id": "t1", "text": "Tip técnico: sensor T900 falla con humedad"},
        ]
    )
    print(kb_search("problema de bomba en modelo X", top_k=2))


