## MCP Tools Specification

### kb_search
Input
```json
{ "query": "string", "top_k": 5, "where": {"brand": {"$eq": "ACME"}, "model": {"$eq": "T900"}} }
```
Output
```json
{ "hits": [{"doc_id":"string","score":0.0,"snippet":"string","metadata":{}}] }
```
Notas:
- `where` es opcional y filtra por metadatos de la colección (Chroma). Operadores soportados: `$eq`, `$contains`, `$in` (según backend).
- Recomendado: etiquetar `metadata` al ingerir (`brand`, `model`, `category`, `source`).

### inventory_lookup
Input: `{ "modelo": "string", "repuesto": "string" }`
Output: `{ "disponible": true, "sku": "string", "eta_dias": 0 }`

### ticket_update
Input: `{ "ticket_id": "string", "comentario": "string" }`
Output: `{ "ok": true }`

### kb_ingest
Permite ingresar documentos en texto y/o URLs (scrapeo HTML básico) a la KB. Puede habilitar curación automática.

Input
```json
{
  "docs": [
    { "id": "string?", "text": "string", "metadata": {} },
    { "filename": "manual.pdf", "file_base64": "...", "mime_type": "application/pdf" },
    { "filename": "manual.docx", "file_base64": "..." },
    { "filename": "tabla.xlsx", "file_base64": "..." }
  ],
  "urls": ["https://.../manual.html", "https://.../manual.pdf", "https://.../tabla.xlsx"],
  "auto_curate": true
}
```
Output
```json
{ "ingested": 3, "from_urls": 1, "curated": true, "stats": {"input": 5, "curated": 3, "quarantine": 2} }
```
### kb_curate
Normaliza, enriquece, fragmenta y puntúa contenido para la KB sin escribirlo. Sirve para validar/previzualizar cargas, o como paso previo a `kb_ingest`.

Input
```json
{
  "docs": [{ "id": "raw-1", "text": "...", "metadata": {"source":"manual_pdf","brand":"ACME","model":"T900"} }],
  "urls": ["https://example.com/manual.html"],
  "url_headers": {"Cookie": "..."}
}
```
Output
```json
{
  "docs": [
    { "id": "acme-t900#c0", "text": "...chunk...", "metadata": {"brand":"ACME","model":"T900","chunk_index":0,"fingerprint":"sha256:...","quality_score":0.9} }
  ],
  "quarantine": [{"id":"acme-t900#cN","reason":"low_quality_or_too_short"}],
  "stats": {"input": 2, "curated": 1, "quarantine": 1}
}
```

Notas
- El esquema de salida sigue el “CanonicalDoc” (ver docs/llm.md o docs/arquitectura.md).
- `quality_score` y `fingerprint` ayudan a deduplicar, revisar y gobernar el contenido.

### Límites y tiempos
- Timeout por tool: 5s (dev), 2s (prod) con reintentos controlados.



### kb_search_error
Búsqueda técnica orientada a códigos de servicio/error. Centraliza filtros de metadata (`brand`, `line`, `kb_scope` o `source` exacto), usa filtro textual `where_document` para exigir el código en el chunk y devuelve boosts explicables (`technical_score`, `technical_boosts`).

Input
```json
{
  "query": "tengo el error service 110",
  "error_code": "110",
  "brand": "Rational",
  "line": "ICombi",
  "kb_scope": "IA",
  "source": "https://fixeat-dev.s3.us-east-2.amazonaws.com/IA/Rational/ICombi/80.51.332_ET_es-ES_IA.pdf",
  "top_k": 10,
  "context_chars": 2500,
  "semantic_weight": 0.2,
  "keyword_weight": 0.8
}
```

Output
```json
{
  "hits": [
    {
      "doc_id": "80.51.332_ET_es-ES_IA_page_148_chunk_0",
      "technical_score": 6.8,
      "technical_boosts": {
        "text_code_match": 2.0,
        "ia_scope": 1.0,
        "brand": 0.25,
        "line": 0.25
      },
      "metadata": {
        "brand": "Rational",
        "line": "ICombi",
        "kb_scope": "IA",
        "page": 148,
        "source": "https://.../80.51.332_ET_es-ES_IA.pdf"
      },
      "context": "...Servicio 110...",
      "document_url": "https://.../80.51.332_ET_es-ES_IA.pdf#page=148"
    }
  ],
  "search_type": "technical_error"
}
```
