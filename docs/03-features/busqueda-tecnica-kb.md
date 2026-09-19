# Búsqueda técnica en KB por documentos IA y códigos de servicio

Esta guía describe los cambios mínimos para que la IA priorice documentación técnica canónica, especialmente PDFs en la carpeta `IA/<Marca>/<Línea>/` y consultas de técnicos como `tengo el error service 110`.

## Objetivo

Cuando un técnico escribe una descripción corta con un código de error o servicio, el sistema debe:

1. Limitar la búsqueda a la marca/línea correcta cuando estén disponibles.
2. Priorizar documentos marcados para consumo de IA (`kb_scope=IA` o `is_ai_document=true`).
3. Exigir o favorecer chunks que contienen el código técnico exacto.
4. Devolver contexto con `source`, `page` y `document_url` para auditar la fuente.

## Metadata normalizada en ingesta

Durante la ingesta, el servidor enriquece la metadata de cada documento/chunk con campos derivados de `source`, `source_ref` o `doc_id`:

| Campo | Para qué sirve |
| --- | --- |
| `source_file` | Nombre del archivo original, por ejemplo `80.51.332_ET_es-ES_IA.pdf`. |
| `source_prefix` | Ruta del archivo sin el nombre, por ejemplo `IA/Rational/ICombi/`. |
| `source_bucket` | Bucket/host base derivado de la URL. |
| `kb_scope` | Se marca como `IA` cuando el path contiene `/IA/` o el archivo termina en `_IA`. |
| `is_ai_document` | Booleano para identificar documentos consumibles por IA. |
| `brand` | Marca inferida desde rutas tipo `/IA/Rational/ICombi/...` o `/kb/Rational/...`. |
| `line` | Línea/familia inferida desde la ruta, por ejemplo `ICombi`. |
| `primary_error_code` | Primer código técnico detectado en el chunk. |
| `error_codes` | Códigos técnicos detectados, separados por coma. |

> Recomendación: en nuevas ingestas, envía `brand`, `line` y `kb_scope` explícitos cuando puedas. La inferencia existe para compatibilidad, pero la metadata explícita es más confiable.

## Nuevo endpoint recomendado

Usa `POST /tools/kb_search_error` para casos de soporte técnico con código de error/servicio.

### Payload recomendado para el caso `service 110`

```bash
cat > /tmp/kb_error_110.json <<'JSON'
{
  "query": "tengo el error service 110",
  "error_code": "110",
  "brand": "Rational",
  "line": "ICombi",
  "top_k": 10,
  "context_chars": 2500
}
JSON

curl -sS -X POST http://localhost:7070/tools/kb_search_error \
  -H "Content-Type: application/json" \
  --data @/tmp/kb_error_110.json \
| jq '.hits[] | {
    doc_id,
    page:(.metadata.page),
    source:(.metadata.source),
    technical_score,
    technical_boosts,
    score,
    semantic_score,
    keyword_score,
    snippet,
    context,
    document_url
  }'
```

### Payload con PDF exacto

Si quieres forzar un PDF específico:

```bash
cat > /tmp/kb_error_110_pdf.json <<'JSON'
{
  "query": "tengo el error service 110",
  "error_code": "110",
  "source": "https://fixeat-dev.s3.us-east-2.amazonaws.com/IA/Rational/ICombi/80.51.332_ET_es-ES_IA.pdf",
  "top_k": 10,
  "context_chars": 2500
}
JSON

curl -sS -X POST http://localhost:7070/tools/kb_search_error \
  -H "Content-Type: application/json" \
  --data @/tmp/kb_error_110_pdf.json \
| jq '.hits[] | {
    doc_id,
    page:(.metadata.page),
    source:(.metadata.source),
    technical_score,
    technical_boosts,
    context,
    document_url
  }'
```

## Por qué existe `where_document`

`where` filtra metadata (`brand`, `line`, `kb_scope`, `source`, etc.). `where_document` filtra texto del chunk en Chroma.

El endpoint técnico usa internamente:

```json
"where_document": {"$contains": "110"}
```

Esto reduce resultados que hablan de “servicio” pero no contienen el código exacto.

## Cómo interpreta el ranking técnico

`kb_search_error` usa búsqueda híbrida y luego suma boosts explicables:

| Boost | Why |
| --- | --- |
| `primary_error_code` | El chunk fue etiquetado con ese código como principal. |
| `error_codes` | El chunk contiene el código entre sus códigos detectados. |
| `text_code_match` | El texto visible (`snippet/context`) contiene `service 110`, `Servicio 110`, `S_110`, etc. |
| `ia_scope` | El documento está marcado como consumible por IA. |
| `ia_source` | La fuente contiene `/IA/` o `_IA`. |
| `brand` / `line` | La metadata coincide con la marca/línea solicitada. |

La respuesta incluye `technical_score` y `technical_boosts` para auditar por qué un hit subió.

## Checklist de despliegue

1. Desplegar el código actualizado.
2. Reiniciar el servidor MCP/API.
3. Reingestar documentos críticos o ejecutar una migración de metadata para que chunks existentes tengan `kb_scope`, `line`, `primary_error_code` y `error_codes`.
4. Probar `/tools/kb_search_error` con queries reales de técnicos.
5. Revisar que el resultado correcto aparezca en top 3 y que `document_url` apunte a la página esperada.

## Limitaciones actuales

- `where_document` con `$contains` busca coincidencia literal; por eso el endpoint también expande la query con `servicio`, `service` y `S_`.
- Los chunks antiguos no tendrán metadata nueva hasta reingesta o migración.
- Si el extractor PDF no extrajo correctamente `Servicio 110` en la página esperada, habrá que mejorar la extracción/chunking por página.
