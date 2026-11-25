# Arquitectura

```
USGS ──► /vigilar (api/main.py, Render) ──► Supabase (zonas_sismicas) ──► Telegram
              ▲ cron: .github/workflows            ▲ lee/escribe          ▲ bot (Gemini)
              │                                    │
         modelo_sismos.joblib              web (public/, Vercel) ── Supabase Auth + alertas_usuario
```

| Carpeta | Qué hay |
|---|---|
| `public/` | Web estática (Vercel). Supabase JS con la clave anon |
| `api/` | FastAPI: `/predict`, `/health`, `/vigilar`, webhook Telegram; `train.py`, modelo `.joblib` |
| `agente/` | `vigilante.py` que llama a `/vigilar` desde el cron de GitHub Actions |
| `mcp/` | Servidor MCP que consulta `zonas_sismicas` |
| `data/` | CSV de entrenamiento (`zona_dia_11a.csv`, ignorado por git) |
| `scripts/` | Telemetría de tokens |
| `docs/` | Documentación (este archivo, `todo.md`, `claude/`, `deploy/`) |

## Usuarios (Supabase Auth)
- La web usa `signUp` / `signInWithPassword`; los usuarios viven en `auth.users`. Cada alerta guarda `user_id`.
- Que cada usuario vea solo lo suyo depende de **RLS** en `alertas_usuario` (no está en el repo; verificar en el dashboard). SQL recomendado:
  `alter table alertas_usuario enable row level security;` + políticas `select/insert/delete` con `using (auth.uid() = user_id)`; `zonas_sismicas` con `select` público y sin escritura.
- RLS activo (SQL en `db/seguridad_y_telegram.sql`). Vínculo Telegram: la web crea un código en `vinculos_telegram`; el bot recibe `/start <código>` y guarda `chat_id`. `/vigilar` (service key) envía cada alerta al chat de su dueño y omite a quien no vinculó.

## RAG (implementado 2026-11-24)
- Base de conocimiento: `rag/conocimiento/*.md` (se parte por `## `). SQL: `db/rag_pgvector.sql` (tabla `conocimiento` con `vector(768)` y función `buscar_conocimiento`).
- Indexar: `GET /rag/ingestar?key=<CRON_SECRET>` en Render (embeddings `gemini-embedding-001`, 768 dim; `RAG_EMBEDDINGS=openai` usa `text-embedding-3-small`, y obliga a reindexar). Repetir tras editar los `.md`.
- Consulta: `procesar_consulta_telegram` embebe la pregunta, recupera los 3 fragmentos con similitud >= 0.55 y los añade al prompt de Gemini; los datos en vivo siguen entrando por SQL (RAG híbrido). Si falla, el bot responde sin RAG.

## Métricas de actividad (ventana de 96 h, fuente USGS ≥ M2.5)
- `loc200_n96_con`: sismos a ≤ 200 km del centro (`sin45` = leves < M4.5). `loc500_n96_con`: a ≤ 500 km, incluye los anteriores. `lat_n96_prof70_150`: profundidad intermedia (70–150 km) **dentro de 500 km**.
- Web (`nivelRiesgo`) y Telegram (`describir_nivel_usuario`) usan la misma lógica; sin sismos cerca la etiqueta es «Alza del modelo / solo estadística» y se indica qué impulsa el valor.

## Modelos del bot de Telegram
- Orden: OpenAI (`OPENAI_MODEL`, por defecto `gpt-6-luna`, `OPENAI_REASONING=low`) → Gemini (`GEMINI_MODELS`, lista separada por comas). Cada fallo se registra como `[openai]`/`[gemini]` en el log de Render.
- Variables en Render: `OPENAI_API_KEY`, `GEMINI_API_KEY`, opcionales las anteriores. Probado 2026-11-24: Luna responde; embeddings de OpenAI dieron «no credits», los de Gemini funcionan.
