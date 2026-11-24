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
- `/vigilar` usa la service key (salta RLS) y hoy envía **todas** las alertas al único `TELEGRAM_CHAT_ID` (el dueño). Para que cada usuario reciba las suyas hay que guardar su `telegram_chat_id` en la alerta/perfil.

## RAG
- No hay RAG. El bot de Telegram inyecta toda la tabla `zonas_sismicas` en el prompt de Gemini (contexto fijo, no recuperación).
- Propuesta: Supabase **pgvector** + embeddings de Gemini sobre una base de conocimiento (glosario, sismos históricos significativos por zona, guías de protección civil) y recuperar los 3-4 fragmentos más cercanos a la pregunta; los datos en vivo siguen entrando por SQL.
