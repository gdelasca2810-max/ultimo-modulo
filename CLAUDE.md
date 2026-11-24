# CLAUDE.md

Guía para Claude Code en este proyecto. Aquí solo van **reglas y hechos vigentes**, cortos. El detalle vive en `docs/` y la crónica en `CLAUDE_HISTORY.md`.

## Descripción
- Repo: https://github.com/gdelasca2810-max/ultimo-modulo (privado, rama `main`).
- Deploy frontend: Vercel (`vercel.json`, publica `public/`). Detalle en `docs/deploy/vercel.md`.
- Deploy backend: Render vía Blueprint `render.yaml` (estático, `public/`, autodeploy). URL: https://ultimo-modulo.onrender.com. Detalle en `docs/deploy/render.md`.
- Backend real: `api/main.py` (FastAPI + `modelo_sismos.joblib`), `/vigilar?key=CRON_SECRET` → Supabase + Telegram. Frontend en Vercel: https://ultimo-modulo.vercel.app.
- No commitear `.github/workflows/vigilancia_sismica.yml` (GitHub rechaza el push).
<!-- Rellenar: qué es el proyecto, stack, comandos esenciales. -->

## Mapa de documentos (leer bajo demanda con `@`)
| Tema | Archivo |
|---|---|
| Reglas de memoria y documentación | `docs/claude/memoria.md` |
| Uso de modelos y agentes | `docs/claude/modelos.md` |
| Arquitectura, usuarios, RAG | `docs/arquitectura.md` |
| Pendientes | `docs/todo.md` |
| Despliegue en Vercel | `docs/deploy/vercel.md` |
| Despliegue en Render | `docs/deploy/render.md` |
| Historia de sesiones | `CLAUDE_HISTORY.md` |

## Memoria y documentación (resumen; detalle en `docs/claude/memoria.md`)
- **Al terminar cada tarea (una sola vez, sin que lo pidan)** actualizar las capas que toque, sin duplicar texto:
  1. `CLAUDE.md`: solo el hecho técnico vigente (una línea).
  2. El doc de su tema en `docs/`.
  3. `CLAUDE_HISTORY.md`: qué se hizo en la sesión (fecha, 2-4 líneas).
  4. Memoria: `project_<tema>_<fecha>.md` + una línea en `MEMORY.md`.
- Memoria = un archivo por hecho, con frontmatter (`name`, `description`, `metadata.type`: user | feedback | project | reference). `MEMORY.md` es solo el índice: una línea por archivo, sin contenido.
- Antes de crear un recuerdo, buscar uno existente y actualizarlo. Borrar los que resulten falsos.
- Fechas siempre absolutas (`AAAA-MM-DD`), nunca «ayer» o «el jueves».
- «continúa» / «sigue»: reanudar SOLO la tarea interrumpida; primero listar qué falta y qué ya está hecho.
- Si `MEMORY.md` pasa de ~17 KB, compactarlo: mover detalle a los archivos de tema y unir entradas viejas.

## Modelos (resumen; detalle en `docs/claude/modelos.md`)
- La sesión corre en **Sonnet** (`.claude/settings.json`).
- **Opus** solo para lo de alto impacto: arquitectura, seguridad, pagos, features grandes, deuda crítica → primero un plan con el agente `Plan` y `model: "opus"`; luego se ejecuta con Sonnet.
- **Haiku** para lo mecánico de solo lectura (buscar, contar, listar, leer registros) → agente `haiku-lectura`.
- Sub-tareas cortas: hacerlas directo, sin agente. Nunca un fork para cambiar de modelo (el fork usa el modelo de la sesión).
