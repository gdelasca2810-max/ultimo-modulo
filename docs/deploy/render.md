# Despliegue en Render

- Blueprint: `render.yaml` (sitio estático, publica `public/`, autodeploy en cada push a `main`).
- Repo: https://github.com/gdelasca2810-max/ultimo-modulo (privado).
- URL pública: https://ultimo-modulo.onrender.com (Live desde 2026-11-23).
- Alta (una vez): Render Dashboard → New → Blueprint → conectar la cuenta de GitHub gdelasca2810-max → elegir `ultimo-modulo` → Apply.
- Cuando exista una app real (Node, Python…), cambiar `runtime: static` por `runtime: node|python` con `buildCommand` y `startCommand`.

## Backend de alertas (actual)
- `api/main.py` (FastAPI, modelo `modelo_sismos.joblib` v2_features_11a) corre en Render; `/vigilar` exige `?key=<CRON_SECRET>` (el valor está en las variables de Render, no en el repo).
- `/vigilar`: lee USGS (últimas 96 h), actualiza `zonas_sismicas`, evalúa `alertas_usuario` (Supabase) y avisa por Telegram en lotes de 4 (máx. 4000 chars, respaldo en texto plano si falla el HTML).
- Git: no incluir `.github/workflows/vigilancia_sismica.yml` en commits a `origin/main` (GitHub rechaza el push); usar `git stash push --include-untracked` si hay otros cambios sin commitear.
