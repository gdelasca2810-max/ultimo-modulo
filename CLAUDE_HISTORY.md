# Historia de sesiones

Una entrada por sesión, la más reciente arriba. Formato:

## AAAA-MM-DD — tema
- Qué se hizo (2-4 líneas).
- Qué quedó pendiente.

## 2026-11-24 — Mensaje Telegram, estructura y docs
- `api/main.py`: `construir_mensaje_alerta` (lenguaje simple, plurales), caso sin sismos <500 km = AVISO POR MODELO, `<` escapado. CSV movido a `data/`. Nuevos `docs/arquitectura.md` (usuarios/RLS/RAG) y `docs/todo.md`. Pendientes ahora en `docs/todo.md`.

## 2026-11-24 — Alertas Telegram y umbral 0.5x
- Frontend: umbral editable (`id="umbral"`, min 0.1, `datalist` con 0.5x para pruebas); 20 alertas de prueba a 0.5x en `alertas_usuario`.
- Backend `api/main.py`: `formatear_frecuencia_habitual`, `describir_nivel_usuario` (estados coherentes + `resumen_simple`), envío a Telegram por lotes (4, máx. 4000 chars) con respaldo en texto plano ante 400. `/vigilar` desplegado en Render sin 500/400.
- Pendiente: caso `loc200_con==0` y `loc500_con==0` con multiplicador >=1.5 (p. ej. Filipinas Luzon) no debe salir MODERADO; plural «1 sismo cercano»; subir alertas web de 0.5x a 1.5x/2.0x tras las pruebas.

## 2026-11-24 — Presentación
- Fases técnicas dadas por terminadas (ML calibrado, API FastAPI en Render, Supabase, web en Vercel, MCP). Se creó el deck de 12 diapositivas (Artifact https://claude.ai/artifact/AEEfteDMUSCFABaJu7QxRB) con notas del orador.
- Pendiente: completar `[Tu nombre]`, `[Fecha]`, métricas (ROC-AUC/PR-AUC/Brier desde `/health`) y zona de ejemplo; revisar el deck; ensayar la demo; decidir qué commitear (workflow, `.gitignore`, `.vscode/`, `AGENTS.MD`).

## 2026-11-23 — Subida a GitHub
- Se inicializó git y se creó el repo privado gdelasca2810-max/ultimo-modulo; commit inicial con la estructura (CLAUDE.md, docs/, .claude/).
- Pendiente: nada.
- Corrección: el repo se movió a la cuenta gdelasca2810-max (el usuario no quería Nutriberry; borra él el repo viejo).

## 2026-11-23 — Render
- Se añadió `render.yaml` (sitio estático de arranque en `public/`) y `docs/deploy/render.md`.
- Blueprint conectado por el usuario el 2026-11-23. Live en https://ultimo-modulo.onrender.com (HTTP 200 verificado).

## 2026-11-23 — Vercel
- Se añadió `vercel.json` y `docs/deploy/vercel.md`. Reparto: Vercel = frontend, Render = backend (de momento también sirve el estático).
- Pendiente: el usuario importa el repo en Vercel y pasa la URL.
