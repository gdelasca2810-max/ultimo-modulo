# Pendientes y estado

## Hecho (2026-11-24)
- [x] RLS y Telegram por usuario (`db/seguridad_y_telegram.sql`).
- [x] RAG con pgvector funcionando (`db/rag_pgvector.sql`, `/rag/ingestar`).
- [x] Historial de 90 763 sismos M4.5+ desde 2015 y consultas del bot con fuente indicada (`db/sismos_historicos.sql`).
- [x] Bot con OpenAI (`gpt-6-luna`) y respaldo en Gemini.
- [x] Vercel ya no compila `api/` (`.vercelignore`).
- [x] Deck de 14 diapositivas rellenado (nombre, fecha, métricas, RAG, historial).
- [x] Web rediseñada: mapa en vivo, top 3, tabla ordenable y paginada, lenguaje simple.
- [x] Documentación del modelo (`docs/modelo_ml.md`).

## Para que la web muestre el riesgo de hoy (en curso)
- [ ] Ejecutar `db/zonas_riesgo_hoy.sql` en Supabase.
- [ ] Esperar el despliegue de Render y lanzar `/vigilar` (llena multiplicador y probabilidad).
- [ ] Desplegar la web: `npx vercel --prod`.

## Seguridad
- [ ] Regenerar la clave de OpenAI (se pegó en un chat) y poner límite de gasto mensual.
- [ ] Revisar que no haya claves en el historial de conversación o en el repo.

## Producto
- [ ] Probar la web con sesión iniciada en móvil (bloque de alertas, Telegram, pasos).
- [ ] Alertas de prueba a 0.5x: dejar solo las que se quieran mantener.
- [ ] Fichas de terremotos famosos para el RAG (Tohoku 2011, Chile 2010, Nepal 2015, Turquía 2023) en `rag/conocimiento/`.
- [ ] Comprobar que el cron horario corre (la web mostró datos de hace 26 h). El 2026-11-25 el job de las 16 h falló por respuesta demasiado grande (posible arranque en frío de Render): añadir job de calentamiento a las 15:50 contra `/docs`, activar reintentos y definir `CRON_SECRET` en Render.
- [ ] Regenerar `data/zona_dia_11a.csv` o documentar cómo obtenerlo para poder reentrenar.

## Presentación
- [ ] Revisar visualmente las diapositivas nuevas (RAG, historial, 6 y 7 reescritas).
- [ ] Ensayar la demo con Render despierto (abrir `/health` 2 minutos antes).
- [ ] Actualizar capturas de la web nueva.

## Decisiones pendientes
- [ ] Qué commitear: `.vscode/`, `AGENTS.MD`. El workflow `vigilancia_sismica.yml` NO.
