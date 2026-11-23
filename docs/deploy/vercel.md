# Despliegue en Vercel

- Reparto: **Vercel = frontend** (páginas web, siempre encendido). **Render = backend** con servidor cuando exista (ver `render.md`); mientras tanto Render sirve también el sitio estático.
- Cuenta/equipo: https://vercel.com/gdelasca2810-6370s-projects
- Config: `vercel.json` (sin build, publica `public/`). Cada push a `main` despliega a producción; otras ramas generan previews.
- Alta (una vez): Vercel → Add New → Project → Import Git Repository → conectar GitHub gdelasca2810-max → `ultimo-modulo` → Deploy (sin tocar ajustes; los toma de `vercel.json`).
- Cuando el frontend tenga framework (Next, Vite…), quitar `framework`, `buildCommand` y `outputDirectory` de `vercel.json` para que Vercel los detecte solo.
