# Despliegue en Render

- Blueprint: `render.yaml` (sitio estático, publica `public/`, autodeploy en cada push a `main`).
- Repo: https://github.com/gdelasca2810-max/ultimo-modulo (privado).
- Alta (una vez): Render Dashboard → New → Blueprint → conectar la cuenta de GitHub gdelasca2810-max → elegir `ultimo-modulo` → Apply.
- Cuando exista una app real (Node, Python…), cambiar `runtime: static` por `runtime: node|python` con `buildCommand` y `startCommand`.
