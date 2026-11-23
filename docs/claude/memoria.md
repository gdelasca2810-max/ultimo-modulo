# Cómo se guarda la memoria y la documentación

## Las 4 capas (cada dato en UNA sola, sin copiarlo en las otras)
| Capa | Qué va | Qué NO va |
|---|---|---|
| `CLAUDE.md` | Reglas y hechos vigentes, una línea cada uno, y el mapa de `docs/` | Historia, explicaciones largas, valores que cambian a menudo |
| `docs/<tema>.md` | El detalle de cada tema: cómo funciona, por qué, decisiones | Lo que ya está en el código o en git |
| `CLAUDE_HISTORY.md` | Crónica por sesión: fecha, qué se hizo, qué quedó pendiente | Reglas (van en CLAUDE.md) |
| Memoria (`~/.claude/projects/<proyecto>/memory/`) | Lo que hay que recordar entre conversaciones y no se deduce del repo | Estructura del código, historia de git, lo que ya dice CLAUDE.md |

## Formato de un recuerdo
Un archivo por hecho: `project_<tema>_<AAAA_MM_DD>.md`, `feedback_<tema>.md`, `user_<tema>.md` o `reference_<tema>.md`.

```markdown
---
name: project-tema-2026-12-01
description: una línea que sirva para decidir si es relevante
metadata:
  type: project        # user | feedback | project | reference
---

El hecho. En feedback/project añadir:
**Por qué:** …
**Cómo aplicarlo:** …
Relacionado: [[otro-recuerdo]]
```

- `user`: quién es el usuario y sus preferencias.
- `feedback`: correcciones y formas de trabajar que el usuario aprobó (con el porqué).
- `project`: trabajo en curso, objetivos, decisiones, con fechas absolutas.
- `reference`: enlaces a recursos externos.

## El índice `MEMORY.md`
- Se carga solo al empezar cada sesión: una línea por recuerdo, `- [Título](archivo.md) — gancho corto`.
- Arriba, las reglas obligatorias en una línea cada una (R1, R2…).
- Las sesiones recientes con «LEER PRIMERO» para retomar tras `/clear`.
- Mantenerlo por debajo de ~17 KB: lo viejo se resume y se mueve a los archivos de tema.

## Reglas
- Al terminar cada tarea: actualizar las 4 capas que toque, una vez, sin que lo pidan.
- Antes de escribir un recuerdo nuevo, buscar si ya existe uno y actualizarlo.
- Lo que un recuerdo dice de un archivo o función se comprueba antes de usarlo (puede estar desactualizado).
- Guardar el estado al terminar cada bloque de un trabajo largo, para poder seguir tras `/clear`.
