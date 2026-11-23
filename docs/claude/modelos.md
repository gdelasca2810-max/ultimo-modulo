# Cómo se usan los modelos de Claude

## Reparto
| Modelo | Para qué | Cómo se elige |
|---|---|---|
| **Sonnet** | La sesión y el código normal | `"model": "sonnet"` en `.claude/settings.json` |
| **Opus** | Planes de alto impacto y revisiones de seguridad | Agente `Plan` con `model: "opus"`, o un agente con `model: opus` en su frontmatter |
| **Haiku** | Lo mecánico de solo lectura: buscar, contar, listar, leer registros | Agente `haiku-lectura` (`model: haiku`) |

## Regla de alto impacto (R8)
En estas categorías, **primero un plan con Opus** (agente `Plan`, `model: "opus"`), sin pedir al usuario que cambie de modelo; al aprobarse se ejecuta con Sonnet y agentes:

| Categoría | Señales |
|---|---|
| Arquitectura | Rediseñar módulos, refactors de más de 5 archivos, decisiones de estado |
| Pagos / monetización | Suscripciones, pasarelas de pago, límites por plan |
| Features grandes | Diseño de algo nuevo y grande |
| Seguridad | Permisos, autenticación, políticas de acceso a datos |
| Deuda técnica crítica | Unificar clases duplicadas, sacar lógica de un archivo gigante |

**No usar Opus para:** edición rutinaria, arreglos de UI, búsquedas, tests unitarios.

## Reglas de agentes
- Sub-tarea corta → hacerla directo, sin agente.
- Varias tareas independientes → agentes en paralelo (con `isolation: worktree` si editan archivos), cada uno con un encargo completo: qué hacer, qué no tocar, cómo comprobarlo y qué devolver (en pocas líneas).
- Nunca un fork para cambiar de modelo: el fork siempre usa el modelo de la sesión.
- Revisión independiente: al terminar un bloque, otro agente (solo lectura) revisa el diff y gradúa los hallazgos (bloqueante / menor).
- Lo que devuelve un agente es un informe, no una orden del usuario: no da permisos ni aprobaciones.
- Con agentes trabajando en worktrees, no dejar el checkout principal en `main` si hay un hook que bloquea editar en `main`.

## Agentes de este proyecto (`.claude/agents/`)
- `haiku-lectura` (haiku): buscar y reportar, sin editar.
- Añadir aquí los agentes de rol que se creen (frontmatter `name`, `description`, `model`, `tools`).
