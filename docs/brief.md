# Brief del proyecto

## Problema
Los terremotos fuertes son raros y los datos sísmicos (catálogos, magnitudes, profundidades) son difíciles de interpretar para el público general. Falta una forma simple de saber si una región está hoy más activa de lo normal.

## Qué hacemos
Un monitor que lee cada hora los temblores reales del USGS, usa un modelo de machine learning para estimar si el riesgo de esta semana está por encima de lo habitual en 20 regiones del mundo, y lo explica en lenguaje simple por web y Telegram. **No predice terremotos**: mide riesgo relativo.

## Usuarios
- **Público general** curioso o que vive en zonas sísmicas: quiere entender sin saber de sismología.
- **Usuarios con cuenta:** eligen regiones y sensibilidad y reciben avisos en su Telegram.
- **Quien consulta al bot:** pregunta por una zona o por un sismo pasado y recibe la fuente.
- **Evaluadores (curso):** ven el sistema completo: datos, ML, backend, RAG y despliegue.

## Alcance (lo que SÍ está)
- Modelo ML calibrado y validado con partición temporal (`docs/modelo_ml.md`).
- API (Render), web (Vercel), base de datos y usuarios con RLS (Supabase).
- Alertas por usuario en Telegram, bot conversacional con OpenAI/Gemini.
- RAG con pgvector y historial de 90 763 sismos M4.5+ desde 2015, con la fuente siempre indicada.

## Fuera de alcance
- Predecir cuándo o dónde ocurrirá un terremoto.
- Alertas de emergencia oficiales (siempre remitir a protección civil).
- Sismos menores a M4.5 en el historial; regiones fuera de las 20 vigiladas.

## Éxito
- El usuario entiende el mensaje sin conocimientos técnicos.
- Probabilidades honestas (calibradas) y límites siempre explicados.
- Sistema desplegado y funcionando de punta a punta para la demo.

Relacionado: `docs/arquitectura.md` (las piezas), `docs/modelo_ml.md`, `docs/todo.md` (pendientes).
