# Plan de producción del modelo de riesgo M6.5+ a 7 días

Fecha: 2026-11-25. Estado: **modelo v5 listo en el árbol de trabajo, sin commit ni push** (el modelo v2 anterior sigue en el repo remoto hasta que se haga el push).

## 1. Qué estaba mal y qué se arregló

| Problema | Arreglo |
|---|---|
| El modelo v2 usaba `media_96h_zona` y `p90_con_11a` (estadísticos de 2016–2026, **fuga temporal**) | v5 no las usa. Solo cuatro conteos sísmicos de las últimas 96 h y la tasa histórica de la zona |
| `/vigilar` calculaba las variables distinto al entrenamiento (M≥2,5, sin filtro de profundidad, sin réplicas, radio al punto de `ZONAS_COORDS`) | `api/features_t3.py` es la **única** implementación (M≥4,0, 5–40 km, centroide de la caja destino, cajas laterales, Gardner–Knopoff). La usan el entrenamiento (`scripts/reconstruir_t3_usgs.py`) y `/vigilar` |
| Una zona (Japón Nankai/Kyushu) tenía tasa 0,0 | La tasa de cada zona es una media suavizada hacia la global (peso 60) con todo el historial |

Verificado con datos reales: con solo lo que `/vigilar` descargaría, las 4 features coinciden con el dataset de entrenamiento en **2 000 de 2 000** valores (25 fechas × 20 zonas × 4 variables). El dataset regenerado tras el refactor es idéntico al anterior.

## 2. Modelo v5 (`api/modelo_sismos_v5.joblib`, `api/train_t3_multiplicativo.py`)

`logit(p) = logit(tasa_zona) + sesgo + Σ pesos de escalones activados` — la «tasa normal» de la zona multiplicada por un factor que sube cuando hay actividad local real:

- 3 o más sismos M4,5+ **sin réplicas** a ≤ 200 km en 96 h,
- 5 o más sismos M4,5+ sin réplicas,
- 5 o más sismos M4,0+ a ≤ 200 km.

**Dos ajustes distintos (importante):**

| | Pesos de los 3 escalones | Multiplicador máximo | Uso |
|---|---|---|---|
| Modelo **evaluado** (ajuste ≤ 2023-12-24) | 0,29 / 0,32 / 0,31 | ≈ 2,5× | produce las métricas de test de abajo |
| Modelo **desplegado** (reajuste con todo 2016–2026) | 0,56 / 0,48 / 0,57 | ≈ 4,9× | `api/modelo_sismos_v5.joblib` |

El reajuste incorpora los casos de cola de 2024–2026 (con ≥ 3 sismos locales, 8 positivos en 29 filas), así que los factores son mayores. **Las métricas de test corresponden al modelo evaluado, no al desplegado**: el desplegado no puede validarse fuera de muestra con estos datos. Con los pesos desplegados: ≥ 3 sismos M4,5+ sin réplicas por sí solos dan ≈ 1,7×, y ≥ 5 sismos M4,0+ por sí solos también ≈ 1,7×; **llegan a 3,1× cuando se cumplen ambas cosas (≥ 3 M4,5+ sin réplicas y ≥ 5 M4,0+ en total a ≤ 200 km)** y a 4,9× con ≥ 5 M4,5+ sin réplicas. Con los pesos evaluados solo se llegaba a 2× con los tres escalones a la vez. Si se prefiere la opción conservadora, basta copiar los pesos del modelo evaluado al artefacto (decisión de producto pendiente).

Los escalones se fijaron a priori a partir de la señal medida (tasa observada/esperada de 10–30× en la cola, en train y en test) y el único hiperparámetro (λ) se eligió con validación cruzada por años de 2016–2023. El test 2024+ no se usó para elegir nada.

Test 2024-01-08 en adelante (19 860 filas, 299 positivos), frente a la tasa histórica por zona:

| Métrica | v5 | Baseline | IC95 de la diferencia (bootstrap por meses) |
|---|---|---|---|
| AUC | 0,64641 | 0,63209 | [+0,0006, +0,0363] |
| AP | 0,02917 | 0,02812 | [+0,00003, +0,00256] |
| Brier | 0,01482 | 0,01484 | [−0,00004, 0,0000] |

La mejora es **real pero modesta**: la señal está en una cola rara (en test, ~30 filas con ≥ 3 sismos locales). Por eso el multiplicador de v5 es ≈ 1,0× casi siempre y solo sube con un enjambre local verificable.

## 3. Alertas (qué cambia respecto a antes)

- Regla de alerta: multiplicador ≥ umbral del usuario **y** «enjambre» = multiplicador ≥ 2, probabilidad ≥ 2,5 % y ≥ 3 sismos M4,5+ sin réplicas a ≤ 200 km.
- Con los pesos desplegados el umbral 2× se alcanza con **≥ 5 sismos M4,0+ a ≤ 200 km, de los cuales ≥ 3 sean M4,5+ sin réplicas** (3,1×; 4,9× si los M4,5+ sin réplicas son ≥ 5). Habrá **menos alertas que antes**: los «2×» del modelo v2 salían en buena parte de la fuga y del desajuste de variables. Una zona de tasa muy baja (p. ej. Nankai, 0,2 %) nunca alerta por un multiplicador artificial: 2× sigue siendo < 2,5 %.
- Los textos de la web y de Telegram ya describen las columnas con su definición real (M4,0+ someros; M4,5+ sin réplicas; profundidad intermedia en las zonas laterales).

## 4. Criterios obligatorios (cumplidos por v5) — `api/criterios_despliegue.py`

Brier no empeora ✔ · AUC y AP no empeoran ✔ · bootstrap por bloques sin deterioro significativo ✔ · features idénticas en entrenamiento y `/vigilar` ✔ (`tests/test_features_t3.py` + verificación con datos reales) · sin features prohibidas ni fuga ✔ · geometría documentada ✔ (`config/zonas_t3_historicas.json`).

**Limitación conocida (no bloquea):** las cajas de las zonas están marcadas «DEFINIDA POR CLAUDE - revisar» y **no están validadas por un sismólogo** (`config/zonas_t3_estado_validacion.json`). Presentar las zonas como regiones aproximadas.

## 5. Plan B: baseline por zona

`MODO_PREDICCION=baseline_zona` en Render → `/predict` devuelve la tasa suavizada (`api/baseline_tasa_zona.json`, regenerable con `scripts/generar_baseline_zona.py`) con multiplicador 1,0 y sin enjambre: ninguna alerta Telegram. Quitar la variable lo desactiva. Úsalo si un día se duda de v5. `MODELO_ARTEFACTO` permite elegir otro archivo de modelo (el v2 antiguo **no** es compatible con las features T3 en vivo).

## 6. Activación y orden de despliegue

1. Revisar `git diff` (incluye cambios previos del usuario sin commit en `api/main.py`).
2. Commit **sin** `.github/workflows/vigilancia_sismica.yml`. Hay que subir `api/modelo_sismos_v5.joblib`, `api/features_t3.py`, `config/zonas_t3_historicas.json`, `api/baseline_tasa_zona.json` y `public/index.html`.
3. Render autodesplegará la API; después `npx vercel --prod` para la web.
4. Comprobar `/health` (versión `v5_t3_multiplicativo`) y lanzar `/vigilar` una vez. Cada ejecución descarga unos 1,5 MB de USGS y tarda ~5 s más que antes.
5. No cambia el contrato JSON de `/health`, `/predict` ni `/vigilar`, ni las tablas de Supabase. **Cambia el significado** de `loc200_n96_con/sin45`, `loc500_n96_con` y `lat_n96_prof70_150` en `zonas_sismicas`: los valores antiguos (M≥2,5) se sobrescriben en la primera ejecución.

## 7. Pendientes que no cubre esto

- `agente/vigilante.py` (cron de GitHub Actions, no usado) y `mcp/server.py` siguen con las definiciones antiguas.
- Sumatra/Andaman sigue sin soporte (sin caja ni datos).
- Validación sismológica de las cajas.
- Nueva fecha de reentrenamiento: `python scripts/reconstruir_t3_usgs.py` + `python api/train_t3_multiplicativo.py`.

## 8. Comandos

```text
python -m unittest discover -s tests -v          # 31 pruebas (FastAPI compatible con starlette)
python scripts/reconstruir_t3_usgs.py             # regenera el dataset (sin red, usa data/raw)
python api/train_t3_multiplicativo.py             # evalúa, aplica criterios y escribe api/modelo_sismos_v5.joblib
python scripts/generar_baseline_zona.py           # regenera el baseline
```
