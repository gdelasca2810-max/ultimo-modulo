# Plan de producción del modelo de riesgo M6.5+ a 7 días

Fecha: 2026-11-25. Modelo activo: **v6 (M4,0)**, `api/modelo_sismos_v6.joblib`. Decisión del usuario (2026-11-25): **no existe umbral M4,5**; todas las variables usan M≥4,0.

## 1. Qué estaba mal y qué se arregló

| Problema | Arreglo |
|---|---|
| El modelo v2 usaba `media_96h_zona` y `p90_con_11a` (estadísticos de 2016–2026: **fuga temporal**) | v6 no las usa: solo conteos sísmicos de las últimas 96 h y la tasa histórica de la zona |
| `/vigilar` calculaba las variables distinto al entrenamiento (M≥2,5, sin filtro de profundidad, sin réplicas, radio al punto de `ZONAS_COORDS`) | `api/features_t3.py` es la **única** implementación (M≥4,0, 5–40 km, centroide de la caja destino, cajas laterales, Gardner–Knopoff). La usan el entrenamiento y `/vigilar` |
| Una zona (Japón Nankai/Kyushu) tenía tasa 0,0 | Tasa suavizada hacia la global (peso 60) con todo el historial |

## 2. Definición de las variables (nombres de columna conservados por compatibilidad)

Ventana `[t−96 h, t)`, sismos M≥4,0 de 5–40 km de profundidad salvo indicación:
- `loc200_n96_con`: a ≤ 200 km del centroide de la caja destino (incluye réplicas).
- `loc200_n96_sin45`: igual pero **sin réplicas** (Gardner–Knopoff). El nombre es histórico: ya no hay umbral M4,5.
- `loc500_n96_con`: a ≤ 500 km.
- `lat_n96_prof70_150`: M≥4,0 a 70–150 km de profundidad en las cajas laterales.
- Target `y_M65_7d`: M≥6,5 sin réplicas en la caja destino en `[t, t+7 d)` (sin cambios).

Nota: `loc200_n96_sin45` y `lat_n96_prof70_150` ya **no** coinciden con el CSV histórico original (que usaba M4,5): 96 % y 52 % de igualdad. El generador histórico se conserva como referencia de `loc200_n96_con`, `loc500_n96_con` y los targets (≥ 99,5 %).

Aún existen M4,5 fuera del cálculo de riesgo: el mapa de «sismos fuertes de la semana» (feed USGS 4.5_week) y la búsqueda histórica del bot (base M4,5+ de Supabase). No intervienen en las probabilidades.

## 3. Modelo v6 (`api/train_t3_multiplicativo.py`)

`logit(p) = logit(tasa_zona) + sesgo + Σ pesos de los escalones activados`: la «tasa normal» de la zona multiplicada por un factor que sube con la actividad local real.

Escalones (umbrales fijados mirando solo el entrenamiento ≤ 2023: razón observada/esperada 1,9× con 3–4, 6,8× con 5–7 y 38× con ≥ 8 sismos sin réplicas): `sin45 ≥ 3`, `sin45 ≥ 5`, `sin45 ≥ 8` y `con ≥ 8`. Regularización λ = 5 elegida por validación cruzada por años de 2016–2023.

**Dos ajustes distintos:**

| | Pesos (3 / 5 / 8 sin réplicas, 8 total) | Uso |
|---|---|---|
| **Evaluado** (ajuste ≤ 2023-12-24) | 0,66 / 0,84 / 0,84 / 0,77 | produce las métricas de abajo |
| **Desplegado** (reajuste con 2016–2026) | 0,97 / 0,87 / 0,83 / 0,74 | `api/modelo_sismos_v6.joblib` |

Multiplicadores desplegados (ejemplos): 3 sismos sin réplicas → **2,6×**; 5 → 6,2×; 5 sin réplicas y 8 en total → 13×; 8 sin réplicas → 30×. **Las métricas de test corresponden al modelo evaluado, no al desplegado** (este último no puede validarse fuera de muestra con estos datos).

Test 2024-01-08 en adelante (19 860 filas, 299 positivos) frente a la tasa histórica por zona:

| Métrica | v6 evaluado | Baseline | IC95 de la diferencia (bootstrap por meses) |
|---|---|---|---|
| AUC | 0,64591 | 0,63209 | [−0,0013, +0,0384] |
| AP | 0,03315 | 0,02812 | [−0,0002, +0,0283] |
| Brier | 0,01480 | 0,01484 | [−0,00013, +0,00002] |

**Lectura honesta:** los tres números mejoran, pero ningún intervalo excluye el cero: **no se demuestra una mejora estadística, y tampoco hay deterioro** (los criterios exigen lo segundo). La señal está en muy pocos episodios.

### Qué muestran las fechas (≥ 5 sismos M4,0+ sin réplicas a ≤ 200 km en 96 h)
10 episodios en 10,7 años y 20 zonas; 2 fueron seguidos de un M6,5+ en la caja en los 7 días siguientes (20 %, frente a ~2–3 % de una semana cualquiera): **Sandwich del Sur (2021-08-13, 298 sismos totales tras el M7,5/M8,1)** y **Kamchatka (2025-07-31, 63 totales, tras el M8,8)**. Son secuelas de terremotos gigantes, no enjambres previos a un terremoto sin aviso. Los otros 8 episodios (Banda/Timor 2016, Japón NE 2016 y 2025, Sandwich 2022, Taiwán 2022, Mindanao 2026 ×2, México 2026) no tuvieron M6,5+. n = 10 es muy poco: tratar los multiplicadores altos (≥ 6×) como orientativos.

## 4. Alertas

- Regla: multiplicador ≥ umbral del usuario **y** «enjambre» = multiplicador ≥ 2, probabilidad ≥ 2,5 % y ≥ 3 sismos M4,0+ sin réplicas a ≤ 200 km.
- Con los pesos desplegados el 2× se alcanza con **3 sismos M4,0+ sin réplicas a ≤ 200 km en 96 h** (2,6×). Para alertar, la probabilidad debe además llegar a 2,5 %: una zona de tasa muy baja (p. ej. Nankai, 0,2 %) no alerta por un multiplicador artificial.
- Habrá **menos alertas que con el modelo antiguo** (gran parte de sus «2×» eran fuga/desajuste).

## 5. Criterios obligatorios (cumplidos) — `api/criterios_despliegue.py`

Brier no empeora ✔ · AUC y AP no empeoran ✔ · bootstrap por bloques sin deterioro significativo ✔ · features idénticas en entrenamiento y `/vigilar` ✔ · sin fuga ni features prohibidas ✔ · geometría documentada ✔.

Identidad producción/entrenamiento medida con datos reales: **2 000 de 2 000** valores (25 fechas × 20 zonas × 4 variables) usando solo lo que `/vigilar` descarga. Con 1 100 días de historia hubo un caso distinto: las réplicas forman cadenas (M4,4 ← M4,9 ← M5,8 ← mayor anterior), así que la historia de mainshocks se amplió a **2 600 días** (descarga por tramos de 800 días por el límite de 20 000 eventos de USGS).

**Limitación conocida (no bloquea):** las cajas de las zonas están marcadas «DEFINIDA POR CLAUDE - revisar» y **no están validadas por un sismólogo** (`config/zonas_t3_estado_validacion.json`).

## 6. Rendimiento de `/vigilar`

Descarga ~23 000 eventos M≥4,8 de historia (en tramos) + los M≥4,0 de 96 h. Una réplica solo depende de sismos anteriores, así que las marcas de réplica de la historia se calculan **una vez** y se guardan en memoria 24 h; en cada ejecución solo se descargan y marcan los eventos nuevos (`replicas_incremental`, idéntico al declustering completo: `tests/test_features_t3.py` y comprobación con 23 048 eventos reales de USGS). Medido en local: primera ejecución ~13 s, siguientes ~2 s. En Render la primera tras un reinicio/arranque en frío es más lenta (medido 28 s con la versión anterior sin marcas en caché); Render gratuito duerme tras 15 min sin tráfico y pierde la caché, así que el primer `/vigilar` del día puede rozar el límite de ~30 s de cron-job.org (el servidor sigue procesando aunque el cron corte).

## 7. Plan B: baseline por zona

`MODO_PREDICCION=baseline_zona` en Render → `/predict` devuelve la tasa suavizada (`api/baseline_tasa_zona.json`) con multiplicador 1,0 y sin enjambre: ninguna alerta Telegram. Quitar la variable lo desactiva. `MODELO_ARTEFACTO` permite elegir otro archivo de modelo (el v2 antiguo **no** es compatible con las features T3 en vivo).

## 8. Despliegue

Cambia el **significado** (no los nombres) de `loc200_n96_con/sin45`, `loc500_n96_con` y `lat_n96_prof70_150` en `zonas_sismicas`: la primera ejecución de `/vigilar` sobrescribe los valores antiguos. No cambian los contratos JSON de `/health`, `/predict`, `/vigilar` ni las tablas.

## 9. Pendientes

- `agente/vigilante.py` (cron de GitHub Actions, no usado) y `mcp/server.py` siguen con las definiciones antiguas.
- Sumatra/Andaman sigue sin soporte (sin caja ni datos).
- Validación sismológica de las cajas.
- Si se desea, limitar el multiplicador máximo (p. ej. 10×) por cautela; hoy llega a 30×.

## 10. Comandos

```text
python -m unittest discover -s tests -v          # 31 pruebas (FastAPI compatible con starlette)
python scripts/reconstruir_t3_usgs.py             # regenera el dataset (sin red, usa data/raw)
python api/train_t3_multiplicativo.py             # evalúa, aplica criterios y escribe api/modelo_sismos_v6.joblib
python scripts/generar_baseline_zona.py           # regenera el baseline
```
