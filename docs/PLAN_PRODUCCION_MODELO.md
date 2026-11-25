# Plan de producción del modelo de riesgo M6.5+ a 7 días

Fecha: 2026-11-25. Modelo activo: **v7 (M4,0, radio local 300 km)**, `api/modelo_sismos_v7.joblib`. Decisión del usuario (2026-11-25): **no existe umbral M4,5**; todas las variables usan M≥4,0.

## 1. Qué estaba mal y qué se arregló

| Problema | Arreglo |
|---|---|
| El modelo v2 usaba `media_96h_zona` y `p90_con_11a` (estadísticos de 2016–2026: **fuga temporal**) | v6 no las usa: solo conteos sísmicos de las últimas 96 h y la tasa histórica de la zona |
| `/vigilar` calculaba las variables distinto al entrenamiento (M≥2,5, sin filtro de profundidad, sin réplicas, radio al punto de `ZONAS_COORDS`) | `api/features_t3.py` es la **única** implementación (M≥4,0, 5–40 km, centroide de la caja destino, cajas laterales, Gardner–Knopoff). La usan el entrenamiento y `/vigilar` |
| Una zona (Japón Nankai/Kyushu) tenía tasa 0,0 | Tasa suavizada hacia la global (peso 60) con todo el historial |

## 2. Definición de las variables (nombres de columna conservados por compatibilidad)

Ventana `[t−96 h, t)`, sismos M≥4,0 de 5–40 km de profundidad salvo indicación. **Radio local = 300 km** (`RADIO_LOCAL_KM` en `api/features_t3.py`; antes 200 km):
- `loc200_n96_con`: a ≤ **300 km** del centroide de la caja destino (incluye réplicas). *El nombre `loc200` es histórico: se conserva para no romper Supabase ni la API.*
- `loc200_n96_sin45`: igual pero **sin réplicas** (Gardner–Knopoff). Tampoco hay umbral M4,5.
- `loc500_n96_con`: a ≤ 500 km.
- `lat_n96_prof70_150`: M≥4,0 a 70–150 km de profundidad en las cajas laterales.
- Target `y_M65_7d`: M≥6,5 sin réplicas en la caja destino en `[t, t+7 d)` (sin cambios).

`loc200_n96_con`, `loc200_n96_sin45` y `lat_n96_prof70_150` ya **no** coinciden con el CSV histórico original (84 %, 88 % y 52 %: ahora miden 300 km / M4,0). Siguen coincidiendo `loc500_n96_con` y los targets (≥ 99,5 %).

Aún existen M4,5 fuera del cálculo de riesgo: el mapa de «sismos fuertes de la semana» (feed USGS 4.5_week) y la búsqueda histórica del bot (base M4,5+ de Supabase). No intervienen en las probabilidades.

### Por qué 300 km y 5–40 km (exploración 2026-11-25; selección por validación cruzada 2016–2023, no por el test)

Razón observada/esperada de un M6,5+ en la semana siguiente según la profundidad de los sismos M4,0+ sin réplicas a ≤ 200 km (entrenamiento ≤ 2023 / test 2024+):
- **5–40 km (somero): ≥ 3 sismos → ×4,8 / ×8,3. Es la señal que se repite en las dos épocas.**
- 40–70 km: ≥ 2 → ×2,2 / ×2,3 (débil, consistente).
- 70–150 km: ≥ 3 → ×0 / ×12,8 (no se repite; un episodio). 150–300 km y > 300 km: sin señal.
- Todas las profundidades juntas: ×1,5 / ×2,5 (más débil que solo el somero) → **no quitar el filtro de profundidad**.

Variantes de radio y profundidad (mismos escalones, λ = 5):

| Variante | Mejora en CV (log-loss ×1e5) | AUC test (IC95 vs baseline) | AP test (IC95) |
|---|---|---|---|
| 200 km, 5–40 | +5,1 | +0,014 [−0,001, +0,036] | +0,005 [−0,000, +0,028] |
| 200 km, todas | +3,9 | +0,010 [−0,002, +0,028] | +0,015 [+0,000, +0,045] |
| **300 km, 5–40** | **+5,9 (mejor)** | +0,016 [+0,002, +0,035] | +0,007 [+0,000, +0,025] |
| 300 km, todas | −0,3 | +0,014 [+0,001, +0,031] | +0,007 [+0,000, +0,025] |
| 500 km, 5–40 | +4,0 | +0,031 [+0,008, +0,055] | +0,005 [+0,001, +0,013] |
| 500 km, todas | +5,6 | +0,024 [+0,002, +0,044] | +0,002 [−0,000, +0,004] |

Se probaron 7 variantes: los IC no están corregidos por comparaciones múltiples; 300 y 500 km no son distinguibles. Se eligió 300 km 5–40 km por ser la mejor en CV.

## 3. Modelo v7 (`api/train_t3_multiplicativo.py`)

`logit(p) = logit(tasa_zona) + sesgo + Σ pesos de los escalones activados`: la «tasa normal» de la zona multiplicada por un factor que sube con la actividad local real.

Escalones (umbrales fijados con datos de entrenamiento ≤ 2023): `sin45 ≥ 3`, `sin45 ≥ 5`, `sin45 ≥ 8` y `con ≥ 8`. λ = 5 elegida por validación cruzada por años de 2016–2023.

**Dos ajustes distintos:**

| | Pesos (3 / 5 / 8 sin réplicas, 8 total) | Uso |
|---|---|---|
| **Evaluado** (ajuste ≤ 2023-12-24) | 0,06 / 0,63 / 0,93 / 0,61 | produce las métricas de abajo |
| **Desplegado** (reajuste con 2016–2026) | 0,42 / 0,71 / 1,11 / 0,46 | `api/modelo_sismos_v7.joblib` |

Multiplicadores desplegados: 3 sismos sin réplicas → **1,5×**; 3 sin réplicas y ≥ 8 en total → **2,4×**; 5 sin réplicas → **3,1×**; 5 y ≥ 8 totales → 4,9×; 8 sin réplicas → 14,8×. **Las métricas de test corresponden al modelo evaluado, no al desplegado.**

Test 2024-01-08 en adelante (19 860 filas, 299 positivos) frente a la tasa histórica por zona:

| Métrica | v7 evaluado | Baseline | IC95 de la diferencia (bootstrap por meses) |
|---|---|---|---|
| AUC | 0,64811 | 0,63209 | **[+0,0015, +0,0364]** |
| AP | 0,03499 | 0,02812 | **[+0,0001, +0,0256]** |
| Brier | 0,01480 | 0,01484 | [−0,00013, +0,00002] |

**Lectura honesta:** AUC y AP mejoran con intervalos que excluyen el cero; el Brier no empeora (mejora sin ser significativa). La señal sigue concentrada en muy pocos episodios.

### Qué muestran las fechas (≥ 5 sismos M4,0+ sin réplicas a ≤ 200 km en 96 h, análisis previo)
10 episodios en 10,7 años; 2 seguidos de un M6,5+ en la caja en los 7 días siguientes: Sandwich del Sur (2021-08-13, tras el M7,5/M8,1) y Kamchatka (2025-07-31, tras el M8,8). Son secuelas de terremotos gigantes. n = 10: los multiplicadores altos (≥ 5×) son orientativos.

## 4. Alertas

- Regla: multiplicador ≥ umbral del usuario **y** «enjambre» = multiplicador ≥ 2, probabilidad ≥ 2,5 % y ≥ 3 sismos M4,0+ sin réplicas a ≤ 300 km.
- Con los pesos desplegados el 2× se alcanza con **≥ 5 sismos sin réplicas a ≤ 300 km** (3,1×) o con **3 sin réplicas y ≥ 8 en total** (2,4×). Solo 3 sin réplicas da 1,5× y no alerta.
- Para alertar la probabilidad debe llegar además a 2,5 %: una zona de tasa muy baja (p. ej. Nankai, 0,2 %) no alerta por un multiplicador artificial.

## 5. Criterios obligatorios (cumplidos) — `api/criterios_despliegue.py`

Brier no empeora ✔ · AUC y AP no empeoran ✔ · bootstrap por bloques sin deterioro significativo ✔ · features idénticas en entrenamiento y `/vigilar` ✔ · sin fuga ni features prohibidas ✔ · geometría documentada ✔.

Identidad producción/entrenamiento medida con datos reales: **2 000 de 2 000** valores (25 fechas × 20 zonas × 4 variables) usando solo lo que `/vigilar` descarga. Con 1 100 días de historia hubo un caso distinto: las réplicas forman cadenas (M4,4 ← M4,9 ← M5,8 ← mayor anterior), así que la historia de mainshocks se amplió a **2 600 días** (descarga por tramos de 800 días por el límite de 20 000 eventos de USGS).

**Limitación conocida (no bloquea):** las cajas de las zonas están marcadas «DEFINIDA POR CLAUDE - revisar» y **no están validadas por un sismólogo** (`config/zonas_t3_estado_validacion.json`).

## 6. Rendimiento de `/vigilar`

Descarga ~23 000 eventos M≥4,8 de historia (en tramos) + los M≥4,0 de 96 h. Una réplica solo depende de sismos anteriores, así que las marcas de réplica de la historia se calculan **una vez** y se guardan en memoria 24 h; en cada ejecución solo se descargan y marcan los eventos nuevos (`replicas_incremental`, idéntico al declustering completo: `tests/test_features_t3.py` y comprobación con 23 048 eventos reales de USGS). Medido en local: primera ejecución ~13 s, siguientes ~2 s. En Render la primera tras un reinicio/arranque en frío es más lenta (medido 28 s con la versión anterior sin marcas en caché); Render gratuito duerme tras 15 min sin tráfico y pierde la caché, así que el primer `/vigilar` del día puede rozar el límite de ~30 s de cron-job.org (el servidor sigue procesando aunque el cron corte).

## 7. Plan B: baseline por zona

`MODO_PREDICCION=baseline_zona` en Render → `/predict` devuelve la tasa suavizada (`api/baseline_tasa_zona.json`) con multiplicador 1,0 y sin enjambre: ninguna alerta Telegram. Quitar la variable lo desactiva. `MODELO_ARTEFACTO` permite elegir otro archivo de modelo (el v2 antiguo **no** es compatible con las features T3 en vivo).

## 8. Despliegue

Cambia el **significado** (no los nombres) de `loc200_n96_con/sin45` (ahora ≤ 300 km), `loc500_n96_con` y `lat_n96_prof70_150` en `zonas_sismicas`: la primera ejecución de `/vigilar` sobrescribe los valores antiguos. No cambian los contratos JSON de `/health`, `/predict`, `/vigilar` ni las tablas.

## 9. Pendientes

- `agente/vigilante.py` (cron de GitHub Actions, no usado) y `mcp/server.py` siguen con las definiciones antiguas.
- Sumatra/Andaman sigue sin soporte (sin caja ni datos).
- Validación sismológica de las cajas.
- Si se desea, limitar el multiplicador máximo (p. ej. 10×) por cautela; hoy llega a 30×.

## 10. Comandos

```text
python -m unittest discover -s tests -v          # 31 pruebas (FastAPI compatible con starlette)
python scripts/reconstruir_t3_usgs.py             # regenera el dataset (sin red, usa data/raw)
python api/train_t3_multiplicativo.py             # evalúa, aplica criterios y escribe api/modelo_sismos_v7.joblib
python scripts/generar_baseline_zona.py           # regenera el baseline
```
