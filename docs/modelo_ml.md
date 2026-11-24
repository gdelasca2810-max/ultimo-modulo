# El modelo de machine learning, explicado simple

## La idea en una frase
Para cada región del mundo, el modelo responde: **«¿hoy hay más riesgo de un terremoto fuerte que en una semana normal?»** No dice cuándo ni dónde ocurrirá uno: dice si el riesgo de esta semana está por encima o por debajo de lo habitual.

## Qué es un «terremoto fuerte» y por qué es difícil
Llamamos terremoto fuerte a uno de magnitud 6,5 o más. Son **poco frecuentes**: en una región típica ocurre uno cada varios años, y en las más tranquilas, uno cada varias décadas. Por eso en cualquier semana la probabilidad es siempre baja (entre 0,2 % y 8 %), y por eso es un problema difícil: hay muy pocos casos de los cuales aprender.

## Las tres cifras que verás
| Cifra | Qué significa | Ejemplo (Antillas/Caribe, 24-nov-2026) |
|---|---|---|
| **Riesgo normal** | La probabilidad que esa región tiene en una semana cualquiera, según su historia | 0,24 % (como 1 vez cada 8 años) |
| **Riesgo hoy** | La probabilidad que calcula el modelo para esta semana | 1,64 % |
| **Veces lo normal** | Riesgo hoy ÷ riesgo normal | 6,8 veces lo normal |

**Ojo:** «6,8 veces» suena enorme, pero 1,64 % sigue siendo una probabilidad baja. Por eso mostramos siempre las tres cifras juntas.

Cómo traducimos las «veces lo normal» a palabras:
- Hasta 1,2 veces → **Normal**
- 1,2 a 2 veces → **Algo por encima**
- 2 a 4 veces → **Por encima de lo normal**
- 4 veces o más → **Muy por encima**
- Si sube pero no hubo ningún temblor cerca en 4 días → **Alza solo estadística** (no hay actividad real en la zona).

## ¿En qué mira el modelo?
Cada hora cuenta los temblores de los últimos 4 días (96 horas) alrededor de cada región y los compara con lo que es habitual allí:
- cuántos hubo a menos de 200 km y cuántos de ellos fueron fuertes;
- cuántos hubo a menos de 500 km (el contexto regional);
- cuántos ocurrieron a profundidad intermedia (70–150 km bajo tierra);
- si esa cantidad es mucho mayor que lo normal de la región.

## ¿Cómo aprendió?
Se le dieron unos 11 años de historia de temblores (catálogo del USGS) con una pregunta por cada región y día: «¿hubo un terremoto fuerte en los 7 días siguientes?». El modelo buscó patrones entre lo que pasaba antes y lo que ocurrió después. Después se **calibró** para que sus porcentajes sean honestos: si dice 2 %, debería ocurrir aproximadamente 2 de cada 100 veces.

## ¿Cómo sabemos si funciona?
Se probó como un examen: aprendió con datos hasta finales de 2023 y se evaluó con datos de 2024 a 2026 que nunca había visto, dejando 7 días de separación para que no haya trampa. Se comparó con un modelo más simple (de 5 variables).

| Prueba | Modelo final | Modelo simple | Qué mide |
|---|---|---|---|
| ROC-AUC | 0,64 | 0,625 | Si ordena bien las regiones de más a menos riesgo (0,5 = azar, 1 = perfecto) |
| PR-AUC | 0,046 | 0,027 | Si acierta cuando avisa; casi el doble que el simple |
| Brier | 0,014 | — | Qué tan fiables son sus porcentajes (más cerca de 0 = mejor) |

**Veredicto honesto:** funciona mejor que el modelo simple y sus porcentajes son fiables, pero la mejora es modesta. Predecir terremotos es muy difícil y no pretendemos hacerlo; solo medimos si hoy hay más o menos riesgo que de costumbre.

## Límites
- No predice cuándo ni dónde ocurrirá un terremoto concreto.
- En regiones con muy pocos terremotos fuertes en su historia, el «riesgo normal» es menos preciso.
- Es una guía, no un pronóstico. Ante un sismo real, siguen las indicaciones de protección civil.

---

# Detalle técnico (para quien lo necesite)

- **Objetivo:** `y_M65_7d` = hay al menos un sismo M ≥ 6,5 en la zona en los 7 días siguientes.
- **Datos:** `data/zona_dia_11a.csv` (una fila por zona y día; ignorado por git, hay que regenerarlo para reentrenar). 20 zonas en Supabase (nombres que reconoce el modelo).
- **Variables (10):** `loc200_n96_con`, `loc200_n96_sin45`, `loc200_n96_m45plus`, `loc500_n96_con`, `lat_n96_prof70_150`, `media_96h_zona`, `p90_con_11a`, `ratio_actividad_local` (actividad ÷ (media + 0,25)), `exceso_p90_local` (actividad − percentil 90), `concentracion_espacial` (cercanos ÷ (regionales + 1)).
- **Algoritmo:** `GradientBoostingClassifier` (90 árboles, profundidad 3, tasa de aprendizaje 0,04, `subsample` 0,8, mínimo 30 muestras por hoja) dentro de `CalibratedClassifierCV` (sigmoide, 3 pliegues). Código: `api/train.py`; artefacto `api/modelo_sismos.joblib`, versión `v2_features_11a`.
- **Partición:** entrenamiento hasta 2023-12-31; prueba desde 2024-01-08 (hueco de 7 días).
- **Métricas en prueba** (también en `/health`): ROC-AUC 0,6447 · PR-AUC 0,0458 · Brier 0,01443 (base: 0,625 y 0,0265).
- **Uso en producción:** `calcular_prediccion` en `api/main.py`; `/vigilar` guarda `multiplicador_riesgo` y `probabilidad_m65_7d` en `zonas_sismicas` (`db/zonas_riesgo_hoy.sql`).
