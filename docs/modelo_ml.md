# Modelo de machine learning

## Qué predice
Probabilidad de que ocurra al menos un sismo **M ≥ 6,5 en los próximos 7 días** dentro de una zona tectónica (variable objetivo `y_M65_7d`). Es un evento raro: la tasa base ronda 0,2 % a 8 % semanal según la zona.

## Salida y cómo se interpreta
- `probabilidad_m65_7d`: la estimación del modelo para esta semana.
- `tasa_historica_7d` (tasa base): proporción de semanas con sismo M6.5+ en esa zona durante el entrenamiento.
- **`multiplicador_riesgo` = probabilidad ÷ tasa base.** 1× es lo normal, 2× el doble. No es una predicción de terremotos: mide riesgo relativo. Una zona de tasa muy baja puede marcar 10× y seguir siendo improbable, por eso siempre se muestra la probabilidad real.

## Datos
- Una fila por zona y día (`data/zona_dia_11a.csv`, ~11 años de catálogo USGS; ignorado por git, hay que regenerarlo para reentrenar).
- 21 zonas del Cinturón de Fuego y áreas activas, con coordenadas de referencia en `api/main.py` (`ZONAS_COORDS`).
- Ventana de observación: las últimas 96 horas alrededor del centro de cada zona.

## Variables (v2, 10 en total)
| Variable | Significado |
|---|---|
| `loc200_n96_con` | Sismos a menos de 200 km en 96 h |
| `loc200_n96_sin45` | De esos, los leves (< M4.5) |
| `loc200_n96_m45plus` | De esos, los moderados (M4.5+) |
| `loc500_n96_con` | Sismos a menos de 500 km (contexto regional) |
| `lat_n96_prof70_150` | Sismos a profundidad intermedia (70–150 km) |
| `media_96h_zona` | Actividad media normal de la zona en 96 h |
| `p90_con_11a` | Percentil 90 histórico de actividad local |
| `ratio_actividad_local` | Actividad actual ÷ (media + 0,25) |
| `exceso_p90_local` | Actividad actual − percentil 90 |
| `concentracion_espacial` | Sismos cercanos ÷ (regionales + 1) |

## Algoritmo
`GradientBoostingClassifier` (90 árboles, profundidad 3, tasa de aprendizaje 0,04, `subsample` 0,8, mínimo 30 muestras por hoja) envuelto en `CalibratedClassifierCV` (sigmoide, 3 pliegues) para que las probabilidades sean fiables y no solo un ranking. Se compara con un modelo base de 5 variables. Código: `api/train.py`; artefacto: `api/modelo_sismos.joblib` (versión `v2_features_11a`).

## Validación
- Partición **temporal estricta**: entrenamiento hasta 2023-12-31; prueba desde 2024-01-08 (hueco de 7 días igual al horizonte, para evitar fuga de datos).
- Resultados en prueba (también en `/health`):

| Métrica | Modelo v2 | Base 5 vars | Cómo leerla |
|---|---|---|---|
| ROC-AUC | 0,6447 | 0,625 | 0,5 = azar, 1 = perfecto |
| PR-AUC | 0,0458 | 0,0265 | Precisión con eventos raros; casi el doble que la base |
| Brier | 0,01443 | — | Error de las probabilidades; más bajo es mejor |

## Límites (decirlos siempre)
- Mejora **modesta pero real** sobre la base: predecir sismos grandes es muy difícil.
- No predice cuándo ni dónde ocurrirá un terremoto concreto.
- La tasa base viene solo del periodo de entrenamiento; zonas con pocos eventos tienen estimaciones inestables.
- El CSV de entrenamiento no está en el repositorio.

## Cómo se usa
- `calcular_prediccion` (`api/main.py`) arma las 10 variables y devuelve probabilidad, tasa base y multiplicador.
- `/vigilar` (cada hora) lo calcula para todas las zonas, guarda `multiplicador_riesgo` y `probabilidad_m65_7d` en `zonas_sismicas` (`db/zonas_riesgo_hoy.sql`) y avisa por Telegram.
- Estados en lenguaje simple (web y bot): Normal (< 1,2×), Algo por encima (1,2–2×), Por encima (2–4×), Muy por encima (≥ 4×); con ≥ 1,5× pero sin sismos a menos de 500 km se muestra «Alza solo estadística».
