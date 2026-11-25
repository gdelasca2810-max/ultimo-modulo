# Contraste del dataset con catálogo USGS

Fecha: 2026-11-25. Alcance: auditoría y candidatos locales; no se sustituyó el
modelo activo, no se modificó Supabase ni se hizo commit/push.

## Datos obtenidos

Se descargó el catálogo público USGS ComCat mediante el FDSN Event Web Service,
filtrado a `eventtype=earthquake` y `minmagnitude=2.5`.

- Archivo: `data/raw/usgs_catalogo_m25_2015_2026.csv` (ignorado por Git).
- Rango solicitado: 2015-12-28 a 2026-11-25 UTC.
- Rango devuelto: 2015-12-28 00:09:36 UTC a 2026-11-24 23:52:18 UTC.
- Eventos únicos: 310.185; eventos M>=6,5: 446.
- Parámetros y fecha de descarga: `data/raw/usgs_catalogo_m25_2015_2026.csv.meta.json`.

Los datos actuales de ComCat pueden contener revisiones posteriores a las que
existían al crear el CSV histórico; por ello el contraste prueba coherencia
operativa, no una reproducción forense perfecta del catálogo original.

## Features `loc_*`: evidencia de train-serve skew

Se recalcularon las tres features `loc_*` para las 78.360 filas del CSV con
centros `dest_lat_c/dest_lon_c`, ventana `[t-96h, t)` y radios 200/500 km.

| Umbral USGS | `loc200_con` exacto / MAE | `loc200_sin45` exacto / MAE | `loc500_con` exacto / MAE |
|---|---:|---:|---:|
| M>=2,5 (el usado por producción) | 70,76% / 0,550 | 69,98% / 0,539 | 31,57% / 3,039 |
| M>=4,5 | 83,00% / 0,236 | 95,51% / 0,053 | 52,45% / 0,851 |

Con M>=2,5 el sesgo de los conteos recalculados menos los del CSV fue +0,550,
+0,479 y +3,039 respectivamente; con M>=4,5 fue +0,019, -0,053 y -0,043.
Esto es evidencia fuerte de que el CSV histórico usa un filtro próximo a M4,5
o una selección equivalente, mientras `/vigilar` consulta M>=2,5. Por tanto,
el modelo activo recibe una distribución de entrada distinta de su entrenamiento.

Los scripts reproducibles son:

```text
python scripts/contrastar_catalogo_usgs.py --raw data/raw/usgs_catalogo_m25_2015_2026.csv --min-mag 2.5
python scripts/contrastar_catalogo_usgs.py --raw data/raw/usgs_catalogo_m25_2015_2026.csv --min-mag 4.5
```

## Target histórico

Al contrastar `y_M65_7d` con el catálogo crudo:

- Las 1.457 filas positivas tienen al menos un M>=6,5 en su ventana futura de
  siete días: no hay positivos sin evento USGS crudo correspondiente.
- Un círculo común de 500 km solo recupera 827/1.457 positivos (recall 56,76%),
  con 195 falsos positivos (precisión 80,92%).
- Para recuperar todos los positivos haría falta un radio común de 1.500 km,
  que genera 1.542 falsos positivos. El target histórico no es un único radio
  circular común: es coherente con regiones/polígonos específicos por zona.
- La distancia máxima al M>=6,5 más cercano en una fila positiva llega a
  1.444 km (Chile/Perú), 1.437 km (Tonga/Kermadec/NZ) y 1.255 km
  (Salomon/Vanuatu).

Conclusión: los 204 casos `y_magmax_7d>=6,5` con target cero son coherentes con
una regla geográfica no documentada; el catálogo crudo descarta que el problema
sea simplemente ausencia de eventos M>=6,5 para las filas positivas, pero no
permite identificar por sí solo el filtro exacto de los ceros.

Para repetir el contraste:

```text
python scripts/contrastar_target_usgs.py
```

## Dataset y candidato v4

Se construyó un dataset nuevo, explícito y reproducible desde ComCat:

- `data/candidate/zona_dia_usgs_v4.csv` (ignorado por Git).
- 82.278 filas: 21 zonas de `api/main.py` x 3.918 días.
- 1.090 positivos.
- Features: `loc200_n96_con`, `loc200_n96_sin45`, `loc500_n96_con` y
  `lat_n96_prof70_150`, todas M>=2,5, ventana 96 h y geometría de producción.
- Target v4: existe M>=6,5 a <=500 km en los siete días siguientes.

El generador es `scripts/construir_dataset_usgs_v4.py`. Esta definición es una
especificación operacional nueva; no pretende ser el target histórico oculto.
Por defecto toma los centros de `api/main.py`; para reproducir producción debe
recibir un export no sensible de `zonas_sismicas` con solo
`zona,lat_centro,lon_centro` mediante `--zonas-csv`.

El candidato `api/modelo_sismos_v4_candidate.joblib` se entrenó con ajuste
hasta 2022-12-24, calibración exclusiva de 2023 y test desde 2024-01-08. No se
debe desplegar:

| Métrica test | ML v4 | Baseline tasa histórica por zona |
|---|---:|---:|
| AUC | 0,56667 | 0,64562 |
| Average Precision | 0,02559 | 0,02546 |
| Brier | 0,01365 | 0,01362 |

El IC95 por bootstrap de bloques mensuales para `ML - baseline` fue:
`AUC [-0,12804, -0,01918]`, `AP [-0,01272, 0,02250]` y
`Brier [-0,00008, 0,00013]`. El candidato es claramente peor en ranking y no
demuestra mejora de calibración.

## XGBoost

También se evaluó XGBoost 3.4.1 sin incorporarlo a los requisitos de ejecución
de la API. `api/requirements-evaluation.txt` lo declara solo para reproducir
experimentos.

Antes de mirar el test 2024+, se seleccionó configuración con una división
temporal independiente: ajuste hasta 2021-12-24, calibración de 2022 y
validación de 2023. El baseline de zona ganó las cuatro configuraciones:

| Configuración | AUC 2023 | AP 2023 | Brier 2023 |
|---|---:|---:|---:|
| Baseline zona | 0,75801 | 0,04275 | 0,01646 |
| XGBoost shallow | 0,65546 | 0,02939 | 0,01660 |
| XGBoost conservador | 0,64808 | 0,03098 | 0,01659 |
| XGBoost regularizado | 0,63982 | 0,02480 | 0,01661 |
| XGBoost profundo | 0,64695 | 0,02634 | 0,01661 |

El candidato XGBoost conservador volvió a perder en el test 2024--2026:
AUC 0,55994, AP 0,02695 y Brier 0,01365, frente a 0,64562, 0,02546 y 0,01362
del baseline. Su IC95 por bloques para el delta de AUC fue
`[-0,14219, -0,02694]`. Queda guardado solo como
`api/modelo_sismos_v4_xgboost_candidate.joblib`; no se debe desplegar.

## Decisión técnica

1. Mantener `api/modelo_sismos.joblib` sin reemplazar mientras se decide la
   estrategia de producción; sus métricas históricas siguen contaminadas por
   fuga y por train-serve skew.
2. No activar ninguno de los candidatos v3/v4: ambos pierden frente al baseline
   por zona bajo evaluación temporal honesta.
3. Para un modelo condicional mejor hacen falta polígonos/reglas oficiales de
   cada zona y más features permitidas o un cambio explícito de la política de
   features. Con las cuatro features actuales, la tasa histórica por zona es el
   predictor validado más sólido.
4. Si se desea desplegar v4 en el futuro, la definición de target de 500 km debe
   ser una decisión de producto/ciencia explícita, no una sustitución silenciosa
   del target histórico.

## Cobertura de zonas del modelo activo

Los aliases geográficos no validados se retiraron de `api/main.py`: asignar,
por ejemplo, una zona de Tonga/Fiji a otra de Tonga/Kermadec/NZ produce una
probabilidad aparentemente precisa para una geografía distinta. El artefacto
v2 activo reconoce directamente solo 12 de las 21 etiquetas operativas de la
API; las otras deben continuar devolviendo zona no soportada hasta que exista
un artefacto con su misma geometría. El candidato v4 sí contiene las 21,
incluida Sumatra/Andaman, pero sigue bloqueado por sus métricas inferiores.
