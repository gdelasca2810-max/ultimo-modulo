# Auditoría de datos y reproducibilidad del modelo M6.5+ a 7 días

Fecha: 2026-11-25. Alcance: solo lectura (no se tocó `api/main.py`, el joblib, Supabase ni el workflow).
Convención: **EVIDENCIA** = medido con código sobre archivos del repo · **HIPÓTESIS** = inferido, sin confirmar · **FALTA** = dato que no existe en el repo.

Reproducir: `python scripts/verificar_reproducibilidad_modelo.py --zonas-csv docs/zonas_modelo.csv` (salida 1 si hay hallazgos críticos).

> Actualización 2026-11-25: ya hay un catálogo USGS crudo reproducible local y
> un contraste completo en `docs/contraste_usgs_2026-11-25.md`. Confirma el
> train-serve skew de `loc_*`, demuestra que el target histórico no es un radio
> circular único y mantiene la conclusión de no desplegar los candidatos.

## 1. Veredicto

El dataset **no se puede regenerar** ni auditar de extremo a extremo: no existe el generador ni el catálogo crudo. Además hay fuga temporal medida en dos variables del modelo en producción y un posible desajuste entrenamiento/producción en una tercera. Con lo que hay hoy no se puede afirmar que un modelo nuevo mejore al actual de forma fiable.

## 2. Pipeline de datos (qué hay y qué falta)

| Pieza | Estado | Evidencia |
|---|---|---|
| `data/zona_dia_11a.csv` | Existe, ignorado por git (`*.csv`); 78 360 filas × 45 columnas, 12,9 MB | CSV local, `.gitignore` |
| Script que genera el CSV | **FALTA**. Ninguna referencia en `scripts/`, `agente/`, `api/`, `mcp/`, `db/`, notebooks | búsqueda en todo el repo |
| Catálogo USGS crudo | **FALTA** (ningún CSV/JSON/Parquet de eventos en el repo ni fuera de `data/`) | búsqueda de archivos |
| `docs/todo.md` | Ya lo reconoce: «Regenerar `data/zona_dia_11a.csv` o documentar cómo obtenerlo» | `docs/todo.md:27` |
| `db/sismos_historicos.sql` | Tabla Supabase de eventos **M4.5+** (id, fecha, mag, prof, lat, lon, lugar). Contenido no accesible desde aquí (sin credenciales); según docs ~90 000 sismos. Es un catálogo parcial: no contiene M2.5–4.5, que la API usa en `/vigilar` | `db/sismos_historicos.sql`, `docs/arquitectura.md` |
| Entrenamiento | `api/train.py` (v2, el que sirve producción) y `api/train_temporal.py` (v3 candidato, sin commit) | código |

Lo que habría que tener para reconstruir el CSV: (a) el catálogo USGS con el umbral de magnitud original, (b) la definición exacta de cada región `lat_*` y de cada radio `loc*`, (c) la regla del target `y_*` (ver §5) y (d) las reglas de las columnas `flag_*`, `p90_*` y `media_96h_zona`. Nada de esto está escrito.

`scripts/descargar_catalogo_usgs.py` permite bajar el catálogo crudo desde la API pública de USGS (no se ejecutó). Parámetros a decidir: rango (≥ 2015-12-28 a 2026-10-31), `--minmag` (desconocido para el CSV original; producción usa 2.5) y `eventtype`.

## 3. Datos temporales (EVIDENCIA)

- `t_eval_utc`: 2016-01-05 00:37 UTC → 2026-09-26 00:37 UTC; una evaluación diaria exacta por zona (3 918 filas × 20 zonas); sin nulos; sin duplicados `(zona, t_eval_utc)`; conteos coherentes (`loc200_sin45 ≤ loc200_con ≤ loc500`).
- Target `y_M65_7d`: 1 457 positivos de 78 360 (tasa global 0,01859).

| Año | Filas | Positivos | Tasa |
|---|---|---|---|
| 2016 | 7 240 | 123 | 0,0170 |
| 2017 | 7 300 | 136 | 0,0186 |
| 2018 | 7 300 | 181 | 0,0248 |
| 2019 | 7 300 | 142 | 0,0195 |
| 2020 | 7 320 | 119 | 0,0163 |
| 2021 | 7 300 | 117 | 0,0160 |
| 2022 | 7 300 | 137 | 0,0188 |
| 2023 | 7 300 | 210 | 0,0288 |
| 2024 | 7 320 | 91 | 0,0124 |
| 2025 | 7 300 | 122 | 0,0167 |
| 2026 (hasta 26-sep) | 5 380 | 79 | 0,0147 |

Por zona (positivos): Japon Nankai/Kyushu 7 · Filipinas Luzon 14 · Venezuela/Caribe 14 · Antillas/Caribe 21 · Taiwan 28 · Costa O EEUU 35 · Chile Central/Sur 42 · Mediterraneo 49 · Indonesia Sonda 58 · Sandwich del Sur 59 · Indonesia Banda/Timor 63 · Japon NE 70 · Alaska/Aleutianas 77 · Mexico/Centroam 77 · Filipinas Mindanao 77 · Chile/Peru 91 · Kamchatka/Kuriles 92 · PNG/Nueva Bretana 136 · Salomon/Vanuatu 169 · Tonga/Kermadec/NZ 278. Ninguna zona tiene 0.

**Tamaño efectivo de muestra:** los 1 457 positivos forman solo **202 episodios** (rachas consecutivas de `y=1` por zona; mediana 7 días, máx. 17), porque cada evento enciende ~7 filas diarias con ventana solapada. Toda métrica (AUC, AP, Brier) debe llevar intervalos por bloque/episodio, no por fila. El test de `train.py` tiene 292 positivos (≈ 40 episodios).

## 4. Fuga temporal (EVIDENCIA, crítico)

1. **`media_96h_zona` y `p90_con_11a` son constantes por zona en las 78 360 filas.** No son función de lo ocurrido antes de cada `t_eval_utc`; son estadísticos de la muestra completa 2016–2026, incluido el periodo de test.
2. `media_96h_zona` = media sobre todo el periodo de `lat_n96_som_con` (diferencia máx. 0,005 entre zonas). `p90_con_11a` = percentil 90 de `lat_n96_som_con` sobre todo el periodo (diferencia 0 con la muestra completa; 2 si solo se usa el train). Ambos usan además información del test.
3. `train.py` los toma con `groupby('zona').first()`, así que el modelo servido recibe datos del futuro como feature. Las métricas de test de `/health` (AUC 0,6447, AP 0,0458, Brier 0,01443) están por tanto contaminadas en una cantidad **no medida**.
4. `tasas_zona` (tasa base) sí se calcula solo con train (≤ 2023-12-31) en `train.py`. En `train_temporal.py` se suaviza con una referencia ≤ 2023-12-31.
5. Desajuste de escala: `media_96h_zona` es ~13,8 veces la media de `loc200_n96_con`, pero `ratio_actividad_local = loc200 / (media + 0,25)` los divide como si fueran la misma magnitud (HIPÓTESIS sobre la intención; la escala es EVIDENCIA).

## 5. Target y particiones

- **Definición de `y_M65_7d`:** NO se puede confirmar. EVIDENCIA: `y_M65_7d=1` implica `y_magmax_7d ≥ 6,5` en el 100 % de los casos, y las ventanas están anidadas (48h ⊂ 7d ⊂ 30d; M6,5 ⊂ M6,0 ⊂ M5,5) sin violaciones. Pero hay **204 filas (0,26 %)** con `y_magmax_7d ≥ 6,5` y `y_M65_7d = 0`: el target aplica un filtro adicional no documentado (región, profundidad o réplicas; HIPÓTESIS). Ventana hacia delante de 7 días desde `t_eval_utc`: HIPÓTESIS razonable (consistente con el nombre y con rachas de 7), pero no verificable sin el catálogo.
- **Particiones (EVIDENCIA, calculadas con el código real de cada script):**
  - `train.py` (v2): train ≤ 2023-12-31 (58 360 filas, 1 165 positivos); test ≥ 2024-01-08 (19 860 filas, 292 positivos). La etiqueta de la última fila de train termina el 2024-01-07: no solapa con el test.
  - `train_temporal.py` (v3, candidato): ajuste ≤ 2022-12-24 (50 920 / 955); calibración 2023-01-01..2023-12-24 (7 160 / 206); test igual que v2. Sin solapamiento entre bloques.
  - Riesgo (HIPÓTESIS): `CalibratedClassifierCV(cv=3)` de `train.py` divide el train en 3 bloques contiguos sin hueco; la calibración interna puede ver etiquetas con ventana solapada al bloque de ajuste.
  - Test 2024-01-08 → 2026-09-26: tasa 0,0147 vs 0,0200 en train: hay deriva de la tasa base; 2023 (0,0288) fue atípico.

## 6. Zonas (EVIDENCIA; tabla completa en `docs/zonas_modelo.csv`)

- La API (`api/main.py`) define 21 zonas (`ZONAS_COORDS`); `agente/vigilante.py` define 20 (sin `Venezuela/Caribe`). El modelo y el CSV tienen 20 zonas con otros nombres. Hay 8 alias en `ALIASES_ZONA_MODELO` (cambios locales sin commit en `main.py`).
- **`Sumatra/Andaman`: no existe en el modelo ni en el CSV.** `/vigilar` la omite (`if zona not in artefacto["tasas_zona"]`) y la web puede mostrarla sin riesgo calculado. No hay forma de equipararla científicamente a ninguna zona del CSV: la zona del CSV más cercana, `Indonesia Sonda` (centro −8,0/110,0), está a unos 1 990 km del centro de la API para Sumatra/Andaman (2,5/95,5) y es otra región. **No se debe asumir esa equivalencia.**
- `Sandwich del Sur` está en el modelo pero ninguna zona de la API la usa.
- `Tonga/Fiji` y `Nueva Zelanda/Kermadec` comparten una sola zona de modelo (`Tonga/Kermadec/NZ`), con centro CSV −26,5/180,0 a 839–885 km de los centros de la API: están mezcladas dos regiones sísmicas distintas.
- Los centros de la API (y de Supabase) difieren de los del CSV (`dest_lat_c/dest_lon_c`) en más de 200 km en 11 zonas (hasta 885 km). Como el CSV usa un centro y un radio/región que no se conocen con precisión, los conteos de producción **no miden necesariamente la misma región** que se entrenó (HIPÓTESIS fuerte).
- Alias sin validar: Japón Tohoku/Hokkaido→Japon NE, Mediterráneo Oriental→Mediterraneo (centro a 544 km), México Guerrero/Oaxaca→Mexico/Centroam, Papua Nueva Guinea→PNG/Nueva Bretana, Solomon/Vanuatu→Salomon/Vanuatu (446 km), Taiwan/Ryukyu→Taiwan.

## 7. Desajuste entrenamiento/producción (EVIDENCIA + HIPÓTESIS)

- Las familias `lat_*` y `loc*` tienen escalas incompatibles: media por fila `loc200_n96_con` 0,32; `loc500_n96_con` 1,44; `lat_n96_todo_con` 8,91; `lat_n96_som_con` 4,41. `loc500 ≥ lat_todo` solo en el 13,3 % de las filas. Es decir, `lat_*` cubre una región mayor o un umbral de magnitud menor que el radio de 500 km (EVIDENCIA de que son distintas; la causa exacta es HIPÓTESIS).
- `lat_n96_prof70_150` (**feature del modelo**) pertenece a la familia `lat_*`; la API la calcula como «sismos a ≤ 500 km con profundidad 70–150 km, M ≥ 2,5». Si la región o el umbral difieren (probable), hay **train-serve skew**. Esto afecta también a la nueva etiqueta «Profundidad intermedia» de la web.
- El umbral de magnitud del CSV es desconocido; la API usa 2,5.

## 8. Prioridades (de mayor a menor)

1. **Recuperar o reescribir el generador** del CSV (o su definición exacta): umbral de magnitud, región `lat_*`, radios `loc*`, regla de `y_*`, `flag_*`. Sin esto, nada de lo demás es verificable.
2. **Descargar el catálogo crudo** (`scripts/descargar_catalogo_usgs.py`) y reconstruir los conteos con ventana estrictamente pasada.
3. **Eliminar la fuga:** calcular `media`/`p90` solo con datos anteriores a cada `t_eval_utc` (ventana móvil) o no usarlos. No usar `p90` como feature de un modelo nuevo.
4. **Alinear producción y entrenamiento:** que la API calcule cada feature con la misma región y umbral que el entrenamiento (empezando por `lat_n96_prof70_150`).
5. **Resolver zonas:** definir `Sumatra/Andaman` como zona nueva (con su propia tasa), separar Tonga/Fiji de NZ/Kermadec y fijar un único centro por zona (misma tabla para API, Supabase, agente y entrenamiento).
6. **Evaluar con bloques/episodios**: intervalos de confianza por episodio (≈ 202 en total) y validación hacia delante (walk-forward) con hueco ≥ 7 días, también dentro de la calibración.
7. No usar variables lunares/mareales (el CSV las trae), ni coordenadas como identidad de zona, ni `p90_*`.
8. Comparar siempre contra la línea base de la tasa histórica por zona (ya prevista en `train_temporal.py`).

## 9. Qué compartir con el auditor externo

- Código: `api/train.py`, `api/train_temporal.py`, `api/main.py` (función `calcular_prediccion`, `ALIASES_ZONA_MODELO`, `ZONAS_COORDS`), `agente/vigilante.py`, `scripts/*.py`, `db/sismos_historicos.sql`.
- Docs: este informe, `docs/diccionario_dataset_modelo.md`, `docs/zonas_modelo.csv` (está ignorado por git: añadir con `git add -f` o enviar a mano), `docs/modelo_ml.md`, `AGENTS.MD`.
- Datos: `data/zona_dia_11a.csv` (12,9 MB; sin secretos) y, si existe, el generador original y el catálogo USGS que se usó.
- **No compartir:** `.env`, claves de Supabase/Telegram/OpenAI/Gemini, `CRON_SECRET`, URL privadas ni el contenido de `alertas_usuario`/`vinculos_telegram` (datos de usuarios).
- No cargar `joblib` de origen desconocido (ejecuta código al cargarse): el auditor debe usar el del repo o reentrenar.
