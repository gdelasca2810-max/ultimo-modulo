# Continuidad ML — 2026-11-25

## Estado

- No hubo commit/push ni cambios en `.github/workflows/vigilancia_sismica.yml`.
- El modelo activo `api/modelo_sismos.joblib` **no fue sustituido**.
- Usuario autorizó **opción C**: ampliar features sísmicas, manteniendo exclusión de coordenadas como identidad, lunares, P90/medias globales y flags con fuga.

## Fuente histórica recuperada

El usuario aportó el generador T3 completo. Reglas exactas relevantes:

- Catálogo histórico: USGS M>=4.0; `type=earthquake`.
- Declustering Gardner--Knopoff (`02_declustering.py`); targets y `loc200_n96_sin45` excluyen réplicas.
- `loc200_n96_con`: radio <=200 km al centroide de la caja destino, M>=4.0, profundidad 5..40 km, `[t-96h,t)`.
- `loc200_n96_sin45`: radio <=200 km, M>=4.5, prof. 5..40 km, **sin réplicas**. El nombre no significa “M<4.5”.
- `loc500_n96_con`: radio <=500 km, M>=4.0, prof. 5..40 km.
- `lat_n96_prof70_150`: cajas laterales, M>=4.5, `70<prof<=150`; no es radio 500 km.
- `y_M65_7d`: caja destino, M>=6.5, sin réplicas, cualquier profundidad, `[t,t+7d)`.
- Cajas en `config/zonas_t3_historicas.json`; están marcadas por el generador original como “DEFINIDA POR CLAUDE - revisar”, por lo que reproducen historia pero no tienen validación sismológica.

## Datos disponibles

- Raw USGS: `data/raw/usgs_catalogo_m25_2015_2026.csv`, 310.185 eventos M>=2.5, 2015-12-28..2026-11-24; ignorado por Git.
- Centros Supabase entregados por usuario: coinciden con los centroides históricos de las 20 cajas destino.
- Dataset reconstruido: `data/candidate/zona_dia_t3_reconstruido.csv`, ignorado por Git.

## Reproducción lograda

`scripts/reconstruir_t3_usgs.py` reproduce el CSV histórico con catálogo ComCat actual:

- `y_M65_7d`: 99,99% exactitud.
- `loc200_n96_con`: 99,87%; `loc200_n96_sin45`: 99,96%; `loc500_n96_con`: 99,50%.
- Las pequeñas diferencias son compatibles con revisiones posteriores del catálogo USGS.

## Resultados de modelos (no desplegar)

- Baseline tasa por zona sigue ganando en AUC/Brier.
- v4 alineado con API: GB AUC 0,56667 / AP 0,02559 / Brier 0,01365 vs baseline 0,64562 / 0,02546 / 0,01362.
- XGBoost v4: AUC 0,55994 / AP 0,02695 / Brier 0,01365; peor que baseline.
- T3 ampliado (18 features sísmicas) en test: GB AUC 0,57892 / AP 0,03888 / Brier 0,01504; mejora AP pero pierde AUC/Brier ante baseline 0,64576 / 0,02863 / 0,01484.

## Resultado final del modelo jerárquico

Se ejecutó `api/evaluar_t3_prior_sismico.py`: tasa histórica como prior y
señal sísmica T3 como actualización, con validación 2023 y test 2024+ separados.

- Validación 2023: baseline AUC 0,78702 / AP 0,11214 / Brier 0,02716;
  prior+sismo 0,77720 / 0,17514 / 0,02760.
- Test 2024+: baseline AUC 0,63209 / AP 0,02812 / Brier 0,01484;
  prior+sismo 0,63878 / 0,02857 / 0,01679.

La mejora mínima de ranking no compensa el deterioro material de calibración.
**No hay candidato desplegable** con los datos/features probados. El baseline
de tasa histórica por zona sigue siendo la opción probabilística defendible.

Para desplegar cualquier modelo T3 sería necesario adaptar `/vigilar` para calcular exactamente las cajas laterales y el declustering; no hacerlo hasta que un candidato gane al baseline y se valide científicamente la geometría.

## Archivos principales creados/modificados

- `config/zonas_t3_historicas.json`
- `scripts/reconstruir_t3_usgs.py`
- `api/evaluar_t3_expandido.py`
- `api/evaluar_t3_prior_sismico.py` (ejecutado y verificado de nuevo el 2026-11-25: mismas cifras, ~8 s)
- `scripts/contrastar_catalogo_usgs.py`, `scripts/contrastar_target_usgs.py`
- `docs/contraste_usgs_2026-11-25.md`
- `.gitignore` ignora `data/raw/`, `data/candidate/` y candidatos joblib.

## Actualización 2026-11-25 (fase producción segura)

- Producción **intacta por defecto**: `api/modelo_sismos.joblib` sin cambios. Se añadió el modo opcional `MODO_PREDICCION=baseline_zona` (apagado), con `api/baseline_tasa_zona.json` generado por `scripts/generar_baseline_zona.py`. Detalle y decisiones de producto pendientes en `docs/PLAN_PRODUCCION_MODELO.md`.
- Hallazgo: el artefacto activo tiene `Japon Nankai/Kyushu` con tasa 0,0 (sin positivos ≤2023); por eso el baseline usa tasa suavizada.
- Nuevo: `api/criterios_despliegue.py` (criterios obligatorios + bootstrap por bloques), `config/zonas_t3_estado_validacion.json` (geometría NO validada), `tests/` (25 pruebas con `unittest`).
- `scripts/reconstruir_t3_usgs.py`: filtra `type=earthquake` y puede generar el calendario sin el CSV histórico.
- Siguen vigentes: no hay candidato ML desplegable; la geometría T3 reproduce historia pero no está validada; `/vigilar` no replica las features T3 (train-serve skew del modelo activo).
- Los documentos `docs/auditoria_datos_modelo.md`, `docs/diccionario_dataset_modelo.md`, `docs/definicion_geografica_zonas.md` y `docs/regla_target_m65_7d.md` se escribieron antes de recuperar el generador: donde contradigan este archivo (umbral M4,0 somero, cajas destino, `lat_*` = cajas laterales, target con Gardner–Knopoff), vale este archivo.

## Actualización 2026-11-25 (v5: arreglo completo)

- **Hay candidato desplegable.** `api/train_t3_multiplicativo.py` (tasa de zona × escalones sobre actividad local) supera al baseline en AUC, AP y Brier en test 2024+, con bootstrap por meses sin deterioro (detalle en `docs/PLAN_PRODUCCION_MODELO.md`). Sustituye la conclusión anterior de «no hay candidato desplegable»: antes se probaban modelos genéricos que no capturaban la señal, que está solo en la cola de actividad local.
- `api/features_t3.py` es ahora la única implementación de las features; `/vigilar` la usa (2 000/2 000 valores idénticos al dataset de entrenamiento en 25 fechas).
- `api/main.py` carga `modelo_sismos_v5.joblib` por defecto; baseline opcional con `MODO_PREDICCION=baseline_zona`.
- Sin commit ni push; ver la sección «Activación y orden de despliegue» del plan.

## Actualización 2026-11-25 (v6: todo M4,0)

- Decisión del usuario: **el umbral M4,5 no existe**. `loc200_n96_sin45` = M4,0+ sin réplicas; `lat_n96_prof70_150` = M4,0+ a 70–150 km en las cajas laterales. Estas dos columnas dejan de coincidir con el CSV histórico (que usaba M4,5); `loc200_n96_con`, `loc500_n96_con` y los targets siguen coincidiendo ≥ 99,5 %. Se eliminó `lat_n96_som45_con` del generador.
- Artefacto activo: `api/modelo_sismos_v6.joblib` (reemplaza al v5, retirado). Métricas e interpretación honesta (sin mejora estadísticamente demostrada, sin deterioro) en `docs/PLAN_PRODUCCION_MODELO.md`.
- La historia de mainshocks para el declustering en `/vigilar` pasó a 2 600 días por las cadenas de réplicas; con caché de 12 h.
- Donde `docs/auditoria_datos_modelo.md`, `docs/diccionario_dataset_modelo.md`, `docs/definicion_geografica_zonas.md` o `docs/regla_target_m65_7d.md` mencionen M4,5 para estas variables, vale este archivo.

## Actualización 2026-11-26 (v7: radio local 300 km)

- Exploración de profundidad y radio (detalle en `docs/PLAN_PRODUCCION_MODELO.md`): el somero 5–40 km es la profundidad con señal estable; quitar el filtro de profundidad no mejora; 300 km mejora el modelo en validación cruzada y en test. Aplicado por decisión del usuario.
- `RADIO_LOCAL_KM = 300` en `api/features_t3.py`; las columnas `loc200_*` conservan su nombre pero miden ≤ 300 km. Modelo activo `api/modelo_sismos_v7.joblib` (reemplaza al v6). En test 2024+: AUC +0,016 y AP +0,007 con IC95 que excluyen el cero, Brier sin deterioro.
- Identidad producción/entrenamiento: 2 000/2 000 con datos reales.
