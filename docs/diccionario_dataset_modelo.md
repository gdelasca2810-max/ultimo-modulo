# Diccionario de `data/zona_dia_11a.csv`

78 360 filas = 20 zonas × 3 918 días (2016-01-05 → 2026-09-26, 00:37 UTC). 45 columnas.
Estado de cada definición: **E** = verificado con datos · **H** = hipótesis por nombre/uso, sin confirmar · **?** = desconocido. No existe el generador (ver `docs/auditoria_datos_modelo.md`).

Ventanas: `n96` = 96 h previas (H), `n24` = 24 h previas (H), `d1…d4` = día 1…4 antes (H), `7d`/`30d`/`48h` en `y_*` = horizonte hacia delante (H).

## Identificación
| Columna | Significado | Estado | Uso recomendado |
|---|---|---|---|
| `zona` | Nombre de zona del modelo (20). Ver `docs/zonas_modelo.csv` | E | Solo para agrupar/validar, no como feature |
| `t_eval_utc` | Instante de evaluación (siempre 00:37 UTC, cadencia diaria exacta) | E | Eje temporal de la partición |
| `dest_lat_c`, `dest_lon_c` | Centro de la zona (constante por zona). «dest» no está definido | E (constantes), H (significado) | No usar como feature (identidad de zona) |

## Conteos pasados (candidatas a features; ventana solo hacia atrás: H)
| Columna | Significado | Estado | Notas |
|---|---|---|---|
| `loc200_n96_con` | Sismos a ≤ 200 km del centro en 96 h | H (radio y ventana por nombre y uso en API) | Feature del modelo. Umbral de magnitud desconocido (API: M ≥ 2,5) |
| `loc200_n96_sin45` | De los anteriores, los de M < 4,5 | H (la API los calcula así) | `≤ loc200_n96_con` (E) |
| `loc500_n96_con` | Sismos a ≤ 500 km en 96 h | H | `≥ loc200` (E) |
| `lat_n96_prof70_150` | Sismos de profundidad 70–150 km en 96 h | H | **Feature del modelo** pero de la familia `lat_*`; la API la calcula dentro de 500 km. Posible desajuste (E: `lat_*` tiene otra escala que `loc*`) |
| `lat_n96_som_con` | `som` = somero (H); conteo en la región `lat` en 96 h | H | Media por fila 4,41 |
| `lat_n96_som_sin` | Variante `sin` de la anterior (¿sin réplicas/sin M4.5?) | ? | `con`/`sin` no está definido |
| `lat_n96_som45_con` | Somero con M ≥ 4,5 (H) | H | |
| `lat_n96_prof_con` | `prof` = profundo/intermedio (cualquier profundidad > somera, H) | H | `lat_n96_prof70_150 ≤ lat_n96_prof_con` (E, 100 %) |
| `lat_n96_todo_con` | Total en la región `lat` | H | ≈ `som + prof` en el 97,6 % de las filas (E) |
| `lat_n24_som_con`, `lat_n24_som_sin` | Somero en 24 h | H | |
| `lat_d4_som_con`, `lat_d3_som_con`, `lat_d2_som_con`, `lat_d1_som_con` | Somero por día previo | H | |

La región `lat_*` NO es el radio de 500 km: `loc500 ≥ lat_n96_todo_con` solo en el 13,3 % de las filas (E). Su definición (¿caja lat/lon?, ¿menor umbral de magnitud?) es **?**.

## Estadísticos por zona (FUGA TEMPORAL)
| Columna | Significado | Estado |
|---|---|---|
| `media_96h_zona` | Media sobre **todo 2016-2026** de `lat_n96_som_con` por zona (dif. máx. 0,005) | E. Constante por zona → usa el futuro. Escala ~13,8× la de `loc200` |
| `p90_con_11a` | Percentil 90 sobre **todo el periodo** de `lat_n96_som_con` por zona (11 años: «11a») | E (coincide con cuantil 0,9 completo). Constante por zona → usa el futuro. **No usar como feature** |
| `p90_sin_11a`, `p90_con_viejo` | Variantes del percentil 90 | Constantes por zona (E) → misma fuga; definición exacta ? |

## Flags
`flag_A1`, `flag_A2`, `flag_R1_11a`, `flag_R2_11a`, `flag_P`, `flag_reforzada`: reglas/alertas derivadas (H). Definición **?**. Dependen probablemente de `p90_*`/`media`, por lo que heredan la fuga; no usar sin el generador.

## Variables astronómicas (NO usar)
`luna_dist_km`, `sol_dist_km`, `dias_a_sicigia`, `luna_fase_frac`, `perigeo_flag`, `potencial_mareal_g2`: calculables sin catálogo (E que existen en el CSV). Prohibidas para el modelo (`AGENTS.MD`).

## Targets (futuro; solo para etiquetar)
| Columna | Significado | Estado |
|---|---|---|
| `y_M55_48h`, `y_M55_7d`, `y_M55_30d` | ¿Hubo M ≥ 5,5 en la zona en las 48 h / 7 d / 30 d siguientes? | H (anidamiento verificado: E) |
| `y_M60_48h`, `y_M60_7d`, `y_M60_30d` | Igual con M ≥ 6,0 | H |
| `y_M65_48h`, `y_M65_7d`, `y_M65_30d` | Igual con M ≥ 6,5. **Objetivo del modelo: `y_M65_7d`** (1 457 positivos, 202 episodios) | H. 204 filas con `y_magmax_7d ≥ 6,5` e `y_M65_7d = 0`: filtro adicional **?** |
| `y_magmax_7d` | Magnitud máxima de la zona en los 7 d siguientes (0 en el 10,7 % de las filas) | E (rango y relación con `y_M65_7d`) |

## Features de segunda capa (calculadas en `api/train.py`, no están en el CSV)
`loc200_n96_m45plus = max(0, loc200_con − loc200_sin45)`; `ratio_actividad_local = loc200 / (media_96h_zona + 0,25)`; `exceso_p90_local = loc200 − p90_con_11a`; `concentracion_espacial = loc200 / (loc500 + 1)`. Las dos intermedias dependen de estadísticos con fuga.

## Features seguras recomendadas para un modelo nuevo (condicionado a verificar el generador)
`loc200_n96_con`, `loc200_n96_sin45`, `loc500_n96_con`, y `lat_n96_prof70_150` solo si se alinea su definición con producción. Sin `media`/`p90`/`flag_*`/lunares/coordenadas.
