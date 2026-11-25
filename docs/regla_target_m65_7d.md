# Regla histórica del target `y_M65_7d`

Fecha: 2026-11-25. Alcance: lectura de `data/zona_dia_11a.csv` y del catálogo USGS descargado (`data/raw/usgs_catalogo_m25_2015_2026.csv`, M≥2,5, 310 185 eventos, 446 con M≥6,5, todos `eventtype=earthquake`). El catálogo actual puede tener revisiones posteriores al CSV original. Estados: **CONFIRMADO** (medido) · **HIPÓTESIS**.

## 1. Dónde está escrita la regla
En ningún sitio. No hay generador, SQL, notebook ni documentación del `y_*` de `zona_dia_11a.csv` (`docs/todo.md`: «Regenerar `data/zona_dia_11a.csv`…»). `scripts/construir_dataset_usgs_v4.py` define un target **nuevo** (M≥6,5 a ≤ 500 km en 7 días); no es el histórico. Todo lo siguiente se deduce comparando el CSV con el catálogo.

## 2. Qué se pudo deducir
| Aspecto | Resultado | Estado |
|---|---|---|
| Umbral de magnitud | M ≥ 6,5. Los 1 457 positivos tienen al menos un M≥6,5 del catálogo en los 7 días siguientes, y ninguno queda sin evento no excluido (columna `y1_sin_evento` = 0 en las 20 zonas) | CONFIRMADO (con el catálogo actual) |
| Horizonte | Ventana hacia delante desde `t_eval_utc` (00:37 UTC), 7 días: `(t, t+7d]`. Los positivos forman rachas de 7 filas (mediana 7, máx. 17). Inclusividad exacta de los bordes no probada | Ventana: CONFIRMADO por rachas; bordes: HIPÓTESIS |
| Geometría | **No es un círculo común.** Un círculo de 500 km alrededor del centro recupera 827/1 457 positivos con 195 falsos positivos; para recuperarlos todos haría falta 1 500 km (1 542 falsos positivos). Es una región específica por zona (caja o polígono) | CONFIRMADO que no es un círculo común; forma exacta: HIPÓTESIS |
| Profundidad | Entre los eventos seguros hay profundidades de hasta 671 km (Tonga/Kermadec/NZ) y 640 km (Filipinas Mindanao): **no hay tope de profundidad ≤ 150 km ni filtro somero** | CONFIRMADO |
| Tipo de evento | El catálogo solo contiene `earthquake`; no se pudo probar si el original excluía otros tipos | No determinable |
| Réplicas (declustering) | No se pudo probar ni descartar. Existen columnas `con`/`sin` en el CSV cuyo significado es desconocido | No determinable |
| Eventos M≥6,5 que no cuentan | 204 filas (0,26 %) tienen `y_magmax_7d ≥ 6,5` con `y_M65_7d = 0`: el evento cae fuera de la región de esa zona | CONFIRMADO |
| Superposición entre zonas | Un mismo evento puede contar en varias zonas (p. ej. Antillas/Caribe y Venezuela/Caribe comparten eventos candidatos) | HIPÓTESIS |

## 3. Método (reproducible)
Para cada zona y cada fila `t`, se toman los M≥6,5 con `t < hora ≤ t+7 d`.
1. **Excluidos:** todo evento que aparece en la ventana de alguna fila con `y=0` no cuenta para esa zona.
2. **Seguros:** en una fila con `y=1`, si tras quitar los excluidos queda un único evento, ese evento cuenta para la zona.
3. Se comparan las distancias (haversine, centro `dest_lat_c/dest_lon_c`) de seguros y excluidos.

| Zona | Eventos seguros | Dist. máx. de un evento seguro (km) | Eventos excluidos | Dist. mín. de un evento excluido (km) | Excluidos más cerca que el seguro más lejano | Prof. máx. de eventos seguros (km) |
|---|---:|---:|---:|---:|---:|---:|
| Alaska/Aleutianas | 10 | 1138 | 420 | 239 | 2 | 126 |
| Antillas/Caribe | 1 | 406 | 427 | 598 | 0 | 9 |
| Chile Central/Sur | 6 | 561 | 427 | 701 | 0 | 186 |
| Chile/Peru (Lima-Valpo) | 10 | 1444 | 417 | 528 | 7 | 630 |
| Costa O EEUU | 4 | 684 | 426 | 1155 | 0 | 12 |
| Filipinas Luzon | 2 | 322 | 431 | 572 | 0 | 110 |
| Filipinas Mindanao | 10 | 277 | 420 | 69 | 6 | 640 |
| Indonesia Banda/Timor | 8 | 496 | 422 | 515 | 0 | 624 |
| Indonesia Sonda | 9 | 1082 | 423 | 404 | 5 | 597 |
| Japon NE | 9 | 413 | 422 | 90 | 4 | 46 |
| Japon Nankai/Kyushu | 1 | 294 | 432 | 287 | 1 | 24 |
| Kamchatka/Kuriles | 11 | 945 | 413 | 530 | 5 | 461 |
| Mediterraneo | 5 | 1009 | 422 | 1284 | 0 | 21 |
| Mexico/Centroam | 11 | 905 | 422 | 527 | 2 | 93 |
| PNG/Nueva Bretana | 18 | 659 | 411 | 439 | 2 | 601 |
| Salomon/Vanuatu | 21 | 1255 | 403 | 392 | 10 | 188 |
| Sandwich del Sur | 5 | 469 | 417 | 81 | 1 | 133 |
| Taiwan | 3 | 168 | 428 | 60 | 1 | 63 |
| Tonga/Kermadec/NZ | 34 | 1437 | 378 | 439 | 26 | 671 |
| Venezuela/Caribe | 0 | n/d | 428 | 1007 | n/d | n/d |

- Separación radial limpia (todo seguro más cerca que todo excluido): solo **7 de 20** zonas. En las otras 13 hay eventos excluidos más cerca del centro que el seguro más lejano (p. ej. 26 en Tonga/Kermadec/NZ, 10 en Salomon/Vanuatu): la región no es un círculo centrado en `dest_*`.
- Con un solo evento seguro (Antillas/Caribe, Japon Nankai/Kyushu) o ninguno (Venezuela/Caribe) la geometría no se puede acotar.
- Caveat: magnitudes y localizaciones del catálogo actual pueden diferir de las del momento de construir el CSV; algunos «excluidos» podrían ser eventos con magnitud revisada.

## 4. Qué falta para cerrar la regla
El generador original, o la lista de polígonos/cajas por zona, el catálogo original con su fecha de descarga, y la regla de réplicas/tipo de evento. Sin eso `y_M65_7d` no se puede reconstruir; un target nuevo (p. ej. M≥6,5 a ≤ 500 km) debe tratarse como decisión de producto, no como sustituto del histórico.
