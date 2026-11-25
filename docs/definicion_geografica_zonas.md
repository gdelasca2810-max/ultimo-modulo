# Definición geográfica de las zonas

Fecha: 2026-11-25. No hay polígonos ni cajas definidos en el repo, así que **no** se generó `data/export/zonas_geometria_modelo.geojson`. Estado de cada dato: **CONFIRMADO** (está en código o fue medido) o **HIPÓTESIS**.

## 1. Qué existe en el repo
- **Tabla `zonas_sismicas` (Supabase):** el repo NO contiene su `CREATE TABLE`. Solo hay `db/seguridad_y_telegram.sql` (RLS: lectura pública) y `db/zonas_riesgo_hoy.sql` (añade `multiplicador_riesgo`, `probabilidad_m65_7d`, `actualizado_en`). Columnas que usa el código: `zona`, `lat_centro`, `lon_centro`, `loc200_n96_con`, `loc200_n96_sin45`, `loc500_n96_con`, `lat_n96_prof70_150`, `tasa_historica_m65_7d`, `multiplicador_riesgo`, `probabilidad_m65_7d`, `actualizado_en`. Los valores reales de `lat_centro/lon_centro` **no están en el repo**; `/vigilar` los prefiere sobre las constantes de `api/main.py`.
- **Constantes de código (CONFIRMADO):** `ZONAS_COORDS` en `api/main.py` (21 zonas) y en `agente/vigilante.py` (20, sin `Venezuela/Caribe`): un solo punto por zona.
- **Radios de producción (CONFIRMADO en `api/main.py`, `agente/vigilante.py`, `mcp/server.py`):** distancia haversine al centro; `loc200` ≤ 200 km; `loc500` ≤ 500 km; `lat_n96_prof70_150` = ≤ 500 km y profundidad 70–150 km; magnitud ≥ 2,5; ventana de 96 h. No hay otros radios, cajas ni polígonos en el repo.
- **Entrenamiento histórico (`data/zona_dia_11a.csv`):** un centro por zona (`dest_lat_c`, `dest_lon_c`, constante: CONFIRMADO). El radio o región de `loc*` y `lat_*` es HIPÓTESIS: según `docs/contraste_usgs_2026-11-25.md`, recalculando con círculos de 200/500 km coinciden exactamente el 83 %, 95,5 % y 52 % de las filas con M≥4,5, y solo el 71 %, 70 % y 32 % con M≥2,5.

## 2. Zonas del modelo/CSV: centros y geometría mínima implicada por el target
| Zona del modelo/CSV | Centro CSV (lat, lon) | Centro `api/main.py` | Centro `agente/vigilante.py` | Eventos seguros | Radio mín. que los cubre (km) | Caja mínima de eventos seguros (lat / lon) |
|---|---|---|---|---:|---:|---|
| Alaska/Aleutianas | 56, -162.5 | 54, -163 | 54, -163 | 10 | 1138 | 51.4..61.3 / -178.0..-149.2 |
| Antillas/Caribe | 15, -63 | 18.2, -66.5 | 18.2, -66.5 | 1 | 406 | 16.5..16.5 / -59.6..-59.6 |
| Chile Central/Sur | -35, -72.5 | -36.5, -73 | -36.5, -73 | 6 | 561 | -39.3..-30.0 / -75.0..-70.4 |
| Chile/Peru (Lima-Valpo) | -21, -73.5 | -18.5, -70.5 | -18.5, -70.5 | 10 | 1444 | -33.0..-8.1 / -74.7..-70.2 |
| Costa O EEUU | 40.5, -121.5 | 37.5, -122 | 37.5, -122 | 4 | 684 | 35.8..44.5 / -125.0..-115.1 |
| Filipinas Luzon | 16.5, 121.5 | 15.5, 121 | 15.5, 121 | 2 | 322 | 13.7..17.5 / 120.7..120.8 |
| Filipinas Mindanao | 7.5, 125 | 7, 126.5 | 7, 126.5 | 10 | 277 | 5.5..9.9 / 123.2..126.9 |
| Indonesia Banda/Timor | -8, 127.5 | -6.5, 129 | -6.5, 129 | 8 | 496 | -7.5..-6.2 / 123.1..131.2 |
| Indonesia Sonda | -8, 110 | -6, 105 | -6, 105 | 9 | 1082 | -8.3..-5.6 / 104.8..119.8 |
| Japon NE | 39, 142.5 | — | — | 9 | 413 | 37.4..42.7 / 141.4..143.4 |
| Japon Nankai/Kyushu | 32.5, 134.5 | 32.5, 133.5 | 32.5, 133.5 | 1 | 294 | 31.8..31.8 / 131.5..131.5 |
| Kamchatka/Kuriles | 50, 154.5 | 51, 158 | 51, 158 | 11 | 945 | 44.6..56.9 / 149.0..163.2 |
| Mediterraneo | 37.5, 32 | — | — | 5 | 1009 | 34.2..38.4 / 20.6..39.1 |
| Mexico/Centroam | 16.5, -97 | — | — | 11 | 905 | 13.2..18.5 / -103.0..-89.3 |
| PNG/Nueva Bretana | -5, 147 | — | — | 18 | 659 | -7.8..-3.6 / 142.8..152.8 |
| Salomon/Vanuatu | -14.5, 163 | — | — | 21 | 1255 | -22.7..-6.2 / 155.2..170.4 |
| Sandwich del Sur | -58, -26 | — | — | 5 | 469 | -60.8..-55.3 / -31.9..-20.0 |
| Taiwan | 23.5, 121 | — | — | 3 | 168 | 23.1..24.7 / 121.3..122.0 |
| Tonga/Kermadec/NZ | -26.5, 180 | — | — | 34 | 1437 | -37.4..-15.3 / -179.5..180.0 |
| Venezuela/Caribe | 10.5, -66.5 | 10.5, -66.9 | — | 0 | n/d | n/d |

«Eventos seguros»: terremotos M≥6,5 del catálogo USGS actual que son el único evento no excluido en la ventana de alguna fila con `y_M65_7d=1` (método en `docs/regla_target_m65_7d.md`). Son una **cota inferior**: la región real contiene al menos esa caja y ese radio. No son la geometría. `Venezuela/Caribe` no tiene ningún evento seguro (sus 14 filas positivas siempre tienen ≥ 2 candidatos).

## 3. Zonas de la API frente al CSV
| Zona de la API (`ZONAS_COORDS`) | Centro API (lat, lon) | ¿Existe en el CSV con ese nombre? |
|---|---|---|
| Alaska/Aleutianas | 54, -163 | sí |
| Antillas/Caribe | 18.2, -66.5 | sí |
| Chile Central/Sur | -36.5, -73 | sí |
| Chile/Peru (Lima-Valpo) | -18.5, -70.5 | sí |
| Costa O EEUU | 37.5, -122 | sí |
| Filipinas Luzon | 15.5, 121 | sí |
| Filipinas Mindanao | 7, 126.5 | sí |
| Indonesia Banda/Timor | -6.5, 129 | sí |
| Indonesia Sonda | -6, 105 | sí |
| Japon Nankai/Kyushu | 32.5, 133.5 | sí |
| Japon Tohoku/Hokkaido | 38.5, 142 | **no** |
| Kamchatka/Kuriles | 51, 158 | sí |
| Mediterraneo Oriental | 36.5, 26 | **no** |
| Mexico Guerrero/Oaxaca | 16.5, -98.5 | **no** |
| Nueva Zelanda/Kermadec | -34, 179 | **no** |
| Papua Nueva Guinea | -5.5, 148 | **no** |
| Solomon/Vanuatu | -11, 165 | **no** |
| Sumatra/Andaman | 2.5, 95.5 | **no** |
| Taiwan/Ryukyu | 24, 122.5 | **no** |
| Tonga/Fiji | -20, -175 | **no** |
| Venezuela/Caribe | 10.5, -66.9 | sí |

Zona del CSV que ninguna zona de la API usa: `Sandwich del Sur`. Según `docs/contraste_usgs_2026-11-25.md`, los alias geográficos no validados se retiraron de `api/main.py`. `Sumatra/Andaman` no existe en el CSV: p. ej. el M7,8 del 2016-03-02 a −5,0/94,3 no es evento seguro de ninguna zona.

## 4. Qué falta
Polígonos o cajas oficiales por zona, los `lat_centro/lon_centro` reales de Supabase y la región usada para `lat_*`. Sin ellos la geometría histórica sigue siendo hipótesis.
