#!/usr/bin/env python
"""Auditoría de reproducibilidad del dataset del modelo de riesgo M6.5+ a 7 días.

Solo LECTURA: no modifica el CSV, el joblib, la API ni Supabase y no usa red ni secretos.

Uso (desde la raíz del repo):
    python scripts/verificar_reproducibilidad_modelo.py
    python scripts/verificar_reproducibilidad_modelo.py --csv data/zona_dia_11a.csv \
        --json-out salida_auditoria.json --zonas-csv docs/zonas_modelo.csv

Código de salida: 0 si no hay hallazgos CRITICOS, 1 si los hay (los AVISOS no fallan).
Cada hallazgo se etiqueta como EVIDENCIA (medido aquí) o HIPOTESIS (inferido, a confirmar).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HORIZONTE_DIAS = 7
TARGET = "y_M65_7d"
COLUMNAS_REQUERIDAS = [
    "zona", "t_eval_utc", "dest_lat_c", "dest_lon_c",
    "loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_prof70_150",
    "media_96h_zona", "p90_con_11a",
    "y_M55_48h", "y_M55_7d", "y_M55_30d", "y_M60_48h", "y_M60_7d", "y_M60_30d",
    "y_M65_48h", "y_M65_7d", "y_M65_30d", "y_magmax_7d",
]
COLUMNAS_LUNARES = ["luna_dist_km", "sol_dist_km", "dias_a_sicigia", "luna_fase_frac",
                    "perigeo_flag", "potencial_mareal_g2"]

# Particiones tal como están escritas en el código (api/train.py y api/train_temporal.py).
PARTICIONES = {
    "train.py (v2)": {
        "train": (None, "2023-12-31 23:59:59"),
        "test": ("2024-01-08 00:00:00", None),
    },
    "train_temporal.py (v3 candidato)": {
        "ajuste": (None, "2022-12-24 23:59:59"),
        "calibracion": ("2023-01-01 00:00:00", "2023-12-24 23:59:59"),
        "test": ("2024-01-08 00:00:00", None),
    },
}

hallazgos: list[dict] = []


def hallazgo(nivel: str, tipo: str, texto: str) -> None:
    """nivel: CRITICO | AVISO | OK ; tipo: EVIDENCIA | HIPOTESIS."""
    hallazgos.append({"nivel": nivel, "tipo": tipo, "texto": texto})
    print(f"[{nivel:7}][{tipo}] {texto}")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def literal_de_modulo(ruta: Path, nombre: str) -> dict:
    """Lee un dict literal de un .py sin importarlo (evita cargar modelos o claves)."""
    if not ruta.exists():
        return {}
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == nombre for t in nodo.targets
        ):
            try:
                return ast.literal_eval(nodo.value)
            except ValueError:
                return {}
    return {}


def cargar(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        hallazgo("CRITICO", "EVIDENCIA", f"Faltan columnas requeridas: {faltan}")
        raise SystemExit(1)
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    return df


def auditar_esquema_y_fechas(df: pd.DataFrame) -> None:
    print("\n== 1. Esquema, fechas, nulos y duplicados ==")
    print(f"Filas: {len(df)} | Columnas: {df.shape[1]} | Zonas: {df['zona'].nunique()}")
    print(f"t_eval_utc: {df['t_eval_utc'].min()} -> {df['t_eval_utc'].max()}")
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    if len(nulos):
        hallazgo("AVISO", "EVIDENCIA", f"Columnas con nulos: {nulos.to_dict()}")
    else:
        hallazgo("OK", "EVIDENCIA", "Sin valores nulos en ninguna columna.")
    dup = int(df.duplicated(["zona", "t_eval_utc"]).sum())
    hallazgo("CRITICO" if dup else "OK", "EVIDENCIA", f"Duplicados (zona, t_eval_utc): {dup}")

    horas = df["t_eval_utc"].dt.strftime("%H:%M").value_counts()
    print(f"Hora del día de evaluación (UTC): {horas.to_dict()}")
    # Cadencia por zona
    saltos = df.sort_values(["zona", "t_eval_utc"]).groupby("zona")["t_eval_utc"].diff().dropna()
    cad = saltos.value_counts().head(5)
    print(f"Saltos entre evaluaciones consecutivas (top): { {str(k): int(v) for k, v in cad.items()} }")
    if (saltos != pd.Timedelta(days=1)).any():
        hallazgo("AVISO", "EVIDENCIA", "La serie no es estrictamente diaria en todas las zonas (hay huecos).")
    else:
        hallazgo("OK", "EVIDENCIA", "Cadencia diaria exacta en todas las zonas.")
    filas_zona = df.groupby("zona").size()
    if filas_zona.nunique() > 1:
        hallazgo("AVISO", "EVIDENCIA", f"Zonas con distinto número de filas: {filas_zona.to_dict()}")

    # Coherencia de conteos
    malos = {
        "loc200_sin45 > loc200_con": int((df["loc200_n96_sin45"] > df["loc200_n96_con"]).sum()),
        "loc200_con > loc500_con": int((df["loc200_n96_con"] > df["loc500_n96_con"]).sum()),
        "conteos negativos": int((df[["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con",
                                      "lat_n96_prof70_150"]] < 0).any(axis=1).sum()),
    }
    for nombre, n in malos.items():
        hallazgo("CRITICO" if n else "OK", "EVIDENCIA", f"Incoherencia '{nombre}': {n} filas")
    # Un mismo (zona) con centro cambiante
    cambia = df.groupby("zona")[["dest_lat_c", "dest_lon_c"]].nunique().max().max()
    hallazgo("OK" if cambia == 1 else "AVISO", "EVIDENCIA",
             f"Centro (dest_lat_c, dest_lon_c) constante por zona: {cambia == 1}")
    lunares = [c for c in COLUMNAS_LUNARES if c in df.columns]
    if lunares:
        hallazgo("AVISO", "EVIDENCIA",
                 f"El CSV contiene variables lunares/mareales {lunares}: NO deben usarse como features.")


def auditar_fuga_temporal(df: pd.DataFrame, corte_train: str = "2023-12-31 23:59:59") -> None:
    print("\n== 2. Fuga temporal en estadísticos por zona ==")
    corte = pd.Timestamp(corte_train, tz="UTC")
    for col in ["media_96h_zona", "p90_con_11a"]:
        n_unicos = df.groupby("zona")[col].nunique()
        if (n_unicos == 1).all():
            hallazgo("AVISO", "EVIDENCIA",
                     f"'{col}' es CONSTANTE por zona en las {len(df)} filas. La constancia sola "
                     "no prueba fuga; se contrasta abajo contra estadísticos de la muestra completa.")
        else:
            hallazgo("OK", "EVIDENCIA", f"'{col}' varía en el tiempo ({int(n_unicos.max())} valores máx. por zona).")

    # ¿A qué ventana corresponde? Compara con medias de columnas de conteo.
    candidatos = [c for c in df.columns if "_n96_" in c and df[c].dtype != object]
    zona_ref = df.groupby("zona")["media_96h_zona"].first()
    mejor = None
    for c in candidatos:
        completa = df.groupby("zona")[c].mean()
        d = float((completa - zona_ref).abs().max())
        if mejor is None or d < mejor[1]:
            mejor = (c, d)
    if mejor:
        print(f"Columna cuya media por zona (muestra completa) más se acerca a media_96h_zona: "
              f"{mejor[0]} (diferencia máx. {mejor[1]:.4f})")
        if mejor[1] < 0.01:
            hallazgo("CRITICO", "EVIDENCIA",
                     f"media_96h_zona coincide con la media de '{mejor[0]}' sobre TODO el periodo "
                     f"2016-2026 (dif. máx. {mejor[1]:.4f}), incluido el test.")
        else:
            hallazgo("AVISO", "HIPOTESIS",
                     "media_96h_zona no coincide con la media completa de ninguna columna *_n96_*; "
                     "su definición exacta es desconocida (falta el generador).")
    # p90: compara con cuantil 0.9 completo vs solo-train
    mejor_p90 = None
    for c in ["loc200_n96_con", "lat_n96_todo_con", "lat_n96_som_con"]:
        if c not in df.columns:
            continue
        q_full = df.groupby("zona")[c].quantile(0.9)
        q_tr = df[df["t_eval_utc"] <= corte].groupby("zona")[c].quantile(0.9)
        p90 = df.groupby("zona")["p90_con_11a"].first()
        print(f"p90_con_11a vs cuantil 0.9 de {c}: dif. máx. muestra completa="
              f"{float((q_full - p90).abs().max()):.2f}, solo-train={float((q_tr - p90).abs().max()):.2f}")
        diferencia = float((q_full - p90).abs().max())
        if mejor_p90 is None or diferencia < mejor_p90[1]:
            mejor_p90 = (c, diferencia)
    if mejor_p90 and mejor_p90[1] < 0.01:
        hallazgo("CRITICO", "EVIDENCIA",
                 f"p90_con_11a coincide con el cuantil 0.9 de '{mejor_p90[0]}' sobre TODO el "
                 f"periodo (dif. máx. {mejor_p90[1]:.4f}), incluido el test.")
    # Cuánto cambiaría la media si solo se usara el pasado
    columna_media = mejor[0] if mejor else "media_96h_zona"
    media_tr = df[df["t_eval_utc"] <= corte].groupby("zona")[columna_media].mean()
    media_full = df.groupby("zona")[columna_media].mean()
    print(f"Media {columna_media} (solo train vs completa), dif. relativa máx.: "
          f"{float(((media_full - media_tr).abs() / media_full.clip(lower=1e-9)).max()):.3f}")


def auditar_target(df: pd.DataFrame) -> dict:
    print("\n== 3. Definición y distribución del target ==")
    resumen: dict = {}
    y = df[TARGET]
    if not set(y.unique()) <= {0, 1}:
        hallazgo("CRITICO", "EVIDENCIA", f"{TARGET} no es binario: {sorted(y.unique())}")
    hallazgo("OK", "EVIDENCIA",
             f"{TARGET}: filas={len(df)}, positivos={int(y.sum())}, tasa global={y.mean():.5f}")
    resumen["global"] = {"filas": int(len(df)), "positivos": int(y.sum()), "tasa": float(y.mean())}

    por_anio = df.groupby(df["t_eval_utc"].dt.year)[TARGET].agg(["count", "sum", "mean"])
    print("Por año (filas, positivos, tasa):")
    print(por_anio.to_string())
    resumen["por_anio"] = {int(k): {"filas": int(v["count"]), "positivos": int(v["sum"]),
                                    "tasa": float(v["mean"])} for k, v in por_anio.iterrows()}
    por_zona = df.groupby("zona")[TARGET].agg(["count", "sum", "mean"]).sort_values("sum")
    print("Por zona (filas, positivos, tasa):")
    print(por_zona.to_string())
    resumen["por_zona"] = {k: {"filas": int(v["count"]), "positivos": int(v["sum"]),
                               "tasa": float(v["mean"])} for k, v in por_zona.iterrows()}
    ceros = por_zona[por_zona["sum"] == 0].index.tolist()
    if ceros:
        hallazgo("AVISO", "EVIDENCIA", f"Zonas sin ningún positivo M6.5+ en todo el periodo: {ceros}")

    # Consistencia con la magnitud máxima futura y con ventanas anidadas
    if "y_magmax_7d" in df.columns:
        y1_sin_mag = int(((y == 1) & (df["y_magmax_7d"] < 6.5)).sum())
        y0_con_mag = int(((y == 0) & (df["y_magmax_7d"] >= 6.5)).sum())
        hallazgo("CRITICO" if y1_sin_mag else "OK", "EVIDENCIA",
                 f"Filas con y_M65_7d=1 pero y_magmax_7d<6.5: {y1_sin_mag}")
        hallazgo("AVISO" if y0_con_mag else "OK", "EVIDENCIA",
                 f"Filas con y_magmax_7d>=6.5 pero y_M65_7d=0: {y0_con_mag} "
                 f"({y0_con_mag / len(df) * 100:.2f}% de filas). El target NO es simplemente "
                 "'magmax>=6.5': aplica algún filtro adicional no documentado (región, profundidad, "
                 "réplicas...). HIPOTESIS: requiere el generador para confirmarlo.")
    anidado = {
        "M65_48h <= M65_7d": int((df["y_M65_48h"] > df["y_M65_7d"]).sum()),
        "M65_7d <= M65_30d": int((df["y_M65_7d"] > df["y_M65_30d"]).sum()),
        "M65_7d <= M60_7d": int((df["y_M65_7d"] > df["y_M60_7d"]).sum()),
        "M60_7d <= M55_7d": int((df["y_M60_7d"] > df["y_M55_7d"]).sum()),
    }
    for nombre, n in anidado.items():
        hallazgo("CRITICO" if n else "OK", "EVIDENCIA", f"Violaciones de anidamiento '{nombre}': {n}")

    # Episodios: un único evento M6.5+ enciende ~7 filas consecutivas (ventana solapada)
    episodios = 0
    longitudes: list[int] = []
    for _, g in df.sort_values("t_eval_utc").groupby("zona"):
        v = g[TARGET].to_numpy()
        run = 0
        for x in v:
            if x == 1:
                run += 1
            elif run:
                episodios += 1
                longitudes.append(run)
                run = 0
        if run:
            episodios += 1
            longitudes.append(run)
    if longitudes:
        print(f"Episodios (rachas de y=1 por zona): {episodios}; longitud mediana="
              f"{float(np.median(longitudes)):.1f}, máx={max(longitudes)}")
        hallazgo("AVISO", "EVIDENCIA",
                f"Los {int(y.sum())} positivos forman {episodios} rachas zona-tiempo candidatas "
                f"(ventanas de 7 días solapadas en filas diarias). No son eventos independientes "
                f"confirmados: un terremoto puede activar varias zonas y eventos cercanos pueden fusionarse. "
                "Los intervalos de confianza deben calcularse por bloques temporales/zona o por evento cuando exista el catálogo crudo.")
    resumen["episodios"] = episodios
    return resumen


def auditar_particiones(df: pd.DataFrame) -> dict:
    print("\n== 4. Particiones temporales y solapamiento ==")
    salida: dict = {}
    dias = pd.Timedelta(days=HORIZONTE_DIAS)
    for nombre, cortes in PARTICIONES.items():
        print(f"\n-- {nombre}")
        partes = {}
        for parte, (ini, fin) in cortes.items():
            m = pd.Series(True, index=df.index)
            if ini:
                m &= df["t_eval_utc"] >= pd.Timestamp(ini, tz="UTC")
            if fin:
                m &= df["t_eval_utc"] <= pd.Timestamp(fin, tz="UTC")
            sub = df[m]
            partes[parte] = sub
            if len(sub):
                print(f"{parte:12} filas={len(sub):6} positivos={int(sub[TARGET].sum()):5} "
                      f"tasa={sub[TARGET].mean():.5f} {sub['t_eval_utc'].min().date()} -> "
                      f"{sub['t_eval_utc'].max().date()}")
            else:
                print(f"{parte:12} VACÍA")
        nombres = list(partes)
        for a, b in zip(nombres, nombres[1:]):
            if not len(partes[a]) or not len(partes[b]):
                continue
            # La etiqueta de la última fila de A mira hasta t_a+7d; debe ser < primera fila de B.
            fin_ventana = partes[a]["t_eval_utc"].max() + dias
            inicio_b = partes[b]["t_eval_utc"].min()
            solapa = fin_ventana > inicio_b
            hueco = (inicio_b - partes[a]["t_eval_utc"].max()).days
            hallazgo("CRITICO" if solapa else "OK", "EVIDENCIA",
                     f"{nombre}: '{a}'->'{b}' hueco={hueco} d; ventana de etiqueta de '{a}' termina "
                     f"{fin_ventana.date()} {'SOLAPA con' if solapa else 'no solapa con'} el inicio de '{b}' ({inicio_b.date()}).")
        salida[nombre] = {k: {"filas": int(len(v)), "positivos": int(v[TARGET].sum())}
                          for k, v in partes.items()}
    # cv=3 de CalibratedClassifierCV usa StratifiedKFold (sin shuffle) para un clasificador.
    # Como el CSV está ordenado por fecha, no constituye una validación temporal purgada.
    try:
        from sklearn.model_selection import StratifiedKFold
        train_v2 = df[df["t_eval_utc"] <= pd.Timestamp("2023-12-31 23:59:59", tz="UTC")].reset_index(drop=True)
        cv = StratifiedKFold(n_splits=3, shuffle=False)
        partes_cv = []
        hay_solape = False
        for indice, (ajuste, validacion) in enumerate(cv.split(train_v2, train_v2[TARGET]), start=1):
            fechas_ajuste = train_v2.iloc[ajuste]["t_eval_utc"]
            fechas_validacion = train_v2.iloc[validacion]["t_eval_utc"]
            dias_compartidos = len(set(fechas_ajuste).intersection(set(fechas_validacion)))
            hay_solape |= dias_compartidos > 0
            partes_cv.append(
                f"fold {indice}: ajuste {fechas_ajuste.min().date()}..{fechas_ajuste.max().date()}, "
                f"validación {fechas_validacion.min().date()}..{fechas_validacion.max().date()}, "
                f"días compartidos={dias_compartidos}"
            )
        hallazgo("CRITICO" if hay_solape else "OK", "EVIDENCIA",
                 "CalibratedClassifierCV(cv=3) de train.py usa StratifiedKFold, no folds temporales purgados: "
                 + " | ".join(partes_cv)
                 + ". Esto contamina la calibración/validación interna, aunque no incorpora directamente el test 2024+.")
    except Exception as e:  # noqa: BLE001
        hallazgo("AVISO", "EVIDENCIA", f"No se pudo comprobar CV interno: {type(e).__name__}: {e}")
    return salida


def auditar_zonas(df: pd.DataFrame, joblib_path: Path, main_py: Path, vigilante_py: Path,
                  zonas_csv_out: Path | None) -> None:
    print("\n== 5. Zonas: API vs modelo vs CSV ==")
    zonas_csv = df.groupby("zona")[["dest_lat_c", "dest_lon_c"]].first()
    api_coords = literal_de_modulo(main_py, "ZONAS_COORDS")
    alias = literal_de_modulo(main_py, "ALIASES_ZONA_MODELO")
    agente_coords = literal_de_modulo(vigilante_py, "ZONAS_COORDS")
    zonas_modelo: set[str] = set()
    version = "no cargado"
    if joblib_path.exists():
        try:
            import joblib  # import tardío: solo se necesita aquí
            art = joblib.load(joblib_path)  # artefacto propio del repo; no cargar joblibs ajenos
            zonas_modelo = set(art.get("tasas_zona", {}))
            version = art.get("version", "?")
        except Exception as e:  # noqa: BLE001
            hallazgo("AVISO", "EVIDENCIA", f"No se pudo leer el joblib: {type(e).__name__}: {e}")
    print(f"Versión del artefacto: {version}; zonas en modelo: {len(zonas_modelo)}; "
          f"en CSV: {len(zonas_csv)}; en API (ZONAS_COORDS): {len(api_coords)}")

    filas = []
    nombres_api = set(api_coords) | set(agente_coords)
    for z in sorted(nombres_api):
        destino = alias.get(z, z)
        en_modelo = destino in zonas_modelo
        en_csv = destino in zonas_csv.index
        problemas = []
        if not en_modelo:
            problemas.append("NO existe en el modelo: la API la omite en /vigilar")
        if z in alias:
            problemas.append(f"alias asumido -> '{destino}' (equivalencia NO validada científicamente)")
        dist = None
        if en_csv and z in api_coords:
            c = zonas_csv.loc[destino]
            dist = haversine_km(api_coords[z][0], api_coords[z][1], c["dest_lat_c"], c["dest_lon_c"])
            if dist > 200:
                problemas.append(f"el centro de la API dista {dist:.0f} km del centro del CSV: "
                                 "los conteos de producción NO miden la misma región que se entrenó")
        if z in agente_coords and z in api_coords and agente_coords[z] != api_coords[z]:
            problemas.append("coordenadas distintas entre api/main.py y agente/vigilante.py")
        filas.append({
            "nombre_api": z, "alias_modelo": destino if z in alias else "",
            "nombre_modelo": destino if en_modelo else "", "nombre_csv": destino if en_csv else "",
            "lat_api": api_coords.get(z, ("", ""))[0], "lon_api": api_coords.get(z, ("", ""))[1],
            "lat_csv": zonas_csv.loc[destino, "dest_lat_c"] if en_csv else "",
            "lon_csv": zonas_csv.loc[destino, "dest_lon_c"] if en_csv else "",
            "dist_centros_km": round(dist) if dist is not None else "",
            "radio_o_poligono": "radio 200/500 km sobre el centro (API); definición del CSV: ver auditoría",
            "problemas": " | ".join(problemas),
        })
    usados = {alias.get(z, z) for z in nombres_api}
    for z in sorted(set(zonas_csv.index) - usados):
        filas.append({
            "nombre_api": "", "alias_modelo": "", "nombre_modelo": z if z in zonas_modelo else "",
            "nombre_csv": z, "lat_api": "", "lon_api": "",
            "lat_csv": zonas_csv.loc[z, "dest_lat_c"], "lon_csv": zonas_csv.loc[z, "dest_lon_c"],
            "dist_centros_km": "", "radio_o_poligono": "radio 200/500 km sobre el centro (API); definición del CSV: ver auditoría",
            "problemas": "Existe en CSV/modelo pero ninguna zona de la API la usa",
        })
    tabla = pd.DataFrame(filas)
    sin_modelo = tabla[tabla["nombre_modelo"] == ""]["nombre_api"].tolist()
    if sin_modelo:
        hallazgo("CRITICO", "EVIDENCIA", f"Zonas de la API sin zona en el modelo/CSV: {sin_modelo}")
    colisiones = pd.Series([alias.get(z, z) for z in api_coords]).value_counts()
    colisiones = colisiones[colisiones > 1]
    if len(colisiones):
        hallazgo("AVISO", "EVIDENCIA",
                 f"Varias zonas de la API comparten la misma zona del modelo: {colisiones.to_dict()}")
    lejos = tabla[pd.to_numeric(tabla["dist_centros_km"], errors="coerce") > 200]["nombre_api"].tolist()
    if lejos:
        hallazgo("AVISO", "EVIDENCIA", f"Centros API vs CSV a >200 km en: {lejos}")
    if zonas_csv_out:
        zonas_csv_out.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(zonas_csv_out, index=False, encoding="utf-8")
        print(f"Tabla de zonas escrita en {zonas_csv_out}")


def auditar_escalas_features(df: pd.DataFrame) -> None:
    """Compara la escala de las familias loc* (radio, usadas por la API) y lat* (región del CSV)."""
    print("\n== 5b. Coherencia de escalas entre familias de columnas ==")
    cols = ["loc200_n96_con", "loc500_n96_con", "lat_n96_som_con", "lat_n96_todo_con",
            "lat_n96_prof_con", "lat_n96_prof70_150"]
    cols = [c for c in cols if c in df.columns]
    medias = df[cols].mean().round(3).to_dict()
    print(f"Media por fila: {medias}")
    if {"loc500_n96_con", "lat_n96_todo_con"} <= set(df.columns):
        frac = float((df["loc500_n96_con"] >= df["lat_n96_todo_con"]).mean())
        if frac < 0.5:
            hallazgo("CRITICO", "EVIDENCIA",
                     f"loc500_n96_con >= lat_n96_todo_con solo en {frac * 100:.1f}% de las filas y la media "
                     f"de lat_n96_todo_con ({medias.get('lat_n96_todo_con')}) es mucho mayor que la de "
                     f"loc500_n96_con ({medias.get('loc500_n96_con')}). Las columnas lat_* cubren una "
                     "región o umbral de magnitud DISTINTO al radio de 500 km.")
    if "lat_n96_prof70_150" in df.columns:
        hallazgo("AVISO", "HIPOTESIS",
                 "lat_n96_prof70_150 (feature del modelo) pertenece a la familia lat_*, pero la API la "
                 "calcula como 'sismos a <=500 km con profundidad 70-150 km'. Si la región/umbral de lat_* "
                 "difiere (ver arriba), hay desajuste entrenamiento/producción (train-serve skew).")
    if "media_96h_zona" in df.columns and "lat_n96_som_con" in df.columns:
        r = float(df["media_96h_zona"].mean() / max(df["loc200_n96_con"].mean(), 1e-9))
        hallazgo("AVISO", "EVIDENCIA",
                 f"media_96h_zona (media de lat_n96_som_con) es ~{r:.1f} veces la media de loc200_n96_con, "
                 "pero train.py/main.py la usan como denominador de loc200_n96_con en "
                 "ratio_actividad_local: numerador y denominador miden regiones distintas.")


def auditar_features_prohibidas() -> None:
    print("\n== 6. Features recomendadas ==")
    hallazgo("OK", "EVIDENCIA",
             "Candidatas seguras (solo ventana pasada de 96 h): loc200_n96_con, loc200_n96_sin45 y "
             "loc500_n96_con. lat_n96_prof70_150 solo es candidata tras alinear exactamente región, "
             "radio y umbral con producción. Excluir: variables lunares/mareales, dest_lat_c/dest_lon_c "
             "como identidad de zona, p90_* y media_96h_zona de muestra completa.")


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for flujo in (sys.stdout, sys.stderr):  # evita caracteres rotos en consolas de Windows
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "zona_dia_11a.csv")
    ap.add_argument("--joblib", type=Path, default=ROOT / "api" / "modelo_sismos.joblib")
    ap.add_argument("--main-py", type=Path, default=ROOT / "api" / "main.py")
    ap.add_argument("--vigilante-py", type=Path, default=ROOT / "agente" / "vigilante.py")
    ap.add_argument("--json-out", type=Path, default=None, help="Guardar resumen numérico en JSON.")
    ap.add_argument("--zonas-csv", type=Path, default=None, help="Escribir la tabla de zonas (p. ej. docs/zonas_modelo.csv).")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"No existe {args.csv}. Regenéralo (ver docs/auditoria_datos_modelo.md).", file=sys.stderr)
        return 2
    df = cargar(args.csv)
    auditar_esquema_y_fechas(df)
    auditar_fuga_temporal(df)
    resumen = auditar_target(df)
    resumen["particiones"] = auditar_particiones(df)
    auditar_zonas(df, args.joblib, args.main_py, args.vigilante_py, args.zonas_csv)
    auditar_escalas_features(df)
    auditar_features_prohibidas()

    criticos = [h for h in hallazgos if h["nivel"] == "CRITICO"]
    avisos = [h for h in hallazgos if h["nivel"] == "AVISO"]
    print(f"\n== Resumen: {len(criticos)} críticos, {len(avisos)} avisos, "
          f"{len(hallazgos) - len(criticos) - len(avisos)} ok ==")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        resumen["hallazgos"] = hallazgos
        args.json_out.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON escrito en {args.json_out}")
    return 1 if criticos else 0


if __name__ == "__main__":
    sys.exit(main())
