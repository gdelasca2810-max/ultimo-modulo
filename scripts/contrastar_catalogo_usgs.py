#!/usr/bin/env python
"""Contrasta las features loc_* del CSV contra un catálogo crudo USGS.

Solo lee archivos locales. No demuestra la definición original del dataset: los
eventos USGS pueden haber sido revisados después de generar el CSV. Sí permite
validar o refutar la hipótesis operativa M>=2.5, 96 h y radios de 200/500 km.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con"]


def distancias_km(eventos: pd.DataFrame, zonas: pd.DataFrame) -> np.ndarray:
    """Matriz eventos x zonas de distancias haversine."""
    radio = 6371.0
    lat1 = np.radians(eventos["latitude"].to_numpy())[:, None]
    lon1 = np.radians(eventos["longitude"].to_numpy())[:, None]
    lat2 = np.radians(zonas["dest_lat_c"].to_numpy())[None, :]
    lon2 = np.radians(zonas["dest_lon_c"].to_numpy())[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * radio * np.arcsin(np.sqrt(a))


def resumen(nombre: str, esperado: np.ndarray, calculado: np.ndarray) -> None:
    diferencia = calculado - esperado
    print(
        f"{nombre}: n={len(esperado)} exactos={(diferencia == 0).mean():.2%} "
        f"mae={np.abs(diferencia).mean():.3f} sesgo={diferencia.mean():+.3f} "
        f"max_abs={np.abs(diferencia).max()}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "zona_dia_11a.csv")
    ap.add_argument("--raw", type=Path, required=True, nargs="+",
                    help="Uno o más CSV producidos por descargar_catalogo_usgs.py")
    ap.add_argument("--min-mag", type=float, default=2.5)
    ap.add_argument("--max-fechas", type=int, default=0, help="0 = todas las fechas cubiertas")
    args = ap.parse_args()

    datos = pd.read_csv(args.dataset)
    crudo = pd.concat([pd.read_csv(ruta) for ruta in args.raw], ignore_index=True)
    if "id" in crudo:
        crudo = crudo.drop_duplicates("id", keep="last")
    datos["t_eval_utc"] = pd.to_datetime(datos["t_eval_utc"], utc=True)
    crudo["time_utc"] = pd.to_datetime(crudo["time_utc"], utc=True)
    crudo = crudo[crudo["mag"].ge(args.min_mag)].dropna(subset=["latitude", "longitude", "mag"])

    zonas = datos.groupby("zona", as_index=False)[["dest_lat_c", "dest_lon_c"]].first().sort_values("zona")
    nombres = zonas["zona"].tolist()
    inicio = crudo["time_utc"].min().to_pydatetime() + timedelta(hours=96)
    fin = crudo["time_utc"].max().to_pydatetime()
    datos = datos[(datos["t_eval_utc"] >= inicio) & (datos["t_eval_utc"] <= fin)].copy()
    fechas = sorted(datos["t_eval_utc"].unique())
    if args.max_fechas:
        fechas = fechas[:args.max_fechas]
    if not fechas:
        raise SystemExit("El catálogo no cubre ninguna ventana retrospectiva completa de 96 h.")

    esperado = {feature: [] for feature in FEATURES}
    calculado = {feature: [] for feature in FEATURES}
    for fecha_valor in fechas:
        # pandas.unique devuelve numpy.datetime64; normalizarlo evita una
        # conversión de timedelta deprecada en pandas/numpy recientes.
        fecha = pd.Timestamp(fecha_valor).to_pydatetime()
        fila = datos[datos["t_eval_utc"] == fecha].set_index("zona").reindex(nombres)
        if fila[FEATURES].isna().any().any():
            continue
        ventana = crudo[(crudo["time_utc"] >= fecha - timedelta(hours=96)) & (crudo["time_utc"] < fecha)]
        if len(ventana):
            dist = distancias_km(ventana, zonas)
            m200 = dist <= 200
            m500 = dist <= 500
            m45 = ventana["mag"].to_numpy()[:, None] < 4.5
            valores = {
                "loc200_n96_con": m200.sum(axis=0),
                "loc200_n96_sin45": (m200 & m45).sum(axis=0),
                "loc500_n96_con": m500.sum(axis=0),
            }
        else:
            valores = {feature: np.zeros(len(zonas), dtype=int) for feature in FEATURES}
        for feature in FEATURES:
            esperado[feature].extend(fila[feature].astype(int).to_numpy())
            calculado[feature].extend(valores[feature])

    print(f"Fechas contrastadas: {len(fechas)}; zonas: {len(nombres)}; comparaciones: {len(esperado[FEATURES[0]])}")
    print(f"Catálogo: {crudo['time_utc'].min()} -> {crudo['time_utc'].max()}; min_mag={args.min_mag}")
    for feature in FEATURES:
        resumen(feature, np.asarray(esperado[feature]), np.asarray(calculado[feature]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
