#!/usr/bin/env python
"""Contrasta y_M65_7d con eventos USGS M>=6.5 bajo radios explícitos.

Es una prueba de hipótesis: no presupone que el CSV original utilizara un radio
circular. Sus resultados sirven para identificar si una definición operacional
simple puede explicar el target y para documentar discordancias.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def distancias_km(eventos: pd.DataFrame, lat: float, lon: float) -> np.ndarray:
    r = 6371.0
    lat1 = np.radians(eventos["latitude"].to_numpy(dtype=float))
    lon1 = np.radians(eventos["longitude"].to_numpy(dtype=float))
    lat2, lon2 = np.radians([lat, lon])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def target_circular(tiempos_evento: np.ndarray, evaluaciones: np.ndarray, dias: int) -> np.ndarray:
    return (
        np.searchsorted(tiempos_evento, evaluaciones + np.timedelta64(dias, "D"), side="left")
        - np.searchsorted(tiempos_evento, evaluaciones, side="left")
        > 0
    ).astype(int)


def metricas(y: np.ndarray, pred: np.ndarray) -> dict[str, int | float]:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "exactitud": (tp + tn) / len(y),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "zona_dia_11a.csv")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "usgs_catalogo_m25_2015_2026.csv")
    ap.add_argument("--target", default="y_M65_7d")
    ap.add_argument("--mag", type=float, default=6.5)
    ap.add_argument("--horizonte-dias", type=int, default=7)
    ap.add_argument("--radios", type=float, nargs="+", default=[200, 300, 500, 750, 1000, 1500, 2000])
    args = ap.parse_args()

    datos = pd.read_csv(args.dataset)
    eventos = pd.read_csv(args.raw)
    datos["t_eval_utc"] = pd.to_datetime(datos["t_eval_utc"], utc=True)
    eventos["time_utc"] = pd.to_datetime(eventos["time_utc"], utc=True)
    eventos = eventos[eventos["mag"].ge(args.mag)].dropna(subset=["time_utc", "latitude", "longitude"])
    eventos = eventos.drop_duplicates("id", keep="last").sort_values("time_utc")
    print(f"Eventos M>={args.mag}: {len(eventos)}; radios km: {args.radios}")

    acumulado = {radio: {"y": [], "pred": []} for radio in args.radios}
    por_zona: dict[float, list[tuple[str, dict[str, int | float]]]] = {radio: [] for radio in args.radios}
    distancias_positivos: list[tuple[str, int, float, float, float, int]] = []
    for zona, grupo in datos.groupby("zona", sort=True):
        grupo = grupo.sort_values("t_eval_utc")
        lat, lon = grupo[["dest_lat_c", "dest_lon_c"]].iloc[0]
        y = grupo[args.target].to_numpy(dtype=int)
        evaluaciones = grupo["t_eval_utc"].to_numpy(dtype="datetime64[ns]")
        distancia = distancias_km(eventos, float(lat), float(lon))
        distancias_y1: list[float] = []
        sin_evento_crudo = 0
        for fecha in evaluaciones[y == 1]:
            dentro = (eventos["time_utc"].to_numpy(dtype="datetime64[ns]") >= fecha) & (
                eventos["time_utc"].to_numpy(dtype="datetime64[ns]") < fecha + np.timedelta64(args.horizonte_dias, "D")
            )
            if dentro.any():
                distancias_y1.append(float(distancia[dentro].min()))
            else:
                sin_evento_crudo += 1
        if distancias_y1:
            distancias_positivos.append((zona, len(distancias_y1), float(np.median(distancias_y1)),
                                         float(np.quantile(distancias_y1, 0.95)), max(distancias_y1), sin_evento_crudo))
        for radio in args.radios:
            tiempos = eventos.loc[distancia <= radio, "time_utc"].to_numpy(dtype="datetime64[ns]")
            pred = target_circular(tiempos, evaluaciones, args.horizonte_dias)
            acumulado[radio]["y"].extend(y)
            acumulado[radio]["pred"].extend(pred)
            por_zona[radio].append((zona, metricas(y, pred)))

    print("\nGlobal:")
    for radio in args.radios:
        resultado = metricas(np.asarray(acumulado[radio]["y"]), np.asarray(acumulado[radio]["pred"]))
        print(f"radio={radio:>6.0f} km | {resultado}")
    mejor = max(args.radios, key=lambda radio: metricas(
        np.asarray(acumulado[radio]["y"]), np.asarray(acumulado[radio]["pred"])
    )["precision"])
    print(f"\nMayor precision global: radio={mejor:.0f} km")
    print("Por zona para ese radio:")
    for zona, resultado in por_zona[mejor]:
        print(f"{zona}: {resultado}")
    print("\nDistancia al M>=6.5 crudo más cercano durante la ventana de filas y=1:")
    print("zona | n_con_evento | mediana_km | p95_km | max_km | y1_sin_evento_crudo")
    for fila in distancias_positivos:
        print(f"{fila[0]} | {fila[1]} | {fila[2]:.0f} | {fila[3]:.0f} | {fila[4]:.0f} | {fila[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
