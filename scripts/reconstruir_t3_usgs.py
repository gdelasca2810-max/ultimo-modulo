#!/usr/bin/env python
"""Reconstruye las columnas sísmicas T3 desde ComCat con el generador recuperado.

Replica las reglas históricas: catálogo M>=4.0, cajas laterales/destino,
declustering Gardner--Knopoff, conteos [t-96h,t) y targets [t,t+h). Excluye
intencionadamente lunares y estadísticos globales con fuga. No toca producción.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from features_t3 import contar, decluster_gk, mascaras_zona, centroide  # noqa: E402  (misma lógica que /vigilar)
RAW = ROOT / "data" / "raw" / "usgs_catalogo_m25_2015_2026.csv"
REFERENCIA = ROOT / "data" / "zona_dia_11a.csv"
ZONAS = ROOT / "config" / "zonas_t3_historicas.json"
OUT = ROOT / "data" / "candidate" / "zona_dia_t3_reconstruido.csv"
W96, W24 = 96 * 3600, 24 * 3600


INICIO_HISTORICO, FIN_HISTORICO = "2016-01-05 00:37", "2026-09-26 00:37"


def calendario_evaluacion(referencia_path: Path | None, inicio: str = INICIO_HISTORICO, fin: str = FIN_HISTORICO) -> np.ndarray:
    """Instantes t_eval (epoch s, UTC). Usa el CSV histórico si existe; si no, un calendario diario a las 00:37 UTC."""
    if referencia_path is not None and Path(referencia_path).exists():
        ref = pd.read_csv(referencia_path, usecols=["zona", "t_eval_utc"])
        ref["t_eval_utc"] = pd.to_datetime(ref["t_eval_utc"], utc=True)
        fechas = ref["t_eval_utc"].drop_duplicates().dt.tz_convert(None)
    else:
        fechas = pd.Series(pd.date_range(inicio, fin, freq="D"))
    return np.sort(fechas.to_numpy(dtype="datetime64[s]").astype(np.int64))


def reconstruir(raw_path: Path, referencia_path: Path | None, zonas_path: Path,
                inicio: str = INICIO_HISTORICO, fin: str = FIN_HISTORICO) -> tuple[pd.DataFrame, dict]:
    evaluaciones = calendario_evaluacion(referencia_path, inicio, fin)
    zonas = json.loads(zonas_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(raw_path)
    raw["time_utc"] = pd.to_datetime(raw["time_utc"], utc=True)
    if "type" in raw.columns:  # el generador histórico usa solo eventtype=earthquake
        raw = raw[raw["type"] == "earthquake"]
    raw = raw[(raw["mag"] >= 4.0) & raw["latitude"].notna() & raw["longitude"].notna() & raw["depth_km"].notna()]
    raw = raw.drop_duplicates("id", keep="last").sort_values("time_utc").reset_index(drop=True)
    ts = raw["time_utc"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    lat, lon = raw["latitude"].to_numpy(), raw["longitude"].to_numpy()
    dep, mag = raw["depth_km"].to_numpy(), raw["mag"].to_numpy()
    aft = decluster_gk(ts, lat, lon, mag)
    filas: list[pd.DataFrame] = []
    for zona, definicion in zonas.items():
        clat, clon = centroide(definicion["dest"])
        masc = mascaras_zona(definicion, lat, lon, dep, mag, aft)

        def serie(mascara: np.ndarray) -> np.ndarray:
            return ts[mascara]

        som, prof, prof70 = masc["som"], masc["prof"], masc["prof70"]
        local, local_sin, regional, m_dest = masc["local"], masc["local_sin"], masc["regional"], masc["dest"]
        ini = evaluaciones - W96
        datos = {
            "zona": zona,
            "t_eval_utc": pd.to_datetime(evaluaciones, unit="s", utc=True),
            "dest_lat_c": clat, "dest_lon_c": clon,
            "lat_n96_som_con": contar(serie(som), ini, evaluaciones),
            "lat_n96_som_sin": contar(serie(som & ~aft), ini, evaluaciones),
            "lat_n96_prof_con": contar(serie(prof), ini, evaluaciones),
            "lat_n96_prof70_150": contar(serie(prof70), ini, evaluaciones),
            "lat_n96_todo_con": contar(serie(masc["lat_todo"]), ini, evaluaciones),
            "lat_n24_som_con": contar(serie(som), evaluaciones - W24, evaluaciones),
            "lat_n24_som_sin": contar(serie(som & ~aft), evaluaciones - W24, evaluaciones),
            "loc200_n96_con": contar(serie(local), ini, evaluaciones),
            "loc200_n96_sin45": contar(serie(local_sin), ini, evaluaciones),
            "loc500_n96_con": contar(serie(regional), ini, evaluaciones),
        }
        for i in range(4):
            a = evaluaciones - W96 + W24 * i
            datos[f"lat_d{4 - i}_som_con"] = contar(serie(som), a, a + W24)
        for umbral, etiqueta in ((5.5, "M55"), (6.0, "M60"), (6.5, "M65")):
            objetivo = serie(m_dest & (mag >= umbral) & ~aft)
            for horas, horizonte in ((48, "48h"), (168, "7d"), (720, "30d")):
                datos[f"y_{etiqueta}_{horizonte}"] = (contar(objetivo, evaluaciones, evaluaciones + horas * 3600) > 0).astype(int)
        filas.append(pd.DataFrame(datos))
    salida = pd.concat(filas, ignore_index=True).sort_values(["t_eval_utc", "zona"]).reset_index(drop=True)
    meta = {"raw_rows_m4": int(len(raw)), "replicas_gk": int(aft.sum()), "zonas": len(zonas),
            "filas": int(len(salida)), "positivos_m65_7d": int(salida["y_M65_7d"].sum()),
            "nota": "Catálogo ComCat descargado posteriormente: discrepancias con el CSV original pueden reflejar revisiones."}
    return salida, meta


def comparar(reconstruido: pd.DataFrame, referencia_path: Path) -> None:
    original = pd.read_csv(referencia_path)
    original["t_eval_utc"] = pd.to_datetime(original["t_eval_utc"], utc=True)
    comunes = [c for c in reconstruido.columns if c in original.columns and c not in {"zona", "t_eval_utc", "dest_lat_c", "dest_lon_c"}]
    a = original.set_index(["zona", "t_eval_utc"])[comunes].sort_index()
    b = reconstruido.set_index(["zona", "t_eval_utc"])[comunes].sort_index()
    print("Comparación con zona_dia_11a.csv (exactitud / MAE):")
    for columna in comunes:
        delta = b[columna].to_numpy() - a[columna].to_numpy()
        print(f"  {columna:24s} {(delta == 0).mean():6.2%} / {np.abs(delta).mean():.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--referencia", type=Path, default=REFERENCIA)
    ap.add_argument("--zonas", type=Path, default=ZONAS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--inicio", default=INICIO_HISTORICO, help="Solo si no existe --referencia")
    ap.add_argument("--fin", default=FIN_HISTORICO, help="Solo si no existe --referencia")
    args = ap.parse_args()
    reconstruido, meta = reconstruir(args.raw, args.referencia, args.zonas, args.inicio, args.fin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reconstruido.to_csv(args.out, index=False)
    args.out.with_suffix(args.out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Dataset reconstruido: {args.out} | {meta}")
    if args.referencia.exists():
        comparar(reconstruido, args.referencia)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
