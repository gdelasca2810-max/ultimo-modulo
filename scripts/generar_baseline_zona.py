#!/usr/bin/env python
"""Genera api/baseline_tasa_zona.json: tasa histórica M6.5+ a 7 días por zona, suavizada.

No es un modelo ML: es la tasa de cada zona con un prior hacia la tasa global
(peso equivalente a `--peso-prior` filas) para que ninguna zona quede en 0 %.
Es el mismo estimador que usan las evaluaciones (api/train_temporal.py,
api/evaluar_t3_prior_sismico.py). Sin red, sin secretos.

    python scripts/generar_baseline_zona.py                       # historial completo
    python scripts/generar_baseline_zona.py --hasta 2023-12-31    # solo hasta una fecha
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "zona_dia_11a.csv"
SALIDA = ROOT / "api" / "baseline_tasa_zona.json"
TARGET = "y_M65_7d"


def calcular(df: pd.DataFrame, peso_prior: float) -> dict:
    global_ = float(df[TARGET].mean())
    zonas = {}
    for zona, g in df.groupby("zona"):
        pos, n = int(g[TARGET].sum()), int(len(g))
        zonas[zona] = {"positivos": pos, "filas": n,
                       "tasa": round((pos + peso_prior * global_) / (n + peso_prior), 6)}
    return {"tasa_global": round(global_, 6), "zonas": dict(sorted(zonas.items()))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=SALIDA)
    ap.add_argument("--peso-prior", type=float, default=60.0)
    ap.add_argument("--hasta", default=None, help="Última t_eval_utc a incluir (AAAA-MM-DD); por defecto todo.")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, usecols=["zona", "t_eval_utc", TARGET])
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    if args.hasta:
        df = df[df["t_eval_utc"] <= pd.Timestamp(args.hasta + " 23:59:59", tz="UTC")]
    resultado = calcular(df, args.peso_prior)
    salida = {
        "version": "baseline_tasa_zona_v1",
        "descripcion": "Tasa historica suavizada de M6.5+ a 7 dias por zona. No es un modelo ML.",
        "objetivo": TARGET,
        "periodo": [df["t_eval_utc"].min().strftime("%Y-%m-%d"), df["t_eval_utc"].max().strftime("%Y-%m-%d")],
        "peso_prior": args.peso_prior,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        **resultado,
    }
    args.out.write_text(json.dumps(salida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Escrito {args.out} con {len(salida['zonas'])} zonas; tasa global {salida['tasa_global']}")
    for zona, v in salida["zonas"].items():
        print(f"  {zona:26} positivos={v['positivos']:4d} filas={v['filas']} tasa={v['tasa']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
