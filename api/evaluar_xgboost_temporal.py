"""Barrido pequeño de XGBoost con selección temporal antes del test final.

No escribe artefactos. Usa ajuste <=2021, calibración 2022 y validación 2023;
por tanto no selecciona hiperparámetros mirando el test 2024+.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "candidate" / "zona_dia_usgs_v4.csv"
FEATURES = ["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_prof70_150"]
TARGET = "y_M65_7d"

CONFIGURACIONES = {
    "shallow": dict(n_estimators=100, max_depth=1, learning_rate=0.05, min_child_weight=20, reg_lambda=5.0, reg_alpha=0.1),
    "conservador": dict(n_estimators=160, max_depth=2, learning_rate=0.03, min_child_weight=20, reg_lambda=5.0, reg_alpha=0.1),
    "regularizado": dict(n_estimators=200, max_depth=2, learning_rate=0.02, min_child_weight=50, reg_lambda=15.0, reg_alpha=1.0),
    "profundo": dict(n_estimators=100, max_depth=3, learning_rate=0.03, min_child_weight=40, reg_lambda=15.0, reg_alpha=1.0),
}


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def tasas(df: pd.DataFrame, peso: float = 60.0) -> dict[str, float]:
    global_ = float(df[TARGET].mean())
    g = df.groupby("zona", observed=True)[TARGET].agg(["sum", "count"])
    return {z: float((x["sum"] + peso * global_) / (x["count"] + peso)) for z, x in g.iterrows()}


def main() -> int:
    df = pd.read_csv(DATASET)
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    ajuste = df[df["t_eval_utc"] <= "2021-12-24 23:59:59+00:00"]
    calibracion = df[(df["t_eval_utc"] >= "2022-01-01 00:00:00+00:00") &
                     (df["t_eval_utc"] <= "2022-12-24 23:59:59+00:00")]
    validacion = df[(df["t_eval_utc"] >= "2023-01-01 00:00:00+00:00") &
                    (df["t_eval_utc"] <= "2023-12-24 23:59:59+00:00")]
    p_base = validacion["zona"].map(tasas(pd.concat([ajuste, calibracion]))).to_numpy(dtype=float)
    base = metricas(validacion[TARGET].to_numpy(), p_base)
    print("Baseline zona 2023:", {k: round(v, 5) for k, v in base.items()})

    resultados: list[tuple[str, dict[str, float]]] = []
    for nombre, config in CONFIGURACIONES.items():
        modelo = XGBClassifier(
            **config, subsample=0.8, colsample_bytree=1.0, objective="binary:logistic",
            eval_metric="logloss", n_jobs=4, random_state=42,
        )
        modelo.fit(ajuste[FEATURES], ajuste[TARGET])
        calibrado = CalibratedClassifierCV(modelo, method="sigmoid", cv="prefit")
        calibrado.fit(calibracion[FEATURES], calibracion[TARGET])
        resultado = metricas(validacion[TARGET].to_numpy(), calibrado.predict_proba(validacion[FEATURES])[:, 1])
        resultados.append((nombre, resultado))
        print(nombre, {k: round(v, 5) for k, v in resultado.items()},
              "delta_brier", round(resultado["brier"] - base["brier"], 5))

    mejor = min(resultados, key=lambda x: (x[1]["brier"], -x[1]["ap"], -x[1]["auc"]))
    print("Seleccionado por Brier/AP/AUC temporal:", mejor[0], {k: round(v, 5) for k, v in mejor[1].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
