"""Evalúa modelos con features sísmicas T3 ampliadas y partición temporal limpia.

Usa solo conteos sísmicos observables antes de t_eval. Excluye coordenadas,
estadísticos globales/P90, flags derivados y variables lunares.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "candidate" / "zona_dia_t3_reconstruido.csv"
TARGET = "y_M65_7d"
BASE = [
    "loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con",
    "lat_n96_som_con", "lat_n96_som_sin", "lat_n96_som45_con",
    "lat_n96_prof_con", "lat_n96_prof70_150", "lat_n96_todo_con",
    "lat_n24_som_con", "lat_n24_som_sin",
    "lat_d1_som_con", "lat_d2_som_con", "lat_d3_som_con", "lat_d4_som_con",
]


def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lat_ratio_24_96"] = out["lat_n24_som_con"] / (out["lat_n96_som_con"] + 1.0)
    out["lat_aceleracion_dias"] = (out["lat_d1_som_con"] + out["lat_d2_som_con"]) - (
        out["lat_d3_som_con"] + out["lat_d4_som_con"]
    )
    out["fraccion_sin_replica"] = out["lat_n96_som_sin"] / (out["lat_n96_som_con"] + 1.0)
    return out


FEATURES = BASE + ["lat_ratio_24_96", "lat_aceleracion_dias", "fraccion_sin_replica"]


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def tasas(df: pd.DataFrame, peso: float = 60) -> dict[str, float]:
    global_ = float(df[TARGET].mean())
    g = df.groupby("zona", observed=True)[TARGET].agg(["sum", "count"])
    return {z: float((x["sum"] + peso * global_) / (x["count"] + peso)) for z, x in g.iterrows()}


def modelos() -> dict[str, object]:
    return {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.03, min_samples_leaf=30, subsample=0.8, random_state=42),
        "hist_gradient": HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=7, learning_rate=0.04, l2_regularization=5.0, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=180, max_depth=2, learning_rate=0.03, min_child_weight=25,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, reg_alpha=0.2,
            objective="binary:logistic", eval_metric="logloss", n_jobs=4, random_state=42),
    }


def main() -> int:
    df = enriquecer(pd.read_csv(DATASET))
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    ajuste = df[df["t_eval_utc"] <= "2022-12-24 23:59:59+00:00"]
    calibracion = df[(df["t_eval_utc"] >= "2023-01-01 00:00:00+00:00") &
                     (df["t_eval_utc"] <= "2023-12-24 23:59:59+00:00")]
    referencia = df[df["t_eval_utc"] <= "2023-12-31 23:59:59+00:00"]
    test = df[df["t_eval_utc"] >= "2024-01-08 00:00:00+00:00"]
    p_base = test["zona"].map(tasas(referencia)).to_numpy(dtype=float)
    print("Features:", len(FEATURES), FEATURES)
    print("Baseline:", {k: round(v, 5) for k, v in metricas(test[TARGET], p_base).items()})
    for nombre, modelo in modelos().items():
        modelo.fit(ajuste[FEATURES], ajuste[TARGET])
        calibrado = CalibratedClassifierCV(modelo, method="sigmoid", cv="prefit")
        calibrado.fit(calibracion[FEATURES], calibracion[TARGET])
        p = calibrado.predict_proba(test[FEATURES])[:, 1]
        print(nombre + ":", {k: round(v, 5) for k, v in metricas(test[TARGET], p).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
