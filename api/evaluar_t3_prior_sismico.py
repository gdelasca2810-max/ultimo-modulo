"""Evalúa un modelo jerárquico: tasa histórica por zona + señal sísmica T3.

La tasa es un prior explícito, no una coordenada ni una identidad oculta; el
modelo de señal usa solo conteos sísmicos previos. Se valida con bloques
temporales separados antes de tocar el test 2024+.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "candidate" / "zona_dia_t3_reconstruido.csv"
TARGET = "y_M65_7d"
BASE = ["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_som_con", "lat_n96_som_sin",
        "lat_n96_som45_con", "lat_n96_prof_con", "lat_n96_prof70_150", "lat_n96_todo_con",
        "lat_n24_som_con", "lat_n24_som_sin", "lat_d1_som_con", "lat_d2_som_con", "lat_d3_som_con", "lat_d4_som_con"]


def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["lat_ratio_24_96"] = out["lat_n24_som_con"] / (out["lat_n96_som_con"] + 1.0)
    out["lat_aceleracion_dias"] = out["lat_d1_som_con"] + out["lat_d2_som_con"] - out["lat_d3_som_con"] - out["lat_d4_som_con"]
    out["fraccion_sin_replica"] = out["lat_n96_som_sin"] / (out["lat_n96_som_con"] + 1.0)
    return out, BASE + ["lat_ratio_24_96", "lat_aceleracion_dias", "fraccion_sin_replica"]


def tasas(df: pd.DataFrame, peso: float = 60.0) -> dict[str, float]:
    global_ = float(df[TARGET].mean())
    g = df.groupby("zona", observed=True)[TARGET].agg(["sum", "count"])
    return {z: float((x["sum"] + peso * global_) / (x["count"] + peso)) for z, x in g.iterrows()}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def ejecutar(fit: pd.DataFrame, calibrar: pd.DataFrame, evaluar: pd.DataFrame, features: list[str]) -> tuple[dict, dict]:
    modelo = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.03,
                                        min_samples_leaf=30, subsample=0.8, random_state=42)
    modelo.fit(fit[features], fit[TARGET])
    prior_fit = tasas(fit)
    p_cal_signal = modelo.predict_proba(calibrar[features])[:, 1]
    p_cal_prior = calibrar["zona"].map(prior_fit).to_numpy(dtype=float)
    meta = LogisticRegression(C=0.3, max_iter=1000, random_state=42)
    meta.fit(np.column_stack([logit(p_cal_prior), logit(p_cal_signal)]), calibrar[TARGET])

    prior_eval = tasas(pd.concat([fit, calibrar]))
    p_prior = evaluar["zona"].map(prior_eval).to_numpy(dtype=float)
    p_signal = modelo.predict_proba(evaluar[features])[:, 1]
    p_meta = meta.predict_proba(np.column_stack([logit(p_prior), logit(p_signal)]))[:, 1]
    return metricas(evaluar[TARGET].to_numpy(), p_prior), metricas(evaluar[TARGET].to_numpy(), p_meta)


def main() -> int:
    df, features = preparar(pd.read_csv(DATA))
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    # Selección temporal: fit<=2021, meta-calibración=2022, validación=2023.
    fit_1 = df[df.t_eval_utc <= "2021-12-24 23:59:59+00:00"]
    cal_1 = df[(df.t_eval_utc >= "2022-01-01 00:00:00+00:00") & (df.t_eval_utc <= "2022-12-24 23:59:59+00:00")]
    val = df[(df.t_eval_utc >= "2023-01-01 00:00:00+00:00") & (df.t_eval_utc <= "2023-12-24 23:59:59+00:00")]
    b_val, h_val = ejecutar(fit_1, cal_1, val, features)
    print("Validación 2023 baseline:", {k: round(v, 5) for k, v in b_val.items()})
    print("Validación 2023 prior+sismo:", {k: round(v, 5) for k, v in h_val.items()})

    # Test final: fit<=2022, meta-calibración=2023, test>=2024-01-08.
    fit_2 = df[df.t_eval_utc <= "2022-12-24 23:59:59+00:00"]
    cal_2 = df[(df.t_eval_utc >= "2023-01-01 00:00:00+00:00") & (df.t_eval_utc <= "2023-12-24 23:59:59+00:00")]
    test = df[df.t_eval_utc >= "2024-01-08 00:00:00+00:00"]
    b_test, h_test = ejecutar(fit_2, cal_2, test, features)
    print("Test 2024+ baseline:", {k: round(v, 5) for k, v in b_test.items()})
    print("Test 2024+ prior+sismo:", {k: round(v, 5) for k, v in h_test.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
