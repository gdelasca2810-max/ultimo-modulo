"""Entrenamiento reproducible y temporalmente honesto del riesgo M6.5+ a 7 días.

No sustituye el artefacto servido salvo que se pase explícitamente --output
api/modelo_sismos.joblib. Por defecto genera un candidato para revisión.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "zona_dia_11a.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "modelo_sismos_v3_candidate.joblib"
TARGET = "y_M65_7d"
FEATURES = ["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_prof70_150"]


def metricas(y_true: pd.Series, probabilidades: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, probabilidades)),
        "ap": float(average_precision_score(y_true, probabilidades)),
        "brier": float(brier_score_loss(y_true, probabilidades)),
    }


def redondear_metricas(valores: dict[str, float]) -> dict[str, float]:
    return {nombre: round(valor, 5) for nombre, valor in valores.items()}


def tasas_suavizadas(df: pd.DataFrame, peso_prior: float = 60.0) -> dict[str, float]:
    """Evita tasas observadas cero sin emplear la identidad de zona como feature."""
    tasa_global = float(df[TARGET].mean())
    resumen = df.groupby("zona", observed=True)[TARGET].agg(["sum", "count"])
    return {zona: float((fila["sum"] + peso_prior * tasa_global) / (fila["count"] + peso_prior))
            for zona, fila in resumen.iterrows()}


def particion_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ajuste, calibración y test, separados por huecos de siete días."""
    ajuste = df[df["t_eval_utc"] <= pd.Timestamp("2022-12-24 23:59:59", tz="UTC")].copy()
    calibracion = df[(df["t_eval_utc"] >= pd.Timestamp("2023-01-01", tz="UTC")) &
                     (df["t_eval_utc"] <= pd.Timestamp("2023-12-24 23:59:59", tz="UTC"))].copy()
    test = df[df["t_eval_utc"] >= pd.Timestamp("2024-01-08", tz="UTC")].copy()
    referencia = df[df["t_eval_utc"] <= pd.Timestamp("2023-12-31 23:59:59", tz="UTC")].copy()
    if min(len(ajuste), len(calibracion), len(test)) == 0 or min(ajuste[TARGET].sum(), calibracion[TARGET].sum(), test[TARGET].sum()) == 0:
        raise ValueError("La partición temporal quedó vacía o sin positivos M6.5+.")
    return ajuste, calibracion, test, referencia


def entrenar(output: Path) -> dict:
    df = pd.read_csv(DATA_PATH)
    requeridas = set(FEATURES + [TARGET, "zona", "t_eval_utc"])
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    df[FEATURES] = df[FEATURES].fillna(0).clip(lower=0)
    df[TARGET] = df[TARGET].astype(int)
    ajuste, calibracion, test, referencia = particion_temporal(df)

    base = GradientBoostingClassifier(n_estimators=90, max_depth=2, learning_rate=0.04,
                                      min_samples_leaf=30, subsample=0.8, random_state=42)
    base.fit(ajuste[FEATURES], ajuste[TARGET])
    # cv='prefit': 2023 se dedica a calibrar y no se mezcla con el ajuste 2016--2022.
    modelo = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    modelo.fit(calibracion[FEATURES], calibracion[TARGET])

    probs_modelo = modelo.predict_proba(test[FEATURES])[:, 1]
    tasas = tasas_suavizadas(referencia)
    probs_baseline = test["zona"].map(tasas).to_numpy(dtype=float)
    m_modelo = metricas(test[TARGET], probs_modelo)
    m_baseline = metricas(test[TARGET], probs_baseline)
    sin_local = test["loc200_n96_con"] == 0
    m_sin_local = metricas(test.loc[sin_local, TARGET], probs_modelo[sin_local])

    artefacto = {
        "version": "v3_temporal_calibrated_4features", "modelo": modelo, "features": FEATURES,
        "tasas_zona": tasas,
        # Clave conservada para consumidores antiguos; v3 no la usa como feature.
        "medias_96h": referencia.groupby("zona", observed=True)["loc200_n96_con"].mean().to_dict(),
        "metricas_test": {
            **redondear_metricas(m_modelo),
            "auc_baseline_zona": round(m_baseline["auc"], 5),
            "ap_baseline_zona": round(m_baseline["ap"], 5),
            "brier_baseline_zona": round(m_baseline["brier"], 5),
            "auc_sin_sismos_200km": round(m_sin_local["auc"], 5),
            "ap_sin_sismos_200km": round(m_sin_local["ap"], 5),
            "n_test": int(len(test)), "positivos_test": int(test[TARGET].sum()),
            "n_test_sin_sismos_200km": int(sin_local.sum()),
        },
        "particion": {"ajuste_hasta": "2022-12-24T23:59:59Z", "calibracion": "2023-01-01..2023-12-24",
                       "test_desde": "2024-01-08T00:00:00Z", "hueco_dias": 7},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, output)
    print(f"Candidato guardado: {output}")
    print("Test 2024+ (modelo):", redondear_metricas(m_modelo))
    print("Test 2024+ (tasa zona):", redondear_metricas(m_baseline))
    print("Subgrupo sin <=200 km:", redondear_metricas(m_sin_local), f"n={int(sin_local.sum())}")
    return artefacto


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    entrenar(parser.parse_args().output.resolve())
