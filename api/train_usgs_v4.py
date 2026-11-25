"""Entrena un candidato v4 con dataset USGS de semántica idéntica a la API.

El script no sustituye api/modelo_sismos.joblib por defecto. La calibración usa
solo 2023 y el test empieza tras un hueco de siete días en 2024.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DATASET_DEFAULT = ROOT / "data" / "candidate" / "zona_dia_usgs_v4.csv"
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "modelo_sismos_v4_candidate.joblib"
TARGET = "y_M65_7d"
FEATURES = ["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_prof70_150"]


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def tasas_suavizadas(df: pd.DataFrame, peso_prior: float = 60.0) -> dict[str, float]:
    tasa_global = float(df[TARGET].mean())
    grupos = df.groupby("zona", observed=True)[TARGET].agg(["sum", "count"])
    return {
        zona: float((fila["sum"] + peso_prior * tasa_global) / (fila["count"] + peso_prior))
        for zona, fila in grupos.iterrows()
    }


def bootstrap_bloques(
    y: np.ndarray, p_modelo: np.ndarray, p_base: np.ndarray, fechas: pd.Series, repeticiones: int, semilla: int
) -> dict[str, list[float]]:
    """IC por bloques mensuales globales, no por filas correlacionadas individuales."""
    bloques = fechas.dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy()
    unicos = np.unique(bloques)
    indices = {bloque: np.flatnonzero(bloques == bloque) for bloque in unicos}
    rng = np.random.default_rng(semilla)
    diferencias: dict[str, list[float]] = {"auc": [], "ap": [], "brier": []}
    for _ in range(repeticiones):
        muestra = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([indices[bloque] for bloque in muestra])
        if np.unique(y[idx]).size < 2:
            continue
        modelo = metricas(y[idx], p_modelo[idx])
        base = metricas(y[idx], p_base[idx])
        for nombre in diferencias:
            diferencias[nombre].append(modelo[nombre] - base[nombre])
    return {
        nombre: [float(np.quantile(valores, 0.025)), float(np.quantile(valores, 0.975))]
        for nombre, valores in diferencias.items() if valores
    }


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as flujo:
        for bloque in iter(lambda: flujo.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def crear_estimador(nombre: str):
    if nombre == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=90, max_depth=2, learning_rate=0.04, min_samples_leaf=30,
            subsample=0.8, random_state=42,
        )
    if nombre == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError("Instale xgboost para usar --modelo xgboost") from e
        # Configuración conservadora para ~1-2% de positivos; sin reponderar
        # clases, para preservar la interpretación probabilística antes de calibrar.
        return XGBClassifier(
            n_estimators=160, max_depth=2, learning_rate=0.03,
            min_child_weight=20, subsample=0.8, colsample_bytree=1.0,
            reg_lambda=5.0, reg_alpha=0.1, objective="binary:logistic",
            eval_metric="logloss", n_jobs=4, random_state=42,
        )
    raise ValueError(f"Modelo no soportado: {nombre}")


def entrenar(dataset_path: Path, output: Path, bootstrap: int, nombre_modelo: str) -> dict:
    df = pd.read_csv(dataset_path)
    faltantes = set(FEATURES + [TARGET, "zona", "t_eval_utc"]) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    df[FEATURES] = df[FEATURES].fillna(0).clip(lower=0)
    df[TARGET] = df[TARGET].astype(int)

    ajuste = df[df["t_eval_utc"] <= "2022-12-24 23:59:59+00:00"].copy()
    calibracion = df[(df["t_eval_utc"] >= "2023-01-01 00:00:00+00:00") &
                     (df["t_eval_utc"] <= "2023-12-24 23:59:59+00:00")].copy()
    referencia = df[df["t_eval_utc"] <= "2023-12-31 23:59:59+00:00"].copy()
    test = df[df["t_eval_utc"] >= "2024-01-08 00:00:00+00:00"].copy()
    for nombre, bloque in {"ajuste": ajuste, "calibracion": calibracion, "test": test}.items():
        if not len(bloque) or bloque[TARGET].sum() == 0:
            raise ValueError(f"Bloque temporal inválido: {nombre}")

    base = crear_estimador(nombre_modelo)
    base.fit(ajuste[FEATURES], ajuste[TARGET])
    modelo = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    modelo.fit(calibracion[FEATURES], calibracion[TARGET])

    p_modelo = modelo.predict_proba(test[FEATURES])[:, 1]
    tasas = tasas_suavizadas(referencia)
    p_base = test["zona"].map(tasas).to_numpy(dtype=float)
    m_modelo, m_base = metricas(test[TARGET].to_numpy(), p_modelo), metricas(test[TARGET].to_numpy(), p_base)
    ic = bootstrap_bloques(test[TARGET].to_numpy(), p_modelo, p_base, test["t_eval_utc"], bootstrap, 42)

    artefacto = {
        "version": f"v4_usgs_{nombre_modelo}_4features_candidate",
        "modelo": modelo,
        "features": FEATURES,
        "tasas_zona": tasas,
        "medias_96h": referencia.groupby("zona", observed=True)["loc200_n96_con"].mean().to_dict(),
        "metricas_test": {
            **{clave: round(valor, 5) for clave, valor in m_modelo.items()},
            **{f"{clave}_baseline_zona": round(valor, 5) for clave, valor in m_base.items()},
            "delta_vs_baseline_ic95": {clave: [round(x, 5) for x in intervalo] for clave, intervalo in ic.items()},
            "n_test": int(len(test)), "positivos_test": int(test[TARGET].sum()),
        },
        "particion": {
            "ajuste_hasta": "2022-12-24T23:59:59Z",
            "calibracion": "2023-01-01..2023-12-24",
            "test_desde": "2024-01-08T00:00:00Z", "hueco_dias": 7,
        },
        "dataset": {
            "path": str(dataset_path), "sha256": sha256(dataset_path),
            "target": "M>=6.5 a <=500 km en los siete días siguientes",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, output)
    print("Modelo:", {clave: round(valor, 5) for clave, valor in m_modelo.items()})
    print("Baseline zona:", {clave: round(valor, 5) for clave, valor in m_base.items()})
    print("IC95 delta modelo-base por bloque mensual:", artefacto["metricas_test"]["delta_vs_baseline_ic95"])
    print(f"Candidato guardado: {output}")
    return artefacto


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--modelo", choices=["gradient_boosting", "xgboost"], default="gradient_boosting")
    argumentos = parser.parse_args()
    entrenar(argumentos.dataset.resolve(), argumentos.output.resolve(), argumentos.bootstrap, argumentos.modelo)
