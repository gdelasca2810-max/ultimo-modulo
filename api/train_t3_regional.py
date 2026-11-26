"""Entrenamiento y evaluación del Modelo v8 Regionalizado (Refinado).

Combina la señal global sólida del modelo v7 (escalones someros locales que capturan
enjambres en cualquier parte del mundo) con escalones de refuerzo regionalizados:
- Global: escalones someros locales (sin45 >= 3, 5, 8 y con >= 8).
- Refuerzo R1 (Arcos someros: Japón, Taiwán, México, Sandwich, Costa O EEUU): loc500 >= 12.
- Refuerzo R2 (Subducción Costa: Chile, Perú, Alaska): loc500 >= 8.
- Refuerzo R3 (Losa Profunda: Tonga, Vanuatu, Indonesia, PNG): loc500 >= 8.

Evaluación temporal estricta: train <= 2023, test >= 2024.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "candidate" / "zona_dia_t3_reconstruido.csv"
MODELO_V7 = ROOT / "api" / "modelo_sismos_v7.joblib"
SALIDA_V8 = ROOT / "api" / "modelo_sismos_v8.joblib"
TARGET = "y_M65_7d"

REGIMENES_ZONAS = {
    "R1_Arcos_Someros": [
        "Japon NE", "Taiwan", "Mexico/Centroam", "Sandwich del Sur", "Costa O EEUU"
    ],
    "R2_Subduccion_Costa": [
        "Chile/Peru (Lima-Valpo)", "Chile Central/Sur", "Alaska/Aleutianas"
    ],
    "R3_Losa_Profunda": [
        "Tonga/Kermadec/NZ", "Salomon/Vanuatu", "Indonesia Sonda", "Indonesia Banda/Timor", "PNG/Nueva Bretana"
    ],
    "R4_Fallas_Baja_Tasa": [
        "Venezuela/Caribe", "Mediterraneo", "Japon Nankai/Kyushu", "Filipinas Luzon", "Antillas/Caribe", "Filipinas Mindanao"
    ]
}

def obtener_regimen(zona: str) -> str:
    for reg, zonas in REGIMENES_ZONAS.items():
        if zona in zonas:
            return reg
    return "R4_Fallas_Baja_Tasa"

# Escalones globales v7 + refuerzos regionales
ESCALONES_GLOBALES = [
    ("loc200_n96_sin45", 3),
    ("loc200_n96_sin45", 5),
    ("loc200_n96_sin45", 8),
    ("loc200_n96_con", 8),
]

REFUERZOS_REGIONALES = [
    ("R1_Arcos_Someros", "loc500_n96_con", 12),
    ("R2_Subduccion_Costa", "loc500_n96_con", 8),
    ("R3_Losa_Profunda", "loc500_n96_con", 8),
]

PESO_PRIOR = 60.0

def calcular_tasas_suavizadas(df_fit: pd.DataFrame) -> dict[str, float]:
    g = float(df_fit[TARGET].mean())
    r = df_fit.groupby("zona")[TARGET].agg(["sum", "count"])
    return {z: float((x["sum"] + PESO_PRIOR * g) / (x["count"] + PESO_PRIOR)) for z, x in r.iterrows()}

def construir_matriz_diseno(df: pd.DataFrame) -> np.ndarray:
    columnas = []
    # 1. Escalones globales
    for col, umbral in ESCALONES_GLOBALES:
        columnas.append((df[col].to_numpy(dtype=float) >= umbral).astype(float))
    
    # 2. Refuerzos por régimen
    regimenes_col = df["zona"].map(obtener_regimen).to_numpy()
    for reg, col, umbral in REFUERZOS_REGIONALES:
        masc = (regimenes_col == reg).astype(float)
        val = (df[col].to_numpy(dtype=float) >= umbral).astype(float)
        columnas.append(val * masc)
        
    return np.column_stack(columnas)

class ModeloRegional:
    def __init__(self, lam: float = 5.0):
        self.lam = lam
        self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray):
        n_feats = X.shape[1]
        def objetivo(theta):
            w, b = theta[:-1], theta[-1]
            eta = offset + b + X @ w
            p = expit(eta)
            ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            g = p - y
            loss = -ll + 0.5 * self.lam * np.sum(w ** 2)
            grad = np.concatenate([X.T @ g + self.lam * w, [g.sum()]])
            return loss, grad

        bounds = [(0, None)] * n_feats + [(None, None)]
        init = np.zeros(n_feats + 1)
        res = minimize(objetivo, init, jac=True, method="L-BFGS-B", bounds=bounds)
        self.w = res.x[:-1]
        self.b = float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return expit(offset + self.b + X @ self.w)

def main():
    print("Cargando dataset para entrenamiento y evaluación...")
    df = pd.read_csv(DATA)
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    ts = lambda s: pd.Timestamp(s, tz="UTC")

    # Partición temporal estricta
    corte_train = ts("2023-12-24 23:59:59")
    corte_test = ts("2024-01-08 00:00:00")

    df_train = df[df["t_eval_utc"] <= corte_train].copy()
    df_test = df[df["t_eval_utc"] >= corte_test].copy()

    y_train = df_train[TARGET].to_numpy(float)
    y_test = df_test[TARGET].to_numpy(float)

    prior = calcular_tasas_suavizadas(df_train)
    p_base_test = df_test["zona"].map(prior).to_numpy(float)
    off_train = logit(df_train["zona"].map(prior).to_numpy(float))
    off_test = logit(p_base_test)

    # Cargar modelo v7
    import joblib
    art_v7 = joblib.load(MODELO_V7)
    escalones_v7 = art_v7.get("escalones", [])
    b_v7 = float(art_v7.get("sesgo", 0.0))
    w_v7 = [float(e["peso"]) for e in escalones_v7]
    
    X_v7_test = np.column_stack([(df_test[e["feature"]].to_numpy(float) >= e["umbral"]).astype(float) for e in escalones_v7])
    eta_v7 = off_test + b_v7 + X_v7_test @ np.array(w_v7)
    p_v7_test = expit(eta_v7)

    # Entrenar v8
    X_train_v8 = construir_matriz_diseno(df_train)
    X_test_v8 = construir_matriz_diseno(df_test)

    mod_v8 = ModeloRegional(lam=5.0).fit(X_train_v8, y_train, off_train)
    p_v8_test = mod_v8.predict_proba(X_test_v8, off_test)

    auc_base = roc_auc_score(y_test, p_base_test)
    ap_base = average_precision_score(y_test, p_base_test)
    brier_base = brier_score_loss(y_test, p_base_test)

    auc_v7 = roc_auc_score(y_test, p_v7_test)
    ap_v7 = average_precision_score(y_test, p_v7_test)
    brier_v7 = brier_score_loss(y_test, p_v7_test)

    auc_v8 = roc_auc_score(y_test, p_v8_test)
    ap_v8 = average_precision_score(y_test, p_v8_test)
    brier_v8 = brier_score_loss(y_test, p_v8_test)

    print("\n" + "="*70)
    print("COMPARATIVA DE MÉTRICAS EN TEST (2024+): 19,860 filas, 299 positivos")
    print("="*70)
    print(f"{'Métrica':<12} | {'Baseline':<12} | {'Modelo v7':<12} | {'Modelo v8 (Regional)':<12}")
    print("-"*70)
    print(f"{'ROC-AUC':<12} | {auc_base:<12.5f} | {auc_v7:<12.5f} | {auc_v8:<12.5f}")
    print(f"{'PR-AUC (AP)':<12} | {ap_base:<12.5f} | {ap_v7:<12.5f} | {ap_v8:<12.5f}")
    print(f"{'Brier Score':<12} | {brier_base:<12.5f} | {brier_v7:<12.5f} | {brier_v8:<12.5f}")
    print("="*70)

    # Ajuste de producción con todo el historial (2016-2026)
    prior_prod = calcular_tasas_suavizadas(df)
    off_prod = logit(df["zona"].map(prior_prod).to_numpy(float))
    X_prod_v8 = construir_matriz_diseno(df)
    mod_v8_prod = ModeloRegional(lam=5.0).fit(X_prod_v8, df[TARGET].to_numpy(float), off_prod)

    nombres = [f"GLOBAL::{col}>={u}" for col, u in ESCALONES_GLOBALES]
    nombres += [f"{reg}::{col}>={u}" for reg, col, u in REFUERZOS_REGIONALES]

    pesos_dict = dict(zip(nombres, [round(float(w), 4) for w in mod_v8_prod.w]))

    artefacto_v8 = {
        "version": "v8_t3_regional_multiplicativo",
        "descripcion": "Modelo multiplicativo global con refuerzos regionales por regimen tectonico",
        "regimenes_zonas": REGIMENES_ZONAS,
        "escalones_globales": ESCALONES_GLOBALES,
        "refuerzos_regionales": REFUERZOS_REGIONALES,
        "nombres_escalones": nombres,
        "pesos": [float(w) for w in mod_v8_prod.w],
        "pesos_evaluados": [float(w) for w in mod_v8.w],
        "sesgo": mod_v8_prod.b,
        "sesgo_evaluado": mod_v8.b,
        "tasas_zona": prior_prod,
        "metricas_test_2024": {
            "roc_auc": round(auc_v8, 5),
            "pr_auc": round(ap_v8, 5),
            "brier": round(brier_v8, 5)
        }
    }

    joblib.dump(artefacto_v8, SALIDA_V8)
    print(f"\nArtefacto guardado exitosamente en: {SALIDA_V8}")
    print("Pesos calculados por escalón (modelo desplegado):")
    for k, v in pesos_dict.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
