import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FEATURES_BASE = [
    "loc200_n96_con",
    "loc200_n96_sin45",
    "loc500_n96_con",
    "lat_n96_prof70_150",
    "media_96h_zona",
]

FEATURES_V2 = [
    "loc200_n96_con",
    "loc200_n96_sin45",
    "loc200_n96_m45plus",
    "loc500_n96_con",
    "lat_n96_prof70_150",
    "media_96h_zona",
    "p90_con_11a",
    "ratio_actividad_local",
    "exceso_p90_local",
    "concentracion_espacial",
]

TARGET = "y_M65_7d"


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURES_BASE + ["p90_con_11a"]:
        out[col] = out[col].fillna(0)

    out["loc200_n96_m45plus"] = (out["loc200_n96_con"] - out["loc200_n96_sin45"]).clip(lower=0)
    out["ratio_actividad_local"] = out["loc200_n96_con"] / (out["media_96h_zona"] + 0.25)
    out["exceso_p90_local"] = out["loc200_n96_con"] - out["p90_con_11a"]
    out["concentracion_espacial"] = out["loc200_n96_con"] / (out["loc500_n96_con"] + 1.0)
    return out


def entrenar():
    df = pd.read_csv("data/zona_dia_11a.csv")
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"])
    df = construir_features(df)

    # Partición temporal estricta con hueco de 7 días igual al horizonte
    train = df[df["t_eval_utc"] <= "2023-12-31 23:59:59"]
    test = df[df["t_eval_utc"] >= "2024-01-08 00:00:00"]

    y_train = train[TARGET]
    y_test = test[TARGET]

    # 1. Evaluación modelo base (5 variables) para comparación objetiva
    modelo_base = GradientBoostingClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.05, random_state=42
    )
    modelo_base.fit(train[FEATURES_BASE], y_train)
    probs_base = modelo_base.predict_proba(test[FEATURES_BASE])[:, 1]
    auc_base = roc_auc_score(y_test, probs_base)
    ap_base = average_precision_score(y_test, probs_base)
    brier_base = brier_score_loss(y_test, probs_base)
    print(
        f"[Base 5 vars] Test (2024-2026) -> AUC: {auc_base:.4f} | "
        f"PR-AUC: {ap_base:.4f} | Brier: {brier_base:.5f}"
    )

    # 2. Modelo v2 con variables sismológicas normalizadas + calibración probabilística
    gb_v2 = GradientBoostingClassifier(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        min_samples_leaf=30,
        random_state=42,
    )
    modelo_v2 = CalibratedClassifierCV(estimator=gb_v2, method="sigmoid", cv=3)
    modelo_v2.fit(train[FEATURES_V2], y_train)

    probs_v2 = modelo_v2.predict_proba(test[FEATURES_V2])[:, 1]
    auc_v2 = roc_auc_score(y_test, probs_v2)
    ap_v2 = average_precision_score(y_test, probs_v2)
    brier_v2 = brier_score_loss(y_test, probs_v2)
    print(
        f"[ML v2 10 vars] Test (2024-2026) -> AUC: {auc_v2:.4f} | "
        f"PR-AUC: {ap_v2:.4f} | Brier: {brier_v2:.5f}"
    )

    tasas_zona = train.groupby("zona")[TARGET].mean().to_dict()
    medias_96h = train.groupby("zona")["media_96h_zona"].first().to_dict()
    p90_zona = train.groupby("zona")["p90_con_11a"].first().to_dict()

    artefacto = {
        "version": "v2_features_11a",
        "modelo": modelo_v2,
        "features": FEATURES_V2,
        "tasas_zona": tasas_zona,
        "medias_96h": medias_96h,
        "p90_zona": p90_zona,
        "metricas_test": {
            "auc": round(float(auc_v2), 4),
            "ap": round(float(ap_v2), 4),
            "brier": round(float(brier_v2), 5),
            "auc_base": round(float(auc_base), 4),
            "ap_base": round(float(ap_base), 4),
        },
    }
    joblib.dump(artefacto, "api/modelo_sismos.joblib")
    print("Modelo v2 guardado en api/modelo_sismos.joblib")


if __name__ == "__main__":
    entrenar()