import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = [
    "loc200_n96_con",
    "loc200_n96_sin45",
    "loc500_n96_con",
    "lat_n96_prof70_150",
    "media_96h_zona",
]
TARGET = "y_M65_7d"


def entrenar():
    df = pd.read_csv("zona_dia_11a.csv")
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"])

    # Partición temporal estricta con hueco de 7 días igual al horizonte
    train = df[df["t_eval_utc"] <= "2023-12-31 23:59:59"]
    test = df[df["t_eval_utc"] >= "2024-01-08 00:00:00"]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    modelo = GradientBoostingClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.05, random_state=42
    )
    modelo.fit(X_train, y_train)

    probs = modelo.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    print(f"Validación Test (2024-2026) -> AUC: {auc:.4f} | Average Precision: {ap:.4f}")

    tasas_zona = train.groupby("zona")[TARGET].mean().to_dict()
    medias_96h = train.groupby("zona")["media_96h_zona"].first().to_dict()

    artefacto = {
        "modelo": modelo,
        "features": FEATURES,
        "tasas_zona": tasas_zona,
        "medias_96h": medias_96h,
        "metricas_test": {"auc": round(auc, 4), "ap": round(ap, 4)},
    }
    joblib.dump(artefacto, "api/modelo_sismos.joblib")
    print("Modelo guardado en api/modelo_sismos.joblib")


if __name__ == "__main__":
    entrenar()