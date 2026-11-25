"""Modelo multiplicativo: p = tasa_zona x factor(sismos recientes), con features T3 reproducibles en vivo.

    logit(p) = logit(prior_zona) + b + w . x      (prior_zona fijo como offset, no se aprende)

Solo usa conteos sísmicos previos (loc200/loc500/prof70_150 definidos como T3); NO usa identidad de zona
más que a través del prior estadístico, ni lunares, ni p90/media. Regularización L2 elegida con validación
2023; el test 2024+ no se usa para elegir nada. Evalúa con docs/criterios (api/criterios_despliegue.py).

    python api/train_t3_multiplicativo.py                 # evalúa y escribe el candidato
    python api/train_t3_multiplicativo.py --solo-evaluar  # sin escribir artefacto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import criterios_despliegue as cd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "candidate" / "zona_dia_t3_reconstruido.csv"
VERSION = "v7_t3_m40_r300_multiplicativo"
SALIDA = Path(__file__).resolve().parent / "modelo_sismos_v7.joblib"
TARGET = "y_M65_7d"
# Features reproducibles por /vigilar con la misma definición (api/features_t3.py).
FEATURES = ["loc200_n96_sin45", "loc200_n96_con"]
# Escalones fijados con datos de entrenamiento (no tuneados con el test): la señal medida está en la cola de actividad
# local, no en una relación lineal. Cada columna es un indicador 0/1 con coeficiente >= 0.
# Umbrales fijados mirando SOLO el entrenamiento (<=2023): razón observada/esperada 1,9x (3-4), 6,8x (5-7),
# 38x (8+) para sin45 (M4,0+ sin réplicas) y 2,2x (8-11) / 5,6x (12+) para con.
ESCALONES = [("loc200_n96_sin45", 3), ("loc200_n96_sin45", 5), ("loc200_n96_sin45", 8), ("loc200_n96_con", 8)]
PESO_PRIOR = 60.0
LAMBDAS = [0.01, 0.1, 1.0, 5.0, 20.0]


def transformar(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([(df[c].to_numpy(dtype=float) >= u).astype(float) for c, u in ESCALONES])


def tasas(df: pd.DataFrame, peso: float = PESO_PRIOR) -> dict[str, float]:
    g = float(df[TARGET].mean())
    r = df.groupby("zona")[TARGET].agg(["sum", "count"])
    return {z: float((x["sum"] + peso * g) / (x["count"] + peso)) for z, x in r.iterrows()}


class Multiplicativo:
    def __init__(self, lam: float):
        self.lam, self.w, self.b, self.mu, self.sd = lam, None, 0.0, None, None

    def _z(self, X):
        return (X - self.mu) / self.sd

    def fit(self, X, y, offset):
        self.mu, self.sd = np.zeros(X.shape[1]), np.ones(X.shape[1])  # indicadores: sin estandarizar
        Z = self._z(X)

        def f(theta):
            w, b = theta[:-1], theta[-1]
            eta = offset + b + Z @ w
            p = expit(eta)
            ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            g = p - y
            return -ll + 0.5 * self.lam * np.sum(w ** 2), np.concatenate([Z.T @ g + self.lam * w, [g.sum()]])

        r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B",
                     bounds=[(0, None)] * Z.shape[1] + [(None, None)])  # factor >= 1 en la cola
        self.w, self.b = r.x[:-1], float(r.x[-1])
        return self

    def predict(self, X, offset):
        return expit(offset + self.b + self._z(X) @ self.w)


def evaluar(fit: pd.DataFrame, test: pd.DataFrame, lam: float):
    prior = tasas(fit)
    off_fit = logit(fit["zona"].map(prior).to_numpy(float))
    m = Multiplicativo(lam).fit(transformar(fit), fit[TARGET].to_numpy(float), off_fit)
    p_base = test["zona"].map(prior).to_numpy(float)
    p = m.predict(transformar(test), logit(p_base))
    return m, prior, p, p_base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-evaluar", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(DATA)
    df["t_eval_utc"] = pd.to_datetime(df["t_eval_utc"], utc=True)
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731

    # 1) Elegir lambda por validación cruzada dejando un año fuera (2016-2023; hueco de 7 d en los bordes).
    #    El test 2024+ no interviene. Se usa el log-loss agrupado: la validación de un solo año casi no
    #    contiene filas de cola y no discrimina.
    dev = df[df.t_eval_utc <= ts("2023-12-31 23:59:59")]
    resultados = {}
    for lam in LAMBDAS:
        yy, pp, bb = [], [], []
        for anio in range(2016, 2024):
            ini, fin = ts(f"{anio}-01-01"), ts(f"{anio}-12-31 23:59:59")
            fuera = dev[(dev.t_eval_utc >= ini) & (dev.t_eval_utc <= fin)]
            dentro = dev[(dev.t_eval_utc < ini - pd.Timedelta(days=7)) | (dev.t_eval_utc > fin + pd.Timedelta(days=7))]
            _, _, p_, pb_ = evaluar(dentro, fuera, lam)
            yy.append(fuera[TARGET].to_numpy()); pp.append(p_); bb.append(pb_)
        y_, p_, b_ = np.concatenate(yy), np.concatenate(pp), np.concatenate(bb)
        ll = lambda q: float(-np.mean(y_ * np.log(q) + (1 - y_) * np.log(1 - q)))  # noqa: E731
        resultados[lam] = (ll(p_), ll(b_), cd.metricas(y_, p_), cd.metricas(y_, b_))
        mc, mb = resultados[lam][2], resultados[lam][3]
        print(f"  lam={lam:7.1f} logloss {resultados[lam][0]:.6f} (baseline {resultados[lam][1]:.6f}) "
              f"AUC {mc['auc']:.4f}/{mb['auc']:.4f} AP {mc['ap']:.4f}/{mb['ap']:.4f} Brier {mc['brier']:.6f}/{mb['brier']:.6f}")
    mejor = min(resultados, key=lambda k: resultados[k][0])
    print("lambda elegido (mínimo log-loss en CV por años, solo <=2023):", mejor)

    # 2) Test final: ajuste <=2023-12-24, hueco 7 d, test >= 2024-01-08.
    fit = df[df.t_eval_utc <= ts("2023-12-24 23:59:59")]
    test = df[df.t_eval_utc >= ts("2024-01-08")]
    m, prior, p, pb = evaluar(fit, test, mejor)
    y = test[TARGET].to_numpy()
    bloques = test["t_eval_utc"].dt.strftime("%Y-%m").to_numpy()
    r = cd.evaluar_candidato(features=FEATURES, y_test=y, p_cand=p, p_base=pb, bloques_test=bloques,
                             n_boot=2000, identicas_en_vigilar=frozenset(FEATURES))
    print("\nTest 2024-01-08+ (n=%d, positivos=%d):" % (len(y), int(y.sum())))
    for k in ("auc", "ap", "brier"):
        print(f"  {k:5} candidato {r['metricas_candidato'][k]:.5f} | baseline {r['metricas_baseline'][k]:.5f} "
              f"| IC95 delta {tuple(round(x, 5) for x in r['ic95_delta'][k])}")
    print("Coeficientes (escalones):", {f"{c}>={u}": round(float(w), 4) for (c, u), w in zip(ESCALONES, m.w)}, "b =", round(m.b, 4))
    print("Multiplicador medio en test: %.3f; p95 = %.2f; máx = %.2f" % (
        float(np.mean(p / pb)), float(np.percentile(p / pb, 95)), float(np.max(p / pb))))
    for nombre, ok in r["comprobaciones"].items():
        print(f"  [{'OK' if ok else 'NO'}] {nombre}")
    print("DESPLEGABLE (criterios):", r["desplegable"])

    if not args.solo_evaluar:
        fit_total = df[df.t_eval_utc <= ts("2026-12-31")]
        prior_final = tasas(fit_total)
        m_final = Multiplicativo(mejor).fit(transformar(fit_total), fit_total[TARGET].to_numpy(float),
                                            logit(fit_total["zona"].map(prior_final).to_numpy(float)))
        mc, mb = r["metricas_candidato"], r["metricas_baseline"]
        joblib.dump({"version": VERSION, "modelo": None, "features": FEATURES,
                     "tasas_zona": prior_final, "lambda": mejor,
                     # Parámetros planos: p = sigmoid(logit(tasa_zona) + b + sum_i w_i * [feature_i >= umbral_i])
                     "escalones": [{"feature": c, "umbral": u, "peso": float(w)} for (c, u), w in zip(ESCALONES, m_final.w)],
                     "sesgo": float(m_final.b),
                     "metricas_test": {"auc": round(mc["auc"], 4), "ap": round(mc["ap"], 4), "brier": round(mc["brier"], 5),
                                       "auc_base": round(mb["auc"], 4), "ap_base": round(mb["ap"], 4),
                                       "brier_base": round(mb["brier"], 5)},
                     "ic95_delta_vs_baseline": {k: [round(x, 5) for x in v] for k, v in r["ic95_delta"].items()},
                     "criterios": r["comprobaciones"],
                     "entrenamiento": {"ajuste_final_hasta": "2026-09-26", "evaluacion": "ajuste<=2023-12-24, test>=2024-01-08",
                                       "dataset": "data/candidate/zona_dia_t3_reconstruido.csv"}}, SALIDA)
        print("Candidato escrito en", SALIDA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
