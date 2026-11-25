"""Criterios obligatorios para permitir que un modelo ML sustituya al baseline por zona.

Módulo puro (sin red, sin Supabase, sin cargar el modelo activo). Lo usan las pruebas
y los scripts de evaluación. Un candidato solo es desplegable si TODAS las
comprobaciones de `evaluar_candidato` pasan; ver docs/PLAN_PRODUCCION_MODELO.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ESTADO_GEOMETRIA = ROOT / "config" / "zonas_t3_estado_validacion.json"

# Nunca pueden ser features: identidad de zona/coordenadas, calendario lunar/solar,
# estadísticos de muestra completa (fuga) y flags derivados de ellos.
PATRON_PROHIBIDAS = re.compile(
    r"^(zona|dest_lat_c|dest_lon_c|lat_centro|lon_centro|p90_.*|media_96h_zona|flag_.*|"
    r"luna_.*|sol_.*|dias_a_sicigia|perigeo_flag|potencial_mareal_g2|"
    r"ratio_actividad_local|exceso_p90_local)$"
)

# Features que /vigilar calcula con EXACTAMENTE la misma definición que el entrenamiento
# (magnitud, profundidad, geometría y declustering). Hoy no hay ninguna: /vigilar usa
# M>=2.5, sin filtro de profundidad, radios al punto de ZONAS_COORDS y sin
# Gardner-Knopoff, mientras que T3 usa M>=4.0, 5-40 km, centroide de la caja destino,
# cajas laterales y declustering. Añadir una feature aquí exige que /vigilar la replique.
FEATURES_VIGILAR_IDENTICAS: frozenset[str] = frozenset()


def features_prohibidas(features: list[str]) -> list[str]:
    return sorted(f for f in features if PATRON_PROHIBIDAS.match(f))


def validar_features(features: list[str], identicas_en_vigilar: frozenset[str] = FEATURES_VIGILAR_IDENTICAS) -> dict:
    prohibidas = features_prohibidas(features)
    no_reproducibles = sorted(set(features) - set(identicas_en_vigilar))
    return {
        "sin_features_prohibidas": not prohibidas,
        "prohibidas": prohibidas,
        "identicas_a_vigilar": not no_reproducibles,
        "no_reproducibles_en_vigilar": no_reproducibles,
    }


def _estado(ruta: Path, clave: str) -> bool:
    try:
        return bool(json.loads(ruta.read_text(encoding="utf-8")).get(clave, False))
    except (OSError, ValueError):
        return False


def geometria_documentada(ruta: Path = ESTADO_GEOMETRIA) -> bool:
    """Criterio de despliegue: las zonas tienen geometría escrita y reproducible."""
    return _estado(ruta, "geometria_documentada")


def geometria_validada(ruta: Path = ESTADO_GEOMETRIA) -> bool:
    """Limitación informativa: revisión sismológica humana (no bloquea el despliegue)."""
    return _estado(ruta, "geometria_validada_sismologicamente")


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def bootstrap_bloques(y: np.ndarray, p_cand: np.ndarray, p_base: np.ndarray, bloques: np.ndarray,
                      n_boot: int = 1000, seed: int = 42) -> dict[str, tuple[float, float]]:
    """IC95 de (candidato - baseline) remuestreando bloques temporales completos (p. ej. meses).

    Devuelve {'auc': (lo, hi), 'ap': (lo, hi), 'brier': (lo, hi)}. Para auc/ap un delta < 0 es
    peor; para brier un delta > 0 es peor. Los remuestreos sin positivos o sin negativos se descartan.
    """
    y, p_cand, p_base, bloques = map(np.asarray, (y, p_cand, p_base, bloques))
    ids = np.unique(bloques)
    indices = {b: np.flatnonzero(bloques == b) for b in ids}
    rng = np.random.default_rng(seed)
    deltas = {"auc": [], "ap": [], "brier": []}
    for _ in range(n_boot):
        elegidos = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([indices[b] for b in elegidos])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        c, b = metricas(yy, p_cand[idx]), metricas(yy, p_base[idx])
        for k in deltas:
            deltas[k].append(c[k] - b[k])
    if not deltas["auc"]:
        raise ValueError("Ningún remuestreo contiene positivos y negativos.")
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in deltas.items()}


def evaluar_candidato(*, features: list[str], y_test: np.ndarray, p_cand: np.ndarray, p_base: np.ndarray,
                      bloques_test: np.ndarray, n_boot: int = 1000, seed: int = 42,
                      geometria_ok: bool | None = None,
                      identicas_en_vigilar: frozenset[str] | None = None) -> dict:
    """Aplica los criterios obligatorios. `desplegable` es True solo si todos pasan."""
    mc, mb = metricas(y_test, p_cand), metricas(y_test, p_base)
    ic = bootstrap_bloques(y_test, p_cand, p_base, bloques_test, n_boot=n_boot, seed=seed)
    val_feat = validar_features(features, FEATURES_VIGILAR_IDENTICAS if identicas_en_vigilar is None
                                else frozenset(identicas_en_vigilar))
    geo = geometria_documentada() if geometria_ok is None else geometria_ok
    comprobaciones = {
        "brier_no_empeora": mc["brier"] <= mb["brier"],
        "auc_no_empeora": mc["auc"] >= mb["auc"],
        "ap_no_empeora": mc["ap"] >= mb["ap"],
        # Deterioro significativo = IC95 entero en el lado malo.
        "bootstrap_sin_deterioro_significativo": not (ic["auc"][1] < 0 or ic["ap"][1] < 0 or ic["brier"][0] > 0),
        "features_identicas_a_vigilar": val_feat["identicas_a_vigilar"],
        "sin_fuga_ni_features_prohibidas": val_feat["sin_features_prohibidas"],
        "geometria_zonas_documentada": geo,
    }
    return {"desplegable": all(comprobaciones.values()), "comprobaciones": comprobaciones,
            "metricas_candidato": mc, "metricas_baseline": mb, "ic95_delta": ic, "detalle_features": val_feat,
            "geometria_validada_sismologicamente": geometria_validada()}
