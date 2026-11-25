"""Criterios obligatorios de despliegue de un modelo ML frente al baseline por zona."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

import criterios_despliegue as cd  # noqa: E402


def escenario(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    p_base = rng.choice([0.01, 0.02, 0.05], size=n)
    y = (rng.random(n) < p_base).astype(int)
    bloques = np.repeat(np.arange(n // 100), 100)
    return y, p_base, bloques, rng


class Criterios(unittest.TestCase):
    def test_features_prohibidas_detectadas(self):
        malas = ["p90_con_11a", "media_96h_zona", "luna_dist_km", "dest_lat_c", "flag_A1", "zona",
                 "ratio_actividad_local", "exceso_p90_local", "potencial_mareal_g2"]
        self.assertEqual(cd.features_prohibidas(malas), sorted(malas))
        permitidas = ["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_prof70_150"]
        self.assertEqual(cd.features_prohibidas(permitidas), [])

    def test_hoy_ninguna_feature_esta_garantizada_identica_en_vigilar(self):
        v = cd.validar_features(["loc200_n96_con"])
        self.assertTrue(v["sin_features_prohibidas"])
        self.assertFalse(v["identicas_a_vigilar"])
        v = cd.validar_features(["loc200_n96_con"], identicas_en_vigilar=frozenset({"loc200_n96_con"}))
        self.assertTrue(v["identicas_a_vigilar"])

    def test_geometria_documentada_pero_no_validada_por_un_sismologo(self):
        self.assertTrue(cd.geometria_documentada())      # criterio de despliegue
        self.assertFalse(cd.geometria_validada())        # limitación registrada, no bloquea
        self.assertFalse(cd.geometria_documentada(ROOT / "config" / "no_existe.json"))

    def test_candidato_identico_al_baseline_no_es_desplegable_sin_features_replicadas_en_vigilar(self):
        y, p_base, bloques, _ = escenario()
        r = cd.evaluar_candidato(features=["loc200_n96_con"], y_test=y, p_cand=p_base.copy(),
                                 p_base=p_base, bloques_test=bloques, n_boot=100)
        c = r["comprobaciones"]
        self.assertTrue(c["brier_no_empeora"] and c["auc_no_empeora"] and c["ap_no_empeora"])
        self.assertTrue(c["bootstrap_sin_deterioro_significativo"])
        self.assertFalse(c["features_identicas_a_vigilar"])
        self.assertTrue(c["geometria_zonas_documentada"])
        self.assertFalse(r["desplegable"])

    def test_todos_los_criterios_cumplidos_permiten_desplegar(self):
        y, p_base, bloques, _ = escenario()
        r = cd.evaluar_candidato(features=["loc200_n96_con"], y_test=y, p_cand=p_base.copy(),
                                 p_base=p_base, bloques_test=bloques, n_boot=100, geometria_ok=True,
                                 identicas_en_vigilar=frozenset({"loc200_n96_con"}))
        self.assertTrue(all(r["comprobaciones"].values()), r["comprobaciones"])
        self.assertTrue(r["desplegable"])
        # Basta que falle un solo requisito (aquí, una feature prohibida) para bloquearlo.
        r = cd.evaluar_candidato(features=["loc200_n96_con", "p90_con_11a"], y_test=y, p_cand=p_base.copy(),
                                 p_base=p_base, bloques_test=bloques, n_boot=100, geometria_ok=True,
                                 identicas_en_vigilar=frozenset({"loc200_n96_con", "p90_con_11a"}))
        self.assertFalse(r["comprobaciones"]["sin_fuga_ni_features_prohibidas"])
        self.assertFalse(r["desplegable"])

    def test_candidato_peor_es_detectado_con_bootstrap_por_bloques(self):
        y, p_base, bloques, rng = escenario()
        p_ruido = np.clip(rng.random(len(y)) * 0.2, 0.001, 0.99)  # sin información y mal calibrado
        r = cd.evaluar_candidato(features=["loc200_n96_con"], y_test=y, p_cand=p_ruido,
                                 p_base=p_base, bloques_test=bloques, n_boot=200, geometria_ok=True)
        c = r["comprobaciones"]
        self.assertFalse(c["brier_no_empeora"])
        self.assertFalse(c["bootstrap_sin_deterioro_significativo"])
        self.assertFalse(r["desplegable"])

    def test_bootstrap_es_reproducible(self):
        y, p_base, bloques, rng = escenario()
        p = np.clip(p_base * rng.uniform(0.5, 1.5, len(y)), 0.001, 0.99)
        a = cd.bootstrap_bloques(y, p, p_base, bloques, n_boot=50, seed=7)
        b = cd.bootstrap_bloques(y, p, p_base, bloques, n_boot=50, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
