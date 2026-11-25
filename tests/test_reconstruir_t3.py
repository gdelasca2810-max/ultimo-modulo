"""Ventanas y reglas del generador T3 reconstruido (scripts/reconstruir_t3_usgs.py)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("reconstruir_t3_usgs", ROOT / "scripts" / "reconstruir_t3_usgs.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
sys.path.insert(0, str(ROOT / "api"))
import features_t3 as f3  # noqa: E402

ZONA = "Zona de prueba"
DEST = [30, 35, 131, 138]  # centroide 32.5, 134.5


def evento(id_, t, mag, prof=10.0, lat=32.5, lon=134.5, tipo="earthquake"):
    return {"id": id_, "time_utc": t, "latitude": lat, "longitude": lon, "depth_km": prof,
            "mag": mag, "type": tipo}


class ReglasT3(unittest.TestCase):
    def test_caja_con_longitud_mayor_a_180(self):
        lat, lon = np.array([-20.0, -20.0, -20.0]), np.array([-175.0, 175.0, 100.0])  # -175 == 185
        self.assertEqual(f3.inbox(lat, lon, [-38, -15, 172, 188]).tolist(), [True, True, False])

    def test_contar_es_semiabierto(self):
        serie = np.array([100, 150, 200, 200, 250])
        self.assertEqual(int(f3.contar(serie, np.array([100]), np.array([200]))[0]), 2)  # [100,200)

    def test_gardner_knopoff_marca_replicas_solo_dentro_de_ventana_y_radio(self):
        t0 = 1_700_000_000
        tiempo = np.array([t0, t0 + 86400, t0 + 2 * 86400, t0 + 3 * 86400])
        lat = np.array([0.0, 0.1, 5.0, 0.1])  # la 3ª, a ~550 km, queda fuera del radio
        lon = np.zeros(4)
        mag = np.array([7.0, 5.0, 5.0, 7.5])
        aft = f3.decluster_gk(tiempo, lat, lon, mag)
        self.assertEqual(aft.tolist(), [False, True, False, False])

    def _dataset(self, eventos, inicio="2020-01-05 00:37", fin="2020-01-07 00:37"):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame(eventos).to_csv(d / "raw.csv", index=False)
            (d / "zonas.json").write_text(json.dumps({ZONA: {"lat": [[-60, -50, -20, -10]], "dest": DEST}}))
            df, meta = r3.reconstruir(d / "raw.csv", d / "no_existe.csv", d / "zonas.json", inicio, fin)
        return df.set_index("t_eval_utc"), meta

    def test_features_usan_ventana_96h_semiabierta_y_filtros(self):
        eventos = [
            evento("a", "2019-12-31T23:37:00Z", 4.5),   # justo antes de t-96h: fuera
            evento("b", "2020-01-01T00:37:00Z", 4.5),   # exactamente t-96h: dentro
            evento("c", "2020-01-05T00:37:00Z", 4.5),   # exactamente t: fuera de [t-96h, t)
            evento("d", "2020-01-02T00:00:00Z", 3.9),   # M<4.0: fuera
            evento("e", "2020-01-02T00:00:00Z", 4.5, prof=100.0),  # fuera del rango 5..40 km
            evento("f", "2020-01-02T00:00:00Z", 4.5, tipo="explosion"),  # no es terremoto
        ]
        df, meta = self._dataset(eventos)
        t0 = pd.Timestamp("2020-01-05 00:37", tz="UTC")
        self.assertEqual(int(df.loc[t0, "loc200_n96_con"]), 1)
        self.assertEqual(int(df.loc[t0, "loc200_n96_sin45"]), 1)
        self.assertEqual(int(df.loc[t0, "loc500_n96_con"]), 1)
        self.assertEqual(len(df), 3)
        self.assertEqual(meta["zonas"], 1)

    def test_target_semiabierto_7d_en_la_caja_destino(self):
        eventos = [
            evento("g", "2020-01-12T00:37:00Z", 6.6, prof=300.0),  # exactamente t0+7d: fuera para t0
            evento("h", "2020-01-20T00:00:00Z", 6.6, lat=0.0, lon=0.0),  # fuera de la caja
        ]
        df, _ = self._dataset(eventos)
        y = df["y_M65_7d"].tolist()  # cualquier profundidad cuenta
        self.assertEqual(y, [0, 1, 1])

    def test_target_excluye_replicas(self):
        eventos = [
            evento("p", "2020-01-04T00:00:00Z", 7.5),                     # antes del primer t_eval
            evento("q", "2020-01-08T00:00:00Z", 6.6, lat=32.6),            # réplica GK del anterior
        ]
        df, _ = self._dataset(eventos)
        self.assertEqual(df["y_M65_7d"].tolist(), [0, 0, 0])

    def test_calendario_historico_sin_referencia(self):
        ev = r3.calendario_evaluacion(None, "2016-01-05 00:37", "2026-09-26 00:37")
        self.assertEqual(len(ev), 3918)


if __name__ == "__main__":
    unittest.main()
