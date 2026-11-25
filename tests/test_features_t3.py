"""Identidad entrenamiento/producción: las features en vivo son las del generador histórico."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

import features_t3 as f3  # noqa: E402

spec = importlib.util.spec_from_file_location("reconstruir_t3_usgs", ROOT / "scripts" / "reconstruir_t3_usgs.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)

ZONA = "Japon NE"
T_EVAL = pd.Timestamp("2020-01-05 00:37", tz="UTC")


def ev(i, t, mag, dlat=0.0, dlon=0.0, prof=15.0):
    return {"id": f"e{i}", "time_utc": t, "latitude": 39.0 + dlat, "longitude": 142.5 + dlon,
            "depth_km": prof, "mag": mag, "type": "earthquake"}


def catalogo_sintetico():
    ruido = [ev(100 + i, f"2019-{1 + i % 12:02d}-{1 + i % 27:02d}T05:00:00Z", 4.0 + (i % 5) * 0.1,
                dlat=(i % 7) * 0.2) for i in range(60)]
    return ruido + [
        ev(1, "2019-06-01T00:00:00Z", 7.1),                         # mainshock antiguo (>96 h)
        ev(2, "2020-01-02T10:00:00Z", 6.4, dlat=0.2),               # réplica GK del anterior (a 200 d)
        ev(3, "2020-01-04T06:00:00Z", 5.6, dlat=0.1),               # mainshock reciente
        ev(4, "2020-01-04T12:00:00Z", 4.9, dlat=0.15),              # réplica de #3
        ev(5, "2020-01-03T00:00:00Z", 4.7, dlat=1.0, dlon=1.0),     # independiente a ~150 km
        ev(6, "2020-01-03T03:00:00Z", 4.2, dlat=0.5),               # M4.0-4.4: cuenta en 'con', no en 'sin45'
        ev(7, "2020-01-04T20:00:00Z", 4.6, prof=90.0),              # profundidad intermedia, no es 5-40 km
        ev(8, "2020-01-04T22:00:00Z", 4.6, prof=1.0),               # demasiado somero: fuera de 5-40 km
        ev(9, "2020-01-02T00:37:00Z", 4.6),                         # justo en t-96h: dentro
        ev(10, "2020-01-05T00:37:00Z", 4.6),                        # justo en t: fuera
    ]


def preparar(datos):
    datos = datos.copy()
    datos["t"] = pd.to_datetime(datos["time_utc"], utc=True)
    reciente = datos["t"] >= T_EVAL - pd.Timedelta(hours=96)
    return datos, reciente


def vivo(df):
    segundos = [int(x.timestamp()) for x in df["t"]]  # independiente de la unidad de pandas (ns/us)
    cat = f3.Catalogo(segundos, df.latitude, df.longitude, df.depth_km, df.mag)
    return f3.features_vivas({ZONA: f3.cargar_zonas()[ZONA]}, cat, int(T_EVAL.timestamp()))[ZONA]


class IdentidadEntrenamientoProduccion(unittest.TestCase):
    def test_features_en_vivo_son_identicas_a_la_reconstruccion_historica(self):
        crudo = pd.DataFrame(catalogo_sintetico())
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            crudo.to_csv(d / "raw.csv", index=False)
            (d / "z.json").write_text(json.dumps({ZONA: f3.cargar_zonas()[ZONA]}))
            off, _ = r3.reconstruir(d / "raw.csv", d / "no.csv", d / "z.json", "2020-01-05 00:37", "2020-01-05 00:37")
        fila = off.iloc[0]

        datos, reciente = preparar(crudo)
        # En producción solo se descargan M>=4.0 de las últimas 96 h y M>=4.8 de la historia.
        visible = datos[(reciente & (datos.mag >= 4.0)) | (~reciente & (datos.mag >= f3.UMBRAL_MAINSHOCK_GK))]
        self.assertLess(len(visible), len(datos))  # el filtro de producción sí descarta eventos
        v = vivo(visible)

        for col in ("loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_prof70_150"):
            self.assertEqual(v[col], int(fila[col]), col)
        # Y el caso construido no es trivial:
        self.assertGreaterEqual(v["loc200_n96_con"], 4)
        self.assertLess(v["loc200_n96_sin45"], v["loc200_n96_con"])  # réplicas y M<4.5 fuera de sin45
        self.assertEqual(v["lat_n96_prof70_150"], 0)  # la caja lateral de Japón NE queda en Sudamérica

    def test_el_declustering_necesita_la_historia_de_mainshocks(self):
        """Sin los M>=4.8 antiguos, una réplica lejana en el tiempo se contaría como independiente."""
        datos, reciente = preparar(pd.DataFrame(catalogo_sintetico()))
        con_historia = vivo(datos[(reciente & (datos.mag >= 4.0)) | (~reciente & (datos.mag >= 4.8))])
        sin_historia = vivo(datos[reciente & (datos.mag >= 4.0)])
        self.assertLess(con_historia["loc200_n96_sin45"], sin_historia["loc200_n96_sin45"])


if __name__ == "__main__":
    unittest.main()
