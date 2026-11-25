"""Compatibilidad artefacto/API: contrato JSON de /health, /predict y /vigilar, y modo baseline.

Sin red ni credenciales reales: /vigilar se prueba con USGS, Supabase y Telegram simulados.
Ejecutar: python -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

import joblib  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from criterios_despliegue import validar_features  # noqa: E402

CLAVES_PREDICT = {"zona", "version_modelo", "probabilidad_m65_7d", "tasa_historica_7d",
                  "multiplicador_riesgo", "alerta_enjambre"}
CLAVES_VIGILAR = {"eventos_usgs_96h", "version_modelo", "zonas_actualizadas", "alertas_evaluadas",
                  "alerta_enviada_telegram", "chats_notificados", "alertas_sin_telegram_vinculado"}
ZONA = "Japon NE"  # prior ~1,8 %: con enjambre local el modelo llega a 2x y a >=2,5 %
ZONA_RARA = "Japon Nankai/Kyushu"  # prior ~0,2 %: un 2x sigue siendo <0,5 %, no debe disparar «enjambre»
CUERPO = {"zona": ZONA, "loc200_n96_con": 50, "loc200_n96_sin45": 40,
          "loc500_n96_con": 80, "lat_n96_prof70_150": 10}
ENTORNO_FALSO = {"SUPABASE_URL": "http://prueba.invalid", "SUPABASE_SERVICE_KEY": "no-es-una-clave",
                 "TELEGRAM_BOT_TOKEN": "no-es-un-token", "TELEGRAM_CHAT_ID": "0"}


def sin_modo():
    entorno = {k: v for k, v in os.environ.items() if k != "MODO_PREDICCION"}
    return mock.patch.dict(os.environ, entorno, clear=True)


class Consulta:
    """Doble mínimo de la API fluida de supabase-py."""

    def __init__(self, filas, bitacora, tabla):
        self.filas, self.bitacora, self.tabla = filas, bitacora, tabla

    def select(self, *_):
        return self

    def update(self, datos):
        self.bitacora.append((self.tabla, "update", datos))
        return self

    def eq(self, *_):
        return self

    def execute(self):
        return mock.Mock(data=self.filas)


class SupabaseFalso:
    def __init__(self, umbral=0.5, zona=ZONA):
        self.bitacora = []
        self.tablas = {
            "zonas_sismicas": [{"zona": zona, "lat_centro": 39.0, "lon_centro": 142.5,
                                "loc200_n96_con": 0, "loc200_n96_sin45": 0, "loc500_n96_con": 0, "lat_n96_prof70_150": 0,
                                "multiplicador_riesgo": None, "probabilidad_m65_7d": None, "actualizado_en": None}],
            "alertas_usuario": [{"zona": zona, "umbral_multiplicador": umbral, "user_id": "u1"}],
            "vinculos_telegram": [{"user_id": "u1", "chat_id": "1"}],
        }

    def table(self, nombre):
        return Consulta(self.tablas[nombre], self.bitacora, nombre)


def usgs_falso(lat=39.0, lon=142.5, n=6, mag=4.6):
    """Doble de main.descargar_eventos_usgs: n sismos M>=4.5 someros a <=200 km, en las últimas 96 h."""
    ahora = int(datetime.now(timezone.utc).timestamp())
    eventos = [(f"ev{i}", ahora - 3600 * (i + 1), lat + 0.01 * i, lon, 20.0, mag) for i in range(n)]

    def descargar(inicio, fin, min_mag):
        return [e for e in eventos if e[5] >= min_mag]

    return descargar


class ContratoApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_health_conserva_claves(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json()), {"status", "version_modelo", "metricas_test"})
        self.assertEqual(r.json()["status"], "ok")

    def test_modo_por_defecto_es_el_modelo(self):
        with sin_modo():
            r = self.client.post("/predict", json=CUERPO)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json()), CLAVES_PREDICT)
        self.assertEqual(r.json()["version_modelo"], main.artefacto.get("version", "v1"))
        self.assertFalse(main.modo_baseline_activo())

    def test_zona_no_soportada_sigue_dando_404(self):
        for modo in ("modelo", "baseline_zona"):
            with mock.patch.dict(os.environ, {"MODO_PREDICCION": modo}):
                r = self.client.post("/predict", json={**CUERPO, "zona": "Sumatra/Andaman"})
            self.assertEqual(r.status_code, 404, modo)

    def test_baseline_mantiene_contrato_y_no_alerta(self):
        with mock.patch.dict(os.environ, {"MODO_PREDICCION": "baseline_zona"}):
            r = self.client.post("/predict", json=CUERPO)
        self.assertEqual(r.status_code, 200)
        cuerpo = r.json()
        self.assertEqual(set(cuerpo), CLAVES_PREDICT)
        tasa = round(float(main.baseline_zona["zonas"][ZONA]["tasa"]), 5)
        self.assertEqual(cuerpo["version_modelo"], "baseline_tasa_zona_v1")
        self.assertEqual(cuerpo["probabilidad_m65_7d"], tasa)
        self.assertEqual(cuerpo["tasa_historica_7d"], tasa)
        self.assertEqual(cuerpo["multiplicador_riesgo"], 1.0)
        self.assertIs(cuerpo["alerta_enjambre"], False)  # incluso con 50 sismos a <=200 km

    def test_artefacto_cubre_exactamente_las_zonas_historicas(self):
        zonas = set(json.loads((ROOT / "config" / "zonas_t3_historicas.json").read_text(encoding="utf-8")))
        self.assertEqual(zonas, set(main.artefacto["tasas_zona"]))

    def test_artefacto_activo_cumple_el_contrato_interno(self):
        self.assertTrue(main.artefacto["version"].startswith("v5"))
        self.assertTrue(all(0 < t < 1 for t in main.artefacto["tasas_zona"].values()))
        self.assertTrue(all(e["peso"] >= 0 for e in main.artefacto["escalones"]))
        self.assertEqual(set(main.artefacto["metricas_test"]),
                         {"auc", "ap", "brier", "auc_base", "ap_base", "brier_base"})
        self.assertTrue(all(main.artefacto["criterios"].values()), main.artefacto["criterios"])

    def test_baseline_cubre_las_zonas_y_nunca_da_cero(self):
        zonas = set(json.loads((ROOT / "config" / "zonas_t3_historicas.json").read_text(encoding="utf-8")))
        self.assertEqual(zonas, set(main.baseline_zona["zonas"]))
        self.assertTrue(all(0 < v["tasa"] < 1 for v in main.baseline_zona["zonas"].values()))
        # Motivo del suavizado: el artefacto v2 antiguo tenía una zona con tasa 0
        # (Japon Nankai/Kyushu: sin positivos hasta 2023), engañosa como probabilidad.
        antiguo = joblib.load(ROOT / "api" / "modelo_sismos.joblib")
        self.assertEqual(antiguo["tasas_zona"][ZONA_RARA], 0.0)

    def test_baseline_sin_archivo_falla_de_forma_explicita(self):
        with mock.patch.object(main, "baseline_zona", None), \
                mock.patch.dict(os.environ, {"MODO_PREDICCION": "baseline_zona"}):
            r = self.client.post("/predict", json=CUERPO)
        self.assertEqual(r.status_code, 503)

    def test_artefacto_activo_cumple_la_politica_de_features(self):
        valida = validar_features(main.artefacto["features"], frozenset(main.artefacto["features"]))
        self.assertTrue(valida["sin_features_prohibidas"], valida)

    def test_el_artefacto_antiguo_v2_tenia_features_prohibidas(self):
        """Documenta por qué se sustituyó: fuga temporal (media/p90 de muestra completa)."""
        antiguo = joblib.load(ROOT / "api" / "modelo_sismos.joblib")
        valida = validar_features(antiguo["features"])
        self.assertIn("media_96h_zona", valida["prohibidas"])
        self.assertIn("p90_con_11a", valida["prohibidas"])

    def _vigilar(self, modo, umbral=0.5, zona=ZONA, lat=39.0, lon=142.5, n=6):
        sb = SupabaseFalso(umbral, zona)
        entorno = {**ENTORNO_FALSO}
        if modo:
            entorno["MODO_PREDICCION"] = modo
        with sin_modo(), mock.patch.dict(os.environ, entorno), \
                mock.patch.object(main, "create_client", return_value=sb), \
                mock.patch.object(main, "descargar_eventos_usgs", usgs_falso(lat, lon, n)), \
                mock.patch.object(main.requests, "post") as post:
            r = self.client.get("/vigilar")
        return r, sb, post

    def test_vigilar_en_baseline_conserva_respuesta_y_no_envia_telegram(self):
        r, sb, post = self._vigilar("baseline_zona")
        self.assertEqual(r.status_code, 200)
        cuerpo = r.json()
        self.assertEqual(set(cuerpo), CLAVES_VIGILAR)
        self.assertEqual(cuerpo["version_modelo"], main.artefacto.get("version", "v1"))
        self.assertEqual(cuerpo["alertas_evaluadas"], 1)
        self.assertIs(cuerpo["alerta_enviada_telegram"], False)
        self.assertEqual(cuerpo["chats_notificados"], 0)
        post.assert_not_called()
        # zonas_sismicas sigue recibiendo las columnas existentes, con multiplicador neutro.
        actualizaciones = [d for t, a, d in sb.bitacora if t == "zonas_sismicas" and a == "update"]
        self.assertEqual(len(actualizaciones), 1)
        self.assertEqual(actualizaciones[0]["multiplicador_riesgo"], 1.0)
        self.assertEqual(actualizaciones[0]["loc200_n96_con"], 6)
        self.assertEqual(actualizaciones[0]["loc200_n96_sin45"], 6)

    def test_vigilar_en_modo_modelo_conserva_las_claves(self):
        r, _, _ = self._vigilar(None, umbral=1000.0)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json()), CLAVES_VIGILAR)
        self.assertEqual(r.json()["eventos_usgs_96h"], 6)

    def test_enjambre_real_en_zona_de_tasa_normal_envia_alerta(self):
        r, sb, post = self._vigilar(None, umbral=2.0, n=6)
        self.assertEqual(r.json()["chats_notificados"], 1)
        post.assert_called_once()
        act = [d for t, a, d in sb.bitacora if t == "zonas_sismicas"][0]
        self.assertGreaterEqual(act["multiplicador_riesgo"], 2.0)

    def test_poca_actividad_no_alerta_aunque_el_umbral_sea_bajo(self):
        r, _, post = self._vigilar(None, umbral=1.0, n=2)  # 2 sismos: por debajo de 3 locales
        self.assertEqual(r.json()["chats_notificados"], 0)
        post.assert_not_called()

    def test_zona_de_tasa_muy_baja_no_alerta_por_un_multiplicador_artificial(self):
        # Mismo enjambre (6 sismos) en una zona de tasa ~0,2 %: 2x sigue siendo <2,5 % => sin alerta.
        r, sb, post = self._vigilar(None, umbral=1.0, zona=ZONA_RARA, lat=32.5, lon=134.5, n=6)
        self.assertEqual(r.json()["chats_notificados"], 0)
        post.assert_not_called()
        act = [d for t, a, d in sb.bitacora if t == "zonas_sismicas"][0]
        self.assertLess(act["probabilidad_m65_7d"], 0.025)


if __name__ == "__main__":
    unittest.main()
