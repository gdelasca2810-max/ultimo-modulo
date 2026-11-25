"""Reglas T3 de features sísmicas: UNA sola implementación para entrenamiento y para /vigilar.

Importado por scripts/reconstruir_t3_usgs.py (dataset histórico) y por api/main.py (producción),
de modo que no puede haber desajuste entrenamiento/producción en la definición de las variables.

Reglas (generador histórico recuperado, ver docs/CONTINUIDAD_ML.md):
- catálogo M>=4.0, type=earthquake; declustering Gardner-Knopoff (réplicas fuera donde se indica);
- ventana de features [t-96h, t);
- loc200_n96_con: <=200 km al centroide de la caja destino, M>=4.0, prof. 5..40 km;
- loc200_n96_sin45: igual con M>=4.5 y SIN réplicas (el nombre no significa «M<4.5»);
- loc500_n96_con: <=500 km, M>=4.0, prof. 5..40 km;
- lat_n96_prof70_150: cajas laterales, M>=4.5, 70<prof<=150.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

W96 = 96 * 3600
# Un mainshock GK puede "cubrir" réplicas hasta ~1064 días (M9); con holgura para la ventana de 96 h.
HISTORIA_GK_DIAS = 1100
UMBRAL_MAINSHOCK_GK = 4.8
ROOT = Path(__file__).resolve().parent.parent
ZONAS_PATH = ROOT / "config" / "zonas_t3_historicas.json"


def cargar_zonas(ruta: Path = ZONAS_PATH) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def centroide(destino: list[float]) -> tuple[float, float]:
    return (destino[0] + destino[1]) / 2, (destino[2] + destino[3]) / 2


def inbox(lat: np.ndarray, lon: np.ndarray, caja: list[float]) -> np.ndarray:
    a1, a2, o1, o2 = caja
    lon_ajustada = np.where((o2 > 180) & (lon < 0), lon + 360, lon)
    return (lat >= a1) & (lat <= a2) & (lon_ajustada >= o1) & (lon_ajustada <= o2)


def inboxes(lat: np.ndarray, lon: np.ndarray, cajas: list[list[float]]) -> np.ndarray:
    mascara = np.zeros(len(lat), dtype=bool)
    for caja in cajas:
        mascara |= inbox(lat, lon, caja)
    return mascara


def distancia_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    a1, a2 = np.radians(lat1), np.radians(lat2)
    x = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def decluster_gk(tiempo: np.ndarray, lat: np.ndarray, lon: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Declustering Gardner-Knopoff (implementación literal del generador recuperado).

    `tiempo` en segundos epoch, ordenado ascendente. Devuelve True para las réplicas.
    """
    aft = np.zeros(len(mag), dtype=bool)
    candidatos = np.where(mag >= UMBRAL_MAINSHOCK_GK)[0]
    candidatos = candidatos[np.argsort(-mag[candidatos])]
    for i in candidatos:
        if aft[i]:
            continue
        longitud = 10 ** (0.1238 * mag[i] + 0.983)
        dias = 10 ** (0.032 * mag[i] + 2.7389) if mag[i] >= 6.5 else 10 ** (0.5409 * mag[i] - 0.547)
        lo = np.searchsorted(tiempo, tiempo[i], side="right")
        hi = np.searchsorted(tiempo, tiempo[i] + int(dias * 86400), side="right")
        hijos = np.arange(lo, hi)
        hijos = hijos[mag[hijos] < mag[i]]
        if len(hijos):
            d = distancia_km(lat[i], lon[i], lat[hijos], lon[hijos])
            aft[hijos[d <= longitud]] = True
    return aft


def contar(serie: np.ndarray, ini: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """Número de eventos de `serie` (ordenada) en [ini, fin)."""
    return np.searchsorted(serie, fin, side="left") - np.searchsorted(serie, ini, side="left")


def mascaras_zona(definicion: dict, lat: np.ndarray, lon: np.ndarray, dep: np.ndarray,
                  mag: np.ndarray, aft: np.ndarray) -> dict[str, np.ndarray]:
    """Máscaras booleanas por evento para una zona. ÚNICA definición de las variables T3."""
    destino, laterales = definicion["dest"], definicion["lat"]
    clat, clon = centroide(destino)
    m_lat = inboxes(lat, lon, laterales)
    d = distancia_km(clat, clon, lat, lon)
    somero = (dep >= 5) & (dep <= 40)
    som = m_lat & (mag >= 4.0) & somero
    return {
        "dest": inbox(lat, lon, destino),
        "lat_todo": m_lat & (mag >= 4.0),
        "som": som,
        "prof": m_lat & (mag >= 4.0) & (dep > 40),
        "prof70": m_lat & (mag >= 4.5) & (dep > 70) & (dep <= 150),
        "local": (d <= 200) & (mag >= 4.0) & somero,
        "local_sin": (d <= 200) & (mag >= 4.5) & somero & ~aft,
        "regional": (d <= 500) & (mag >= 4.0) & somero,
    }


class Catalogo:
    """Eventos ordenados por tiempo con la marca de réplica GK ya calculada."""

    def __init__(self, ts, lat, lon, dep, mag):
        orden = np.argsort(np.asarray(ts), kind="stable")
        self.ts = np.asarray(ts, dtype=np.int64)[orden]
        self.lat = np.asarray(lat, dtype=float)[orden]
        self.lon = np.asarray(lon, dtype=float)[orden]
        self.dep = np.asarray(dep, dtype=float)[orden]
        self.mag = np.asarray(mag, dtype=float)[orden]
        self.aft = decluster_gk(self.ts, self.lat, self.lon, self.mag)


def features_vivas(zonas: dict, cat: Catalogo, ahora_s: int) -> dict[str, dict[str, int]]:
    """Las cuatro variables del modelo, por zona, en la ventana [ahora-96h, ahora).

    `cat` debe contener M>=4.0 de las últimas 96 h y los M>=4.8 de los últimos HISTORIA_GK_DIAS días
    (los necesarios para que el declustering GK sea idéntico al del entrenamiento).
    """
    ini = np.array([ahora_s - W96])
    fin = np.array([ahora_s])
    salida = {}
    for zona, definicion in zonas.items():
        m = mascaras_zona(definicion, cat.lat, cat.lon, cat.dep, cat.mag, cat.aft)

        def n(nombre):
            return int(contar(cat.ts[m[nombre]], ini, fin)[0])

        salida[zona] = {
            "loc200_n96_con": n("local"),
            "loc200_n96_sin45": n("local_sin"),
            "loc500_n96_con": n("regional"),
            "lat_n96_prof70_150": n("prof70"),
        }
    return salida
