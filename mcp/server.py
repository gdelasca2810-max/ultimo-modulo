from datetime import datetime, timedelta, timezone
import math
import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Monitor Riesgo Sismico USGS")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://mknmzuvagniweqlljund.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "sb_publishable_-O6gSSWO-IW49-KoUt6f5A_957UQy16",
)
RENDER_API_URL = os.getenv("RENDER_API_URL", "https://ultimo-modulo.onrender.com")
USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _obtener_zona_supabase(zona: str) -> dict:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    url = f"{SUPABASE_URL}/rest/v1/zonas_sismicas?zona=eq.{zona}&select=*"
    resp = httpx.get(url, headers=headers, timeout=15.0)
    resp.raise_for_status()
    filas = resp.json()
    if not filas:
        raise ValueError(f"Zona no encontrada en Supabase: '{zona}'")
    return filas[0]


@mcp.tool()
def comparar_baseline_zona(zona: str) -> dict:
    """Obtiene de Supabase la tasa historica real de M6.5+ (7d) y la media de sismos 96h de una zona."""
    info = _obtener_zona_supabase(zona)
    return {
        "zona": info["zona"],
        "coordenadas": {"lat": info["lat_centro"], "lon": info["lon_centro"]},
        "tasa_historica_m65_7d_pct": round(info["tasa_historica_m65_7d"] * 100, 3),
        "media_sismos_96h": info["media_96h"],
        "nota": "Tasa calibrada en periodo train 2016-2023. El 86% de los M6.5+ ocurren sin precursores locales.",
    }


@mcp.tool()
def buscar_sismos_zona(zona: str, horas: int = 96) -> dict:
    """Consulta en tiempo real USGS ComCat (M4.0+) en los ultimos 96h alrededor de la zona y calcula las 4 variables del modelo."""
    info = _obtener_zona_supabase(zona)
    lat_c, lon_c = info["lat_centro"], info["lon_centro"]

    fin = datetime.now(timezone.utc)
    inicio = fin - timedelta(hours=horas)

    params = {
        "format": "geojson",
        "starttime": inicio.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": fin.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": lat_c,
        "longitude": lon_c,
        "maxradiuskm": 500,
        "minmagnitude": 4.0,
    }
    resp = httpx.get(USGS_ENDPOINT, params=params, timeout=20.0)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    loc200_con = 0
    loc200_sin45 = 0
    loc500_con = len(features)
    prof70_150 = 0
    eventos_recientes = []

    for f in features:
        props = f["properties"]
        lon, lat, prof = f["geometry"]["coordinates"]
        mag = props.get("mag") or 0.0
        dist = _haversine_km(lat_c, lon_c, lat, lon)

        if dist <= 200:
            loc200_con += 1
            if mag >= 4.5:
                loc200_sin45 += 1
        if 70 <= prof <= 150:
            prof70_150 += 1

        eventos_recientes.append({
            "lugar": props.get("place"),
            "magnitud": mag,
            "profundidad_km": round(prof, 1),
            "distancia_km": round(dist, 1),
        })

    return {
        "zona": zona,
        "ventana_horas": horas,
        "variables_modelo": {
            "loc200_n96_con": loc200_con,
            "loc200_n96_sin45": loc200_sin45,
            "loc500_n96_con": loc500_con,
            "lat_n96_prof70_150": prof70_150,
        },
        "ultimos_eventos": eventos_recientes[:5],
    }


@mcp.tool()
def predecir_riesgo_sismico(
    zona: str,
    loc200_n96_con: int,
    loc200_n96_sin45: int,
    loc500_n96_con: int,
    lat_n96_prof70_150: int,
) -> dict:
    """Llama a la API ML desplegada en Render para calcular la probabilidad M6.5+ a 7 dias y el multiplicador relativo."""
    payload = {
        "zona": zona,
        "loc200_n96_con": loc200_n96_con,
        "loc200_n96_sin45": loc200_n96_sin45,
        "loc500_n96_con": loc500_n96_con,
        "lat_n96_prof70_150": lat_n96_prof70_150,
    }
    resp = httpx.post(f"{RENDER_API_URL}/predict", json=payload, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    mcp.run()