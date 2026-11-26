from datetime import datetime, timedelta, timezone
import math
import os
import httpx
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Monitor Riesgo Sismico USGS")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://mknmzuvagniweqlljund.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "sb_publishable_-O6gSSWO-IW49-KoUt6f5A_957UQy16",
)
RENDER_API_URL = os.getenv("RENDER_API_URL", "https://ultimo-modulo.onrender.com")
VERCEL_APP_URL = os.getenv("VERCEL_APP_URL", "https://ultimo-modulo.vercel.app")
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

        if dist <= 300:
            loc200_con += 1
            if mag >= 4.0:
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


@mcp.tool()
def consultar_zonas_riesgo_supabase(limite: int = 20) -> list[dict]:
    """Consulta el estado en vivo de las zonas sismicas y probabilidades calculadas en Supabase (zonas_riesgo_hoy o zonas_sismicas)."""
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    try:
        url = f"{SUPABASE_URL}/rest/v1/zonas_riesgo_hoy?select=*&order=multiplicador_riesgo.desc&limit={limite}"
        resp = httpx.get(url, headers=headers, timeout=15.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        url_fb = f"{SUPABASE_URL}/rest/v1/zonas_sismicas?select=*&order=multiplicador_riesgo.desc&limit={limite}"
        resp_fb = httpx.get(url_fb, headers=headers, timeout=15.0)
        resp_fb.raise_for_status()
        return resp_fb.json()


@mcp.tool()
def consultar_sismos_historicos_supabase(magnitud_min: float = 6.0, limite: int = 10) -> list[dict]:
    """Consulta los terremotos historicos registrados en Supabase con magnitud mayor o igual al umbral."""
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    url = f"{SUPABASE_URL}/rest/v1/sismos_historicos?mag=gte.{magnitud_min}&order=time.desc&limit={limite}"
    resp = httpx.get(url, headers=headers, timeout=15.0)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def verificar_servicio_render() -> dict:
    """Verifica la salud de la API en Render (/health), latencia y version del modelo activo."""
    t0 = datetime.now(timezone.utc)
    try:
        resp = httpx.get(f"{RENDER_API_URL}/health", timeout=20.0)
        ms = round((datetime.now(timezone.utc) - t0).total_seconds() * 1000, 1)
        return {
            "status_code": resp.status_code,
            "latencia_ms": ms,
            "datos": resp.json() if resp.status_code == 200 else resp.text,
            "render_url": RENDER_API_URL,
        }
    except Exception as e:
        return {
            "error": str(e),
            "render_url": RENDER_API_URL,
            "mensaje": "No se pudo conectar con Render (posible arranque en frio o inactivo)",
        }


@mcp.tool()
def disparar_vigilar_render(clave_cron: str = "") -> dict:
    """Dispara la ejecucion del endpoint /vigilar en Render para actualizar Supabase y notificar Telegram."""
    key = clave_cron or os.getenv("CRON_SECRET", "")
    url = f"{RENDER_API_URL}/vigilar"
    params = {"key": key} if key else {}
    t0 = datetime.now(timezone.utc)
    resp = httpx.post(url, params=params, timeout=60.0)
    ms = round((datetime.now(timezone.utc) - t0).total_seconds() * 1000, 1)
    return {
        "status_code": resp.status_code,
        "duracion_ms": ms,
        "respuesta": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
    }


@mcp.tool()
def verificar_frontend_vercel() -> dict:
    """Verifica el estado HTTP, tiempo de respuesta y disponibilidad de la app frontend en Vercel."""
    t0 = datetime.now(timezone.utc)
    try:
        resp = httpx.get(VERCEL_APP_URL, timeout=15.0)
        ms = round((datetime.now(timezone.utc) - t0).total_seconds() * 1000, 1)
        return {
            "status_code": resp.status_code,
            "latencia_ms": ms,
            "url": VERCEL_APP_URL,
            "disponible": resp.status_code == 200,
        }
    except Exception as e:
        return {
            "error": str(e),
            "url": VERCEL_APP_URL,
            "disponible": False,
        }


@mcp.tool()
def consultar_sismos_usgs(
    magnitud_min: float = 4.5,
    horas: int = 96,
    limite: int = 20,
    zona: str = None,
) -> dict:
    """Consulta directamente la API en vivo de USGS ComCat para obtener sismos segun magnitud, ventana de horas y zona opcional."""
    fin = datetime.now(timezone.utc)
    inicio = fin - timedelta(hours=horas)
    params = {
        "format": "geojson",
        "starttime": inicio.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": fin.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": magnitud_min,
        "limit": min(limite, 100),
    }
    if zona:
        try:
            info = _obtener_zona_supabase(zona)
            params["latitude"] = info["lat_centro"]
            params["longitude"] = info["lon_centro"]
            params["maxradiuskm"] = 500
        except Exception:
            pass
    resp = httpx.get(USGS_ENDPOINT, params=params, timeout=20.0)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    eventos = []
    for f in features:
        p = f["properties"]
        coords = f["geometry"]["coordinates"]
        eventos.append({
            "id": f.get("id"),
            "lugar": p.get("place"),
            "magnitud": p.get("mag"),
            "tiempo_utc": datetime.fromtimestamp(p.get("time", 0) / 1000.0, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "profundidad_km": round(coords[2], 1),
            "coordenadas": {"lat": coords[1], "lon": coords[0]},
        })
    return {
        "total_encontrados": len(eventos),
        "filtro_magnitud_min": magnitud_min,
        "ventana_horas": horas,
        "eventos": eventos,
    }


def obtener_mcp_asgi_app(token: str = None):
    """Crea la aplicacion ASGI de MCP sobre SSE protegida por Token Bearer para despliegue remoto en Render."""
    from starlette.responses import Response

    token_esperado = token or os.getenv("MCP_AUTH_TOKEN", "sismo_evaluador_2026")
    raw_app = mcp.sse_app()

    class MCPAuthMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                headers = dict(scope.get("headers", []))
                auth = headers.get(b"authorization", b"").decode("utf-8")
                query = scope.get("query_string", b"").decode("utf-8")
                from urllib.parse import parse_qs
                t_param = parse_qs(query).get("token", [""])[0]

                esperado = f"Bearer {token_esperado}"
                if auth != esperado and auth != token_esperado and t_param != token_esperado:
                    resp = Response(
                        "Unauthorized: Token MCP inválido o ausente. Envíe 'Authorization: Bearer <token>' o '?token=<token>'",
                        status_code=401,
                    )
                    await resp(scope, receive, send)
                    return
            await self.app(scope, receive, send)

    return MCPAuthMiddleware(raw_app)


if __name__ == "__main__":
    mcp.run()