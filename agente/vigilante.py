import os
import math
from datetime import datetime, timedelta, timezone
import requests
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
RENDER_API_URL = os.getenv("RENDER_API_URL", "https://ultimo-modulo.onrender.com").rstrip("/")
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# Coordenadas centrales de las 20 zonas tectónicas
ZONAS_COORDS = {
    "Alaska/Aleutianas": (54.0, -163.0),
    "Antillas/Caribe": (18.2, -66.5),
    "Chile Central/Sur": (-36.5, -73.0),
    "Chile/Peru (Lima-Valpo)": (-18.5, -70.5),
    "Costa O EEUU": (37.5, -122.0),
    "Filipinas Luzon": (15.5, 121.0),
    "Filipinas Mindanao": (7.0, 126.5),
    "Indonesia Banda/Timor": (-6.5, 129.0),
    "Indonesia Sonda": (-6.0, 105.0),
    "Japon Nankai/Kyushu": (32.5, 133.5),
    "Japon Tohoku/Hokkaido": (38.5, 142.0),
    "Kamchatka/Kuriles": (51.0, 158.0),
    "Mediterraneo Oriental": (36.5, 26.0),
    "Mexico Guerrero/Oaxaca": (16.5, -98.5),
    "Nueva Zelanda/Kermadec": (-34.0, 179.0),
    "Papua Nueva Guinea": (-5.5, 148.0),
    "Solomon/Vanuatu": (-11.0, 165.0),
    "Sumatra/Andaman": (2.5, 95.5),
    "Taiwan/Ryukyu": (24.0, 122.5),
    "Tonga/Fiji": (-20.0, -175.0),
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def descargar_sismos_96h():
    inicio = (datetime.now(timezone.utc) - timedelta(hours=96)).strftime("%Y-%m-%dT%H:%M:%S")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": inicio,
        "minmagnitude": 2.5,
        "limit": 20000,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("features", [])


def enviar_telegram(texto):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "parse_mode": "Markdown"},
        timeout=15,
    )
    resp.raise_for_status()


def main():
    print("1. Conectando a Supabase y descargando catálogo USGS (96h)...")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    eventos = descargar_sismos_96h()
    print(f"   Eventos USGS descargados: {len(eventos)}")

    zonas_db = sb.table("zonas_sismicas").select("*").execute().data or []
    lista_zonas = [z["zona"] for z in zonas_db] if zonas_db else list(ZONAS_COORDS.keys())
    columnas_existentes = set(zonas_db[0].keys()) if zonas_db else set()

    resultados_zona = {}
    for zona in lista_zonas:
        if zona not in ZONAS_COORDS:
            continue
        clat, clon = ZONAS_COORDS[zona]
        loc200_con = 0
        loc200_sin45 = 0
        loc500_con = 0
        prof70_150 = 0

        for ev in eventos:
            props = ev.get("properties") or {}
            geom = ev.get("geometry") or {}
            coords = geom.get("coordinates") or [0, 0, 0]
            mag = props.get("mag")
            if mag is None:
                continue
            lon, lat, depth = coords[0], coords[1], coords[2] or 0.0
            dist = haversine_km(clat, clon, lat, lon)

            if dist <= 500:
                loc500_con += 1
                if 70 <= depth <= 150:
                    prof70_150 += 1
            if dist <= 200:
                loc200_con += 1
                if mag < 4.5:
                    loc200_sin45 += 1

        payload = {
            "zona": zona,
            "loc200_n96_con": loc200_con,
            "loc200_n96_sin45": loc200_sin45,
            "loc500_n96_con": loc500_con,
            "lat_n96_prof70_150": prof70_150,
        }
        pred_resp = requests.post(f"{RENDER_API_URL}/predict", json=payload, timeout=45)
        pred_resp.raise_for_status()
        pred = pred_resp.json()
        resultados_zona[zona] = pred

        # Actualizar solo las columnas existentes en zonas_sismicas
        candidatos = {
            "loc200_n96_con": loc200_con,
            "loc200_n96_sin45": loc200_sin45,
            "loc500_n96_con": loc500_con,
            "lat_n96_prof70_150": prof70_150,
            "multiplicador_riesgo": pred["multiplicador_riesgo"],
            "probabilidad_m65_7d": pred["probabilidad_m65_7d"],
        }
        update_data = {k: v for k, v in candidatos.items() if k in columnas_existentes}
        if update_data:
            sb.table("zonas_sismicas").update(update_data).eq("zona", zona).execute()

        print(f"   {zona}: mult={pred['multiplicador_riesgo']}x | prob7d={pred['probabilidad_m65_7d']*100:.2f}%")

    # Evaluar suscripciones personalizadas de usuarios (alertas_usuario)
    alertas = sb.table("alertas_usuario").select("*").execute().data or []
    print(f"2. Evaluando {len(alertas)} alertas de usuario...")
    notificaciones = []
    for a in alertas:
        zona = a["zona"]
        umbral = float(a.get("umbral_multiplicador", 2.0))
        info = resultados_zona.get(zona)
        if info and info["multiplicador_riesgo"] >= umbral:
            notificaciones.append(
                f"• *{zona}*: Multiplicador *{info['multiplicador_riesgo']}x* "
                f"(umbral ≥ {umbral}x) | Prob. M6.5+ 7d: *{info['probabilidad_m65_7d']*100:.2f}%*"
            )

    if notificaciones:
        mensaje = (
            "🚨 *ALERTA SÍSMICA RELATIVA (USGS + ML)*\n\n"
            + "\n".join(notificaciones)
            + "\n\nPanel: https://ultimo-modulo.vercel.app"
        )
        enviar_telegram(mensaje)
        print("3. Alerta enviada a Telegram correctamente.")
    else:
        print("3. Ninguna zona vigilada superó su umbral en esta ejecución.")


if __name__ == "__main__":
    main()