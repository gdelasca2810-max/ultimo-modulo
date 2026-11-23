import os
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

app = FastAPI(title="API Riesgo Sísmico Relativo (USGS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "modelo_sismos.joblib"
artefacto = joblib.load(MODEL_PATH)

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


class ConsultaRiesgo(BaseModel):
    zona: str
    loc200_n96_con: int
    loc200_n96_sin45: int
    loc500_n96_con: int
    lat_n96_prof70_150: int


def calcular_prediccion(datos: ConsultaRiesgo):
    if datos.zona not in artefacto["tasas_zona"]:
        raise HTTPException(status_code=404, detail=f"Zona no reconocida: {datos.zona}")

    tasa_base = artefacto["tasas_zona"][datos.zona]
    media_96h = artefacto["medias_96h"][datos.zona]

    X = np.array([[
        datos.loc200_n96_con,
        datos.loc200_n96_sin45,
        datos.loc500_n96_con,
        datos.lat_n96_prof70_150,
        media_96h,
    ]])

    prob_7d = float(artefacto["modelo"].predict_proba(X)[0][1])
    multiplicador = round(prob_7d / tasa_base, 2) if tasa_base > 0 else 1.0

    return {
        "zona": datos.zona,
        "probabilidad_m65_7d": round(prob_7d, 5),
        "tasa_historica_7d": round(tasa_base, 5),
        "multiplicador_riesgo": multiplicador,
        "alerta_enjambre": multiplicador >= 2.0,
    }


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@app.get("/health")
def health():
    return {"status": "ok", "metricas_test": artefacto["metricas_test"]}


@app.post("/predict")
def predecir(datos: ConsultaRiesgo):
    return calcular_prediccion(datos)


@app.api_route("/vigilar", methods=["GET", "POST"])
def ejecutar_vigilancia(key: str = Query(default="")):
    cron_secret = os.getenv("CRON_SECRET", "")
    if cron_secret and key != cron_secret:
        raise HTTPException(status_code=401, detail="Clave de cron inválida")

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat_id = os.environ["TELEGRAM_CHAT_ID"]

    inicio = (datetime.now(timezone.utc) - timedelta(hours=96)).strftime("%Y-%m-%dT%H:%M:%S")
    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params={"format": "geojson", "starttime": inicio, "minmagnitude": 2.5, "limit": 20000},
        timeout=30,
    )
    resp.raise_for_status()
    eventos = resp.json().get("features", [])

    sb = create_client(supabase_url, supabase_key)
    zonas_db = sb.table("zonas_sismicas").select("*").execute().data or []
    lista_zonas = [z["zona"] for z in zonas_db] if zonas_db else list(ZONAS_COORDS.keys())
    columnas_existentes = set(zonas_db[0].keys()) if zonas_db else set()

    resultados_zona = {}
    detalles_zona = {}
    for zona in lista_zonas:
        if zona not in ZONAS_COORDS:
            continue
        clat, clon = ZONAS_COORDS[zona]
        loc200_con = loc200_sin45 = loc500_con = prof70_150 = 0

        for ev in eventos:
            props = ev.get("properties") or {}
            coords = (ev.get("geometry") or {}).get("coordinates") or [0, 0, 0]
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

        pred = calcular_prediccion(ConsultaRiesgo(
            zona=zona,
            loc200_n96_con=loc200_con,
            loc200_n96_sin45=loc200_sin45,
            loc500_n96_con=loc500_con,
            lat_n96_prof70_150=prof70_150,
        ))
        resultados_zona[zona] = pred
        detalles_zona[zona] = {
            "loc200_con": loc200_con,
            "loc200_sin45": loc200_sin45,
            "loc500_con": loc500_con,
            "prof70_150": prof70_150,
        }

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

    alertas = sb.table("alertas_usuario").select("*").execute().data or []
    notificaciones = []
    for a in alertas:
        zona = a["zona"]
        umbral = float(a.get("umbral_multiplicador", 2.0))
        info = resultados_zona.get(zona)
        det = detalles_zona.get(zona, {})
        if info and info["multiplicador_riesgo"] >= umbral:
            notificaciones.append(
                f"📍 <b>Zona:</b> {zona}\n"
                f"⚠️ <b>Probabilidad M6.5+ (7 días):</b> {info['probabilidad_m65_7d']*100:.2f}% "
                f"(Base normal: {info['tasa_historica_7d']*100:.2f}%)\n"
                f"📈 <b>Riesgo relativo:</b> <b>{info['multiplicador_riesgo']}x</b> (umbral configurado: {umbral}x)\n"
                f"📊 <b>Base del cálculo (96h):</b> {det.get('loc200_con', 0)} sismos &lt;200km "
                f"({det.get('loc200_sin45', 0)} &lt;M4.5), {det.get('loc500_con', 0)} &lt;500km, "
                f"{det.get('prof70_150', 0)} en prof. 70–150km"
            )

    alerta_enviada = False
    if notificaciones:
        mensaje = (
            "🚨 <b>ALERTA SÍSMICA AUTOMÁTICA (USGS + ML)</b>\n\n"
            + "\n\n".join(notificaciones)
            + "\n\n🌐 Panel web: https://ultimo-modulo.vercel.app"
        )
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            json={"chat_id": tg_chat_id, "text": mensaje, "parse_mode": "HTML"},
            timeout=15,
        ).raise_for_status()
        alerta_enviada = True

    return {
        "eventos_usgs_96h": len(eventos),
        "zonas_actualizadas": len(resultados_zona),
        "alertas_evaluadas": len(alertas),
        "alerta_enviada_telegram": alerta_enviada,
    }


@app.post("/telegram-webhook")
async def telegram_webhook(update: dict):
    message = update.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    texto_usuario = (message.get("text") or "").strip()

    if not texto_usuario or not chat_id:
        return {"ok": True}

    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    zonas_db = sb.table("zonas_sismicas").select("*").execute().data or []

    resumen_zonas = []
    for z in zonas_db:
        nombre = z.get("zona")
        tasa_base = artefacto["tasas_zona"].get(nombre, 0.0)
        mult = z.get("multiplicador_riesgo", "N/A")
        prob = z.get("probabilidad_m65_7d")
        prob_pct = f"{float(prob)*100:.2f}%" if prob is not None else "N/A"
        base_pct = f"{tasa_base*100:.2f}%"
        resumen_zonas.append(
            f"- {nombre}: prob_m65_7d={prob_pct} (tasa_historica={base_pct}), "
            f"multiplicador={mult}x, sismos_200km_96h={z.get('loc200_n96_con', 0)}, "
            f"sismos_menores_4.5_200km={z.get('loc200_n96_sin45', 0)}, "
            f"sismos_500km_96h={z.get('loc500_n96_con', 0)}, "
            f"sismos_prof_70_150km={z.get('lat_n96_prof70_150', 0)}"
        )

    if gemini_key:
        prompt = (
            "Eres el asistente de monitoreo sísmico USGS + Machine Learning. "
            "Responde en español usando EXCLUSIVAMENTE etiquetas HTML básicas de Telegram "
            "(<b>negrita</b> e <i>cursiva</i>). PROHIBIDO usar asteriscos (*) o Markdown.\n\n"
            "Estructura OBLIGATORIA línea por línea (deja una línea en blanco entre cada bloque):\n"
            "🔎 <b>Consulta:</b> [Resumen breve de lo que preguntó el usuario]\n"
            "📍 <b>Lugar / Zona:</b> [Nombre exacto de la zona o zonas]\n"
            "⚠️ <b>Probabilidad de sismo mayor (M6.5+ en 7 días):</b> [Porcentaje] (Frente a una base histórica normal de [tasa_historica] — Riesgo relativo: <b>[multiplicador]x</b>)\n"
            "📊 <b>En base a qué se calcula (últimas 96h):</b>\n"
            "• Sismos cercanos (&lt;200 km): [valor] ([valor menores a M4.5] menores a M4.5)\n"
            "• Sismos regionales (&lt;500 km): [valor]\n"
            "• Sismos intermedios (70–150 km prof.): [valor]\n"
            "💡 <b>Conclusión:</b> [1 frase directa respondiendo a la duda del usuario]\n\n"
            "Si el usuario pregunta por varias zonas o un ranking, muestra hasta 3 zonas con ese mismo formato claro línea por línea.\n\n"
            "Datos actuales en tiempo real:\n"
            + "\n".join(resumen_zonas)
            + f"\n\nPregunta del usuario: {texto_usuario}"
        )
        respuesta = ""
        ultimo_error = ""
        for modelo in (
            "gemini-flash-latest",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
        ):
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={gemini_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=20,
            )
            if r.ok:
                respuesta = (
                    r.json()
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "No pude generar respuesta.")
                )
                break
            ultimo_error = f"{modelo} ({r.status_code}): {r.text[:180]}"

        if not respuesta:
            respuesta = f"Error al consultar Gemini -> {ultimo_error}"
    else:
        respuesta = "Falta configurar GEMINI_API_KEY en Render."

    requests.post(
        f"https://api.telegram.org/bot{tg_token}/sendMessage",
        json={"chat_id": chat_id, "text": respuesta, "parse_mode": "HTML"},
        timeout=15,
    )
    return {"ok": True}