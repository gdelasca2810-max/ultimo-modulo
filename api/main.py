import json
import os
import re
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib
import numpy as np
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
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
    "Venezuela/Caribe": (10.5, -66.9),
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

    tasa_base = float(artefacto["tasas_zona"][datos.zona])
    media_96h = float(artefacto["medias_96h"][datos.zona])

    if artefacto.get("version") == "v2_features_11a":
        p90_local = float(artefacto.get("p90_zona", {}).get(datos.zona, media_96h))
        loc200_m45plus = max(0, datos.loc200_n96_con - datos.loc200_n96_sin45)
        ratio_actividad = datos.loc200_n96_con / (media_96h + 0.25)
        exceso_p90 = datos.loc200_n96_con - p90_local
        concentracion = datos.loc200_n96_con / (datos.loc500_n96_con + 1.0)

        X = np.array([[
            datos.loc200_n96_con,
            datos.loc200_n96_sin45,
            loc200_m45plus,
            datos.loc500_n96_con,
            datos.lat_n96_prof70_150,
            media_96h,
            p90_local,
            ratio_actividad,
            exceso_p90,
            concentracion,
        ]], dtype=float)
    else:
        X = np.array([[
            datos.loc200_n96_con,
            datos.loc200_n96_sin45,
            datos.loc500_n96_con,
            datos.lat_n96_prof70_150,
            media_96h,
        ]], dtype=float)

    prob_7d = float(artefacto["modelo"].predict_proba(X)[0][1])
    multiplicador = round(prob_7d / tasa_base, 2) if tasa_base > 0 else 1.0

    es_elevado = (
        multiplicador >= 2.0
        and prob_7d >= 0.025
        and datos.loc200_n96_con >= 3
    )

    return {
        "zona": datos.zona,
        "version_modelo": artefacto.get("version", "v1"),
        "probabilidad_m65_7d": round(prob_7d, 5),
        "tasa_historica_7d": round(tasa_base, 5),
        "multiplicador_riesgo": multiplicador,
        "alerta_enjambre": es_elevado,
    }


def formatear_frecuencia_habitual(tasa_base: float) -> str:
    if tasa_base <= 0:
        return "frecuencia histórica muy baja en el historial"
    semanas = round(1.0 / tasa_base)
    if semanas >= 104:
        anios = round(semanas / 52.0, 1)
        return f"~1 cada {anios} años"
    return f"~1 cada {semanas} semanas"


def describir_nivel_usuario(
    multiplicador: float,
    prob_7d: float,
    tasa_base: float,
    loc200_con: int,
    prof70_150: int,
    loc500_con: int,
):
    frec_txt = formatear_frecuencia_habitual(tasa_base)

    if multiplicador >= 2.0 and prob_7d >= 0.025 and loc200_con >= 3:
        estado = "🔴 ELEVADO (Enjambre sísmico local activo)"
        resumen_simple = "Hay actividad local relevante y la estimación a 7 días está por encima de lo habitual."
    elif multiplicador >= 2.0 and loc200_con == 0:
        estado = "🟡 AVISO POR MODELO (Sin sismos a menos de 200 km)"
        resumen_simple = (
            f"El modelo marca {multiplicador}x frente a su base, pero NO hay temblores a menos de 200 km en 96h; "
            "no equivale a un enjambre local activo."
        )
    elif multiplicador >= 1.5 and loc200_con == 0 and loc500_con == 0:
        estado = "🟡 AVISO POR MODELO (Sin sismos cercanos ni regionales)"
        resumen_simple = (
            f"El modelo marca {multiplicador}x frente a su base, pero no hubo sismos a menos de 500 km en 96h; "
            "es una variación estadística, no actividad local."
        )
    elif multiplicador >= 1.5 or (loc200_con >= 2 or prof70_150 >= 2):
        if loc200_con == 0 and prof70_150 > 0:
            estado = "🟡 MODERADO (Solo sismos de profundidad intermedia en observación)"
            resumen_simple = (
                "Solo hay sismos de profundidad intermedia (70–150 km, a menos de 500 km); "
                "ninguno a menos de 200 km."
            )
        else:
            estado = "🟡 MODERADO (Actividad en observación)"
            resumen_simple = "Se registra actividad reciente a vigilar, aunque la probabilidad semanal sigue moderada."
    else:
        estado = "🟢 NORMAL (Dentro de lo habitual)"
        resumen_simple = "La situación se mantiene dentro del comportamiento normal de la zona."

    return estado, frec_txt, resumen_simple


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def construir_mensaje_alerta(zona, estado_txt, resumen_simple, frec_txt, info, umbral,
                             cercanos, m45_plus, regionales, profundos=0) -> str:
    prob = info["probabilidad_m65_7d"] * 100
    habitual = info["tasa_historica_7d"] * 100
    mult = info["multiplicador_riesgo"]
    if mult >= 1.1:
        comparacion = f"<b>{mult}x</b> más de lo normal"
    elif mult <= 0.9:
        comparacion = f"<b>{mult}x</b>, por debajo de lo normal"
    else:
        comparacion = f"<b>{mult}x</b>, igual que lo normal"
    if cercanos == 0:
        cerca_txt = "ningún temblor a menos de 200 km"
    else:
        cerca_txt = f"{_plural(cercanos, 'temblor cercano', 'temblores cercanos')} (a menos de 200 km"
        cerca_txt += f", {_plural(m45_plus, 'fuerte', 'fuertes')} de M4.5+)" if m45_plus else ")"
    region_txt = (
        "ninguno a menos de 500 km" if regionales == 0
        else f"{_plural(regionales, 'sismo', 'sismos')} a menos de 500 km (incluye los de menos de 200 km)"
    )
    prof_txt = (
        "ninguno de profundidad intermedia (70–150 km)" if profundos == 0
        else f"{_plural(profundos, 'sismo', 'sismos')} de profundidad intermedia (70–150 km, dentro de los 500 km)"
    )
    if cercanos == 0 and regionales == 0:
        impulsor = "Sin sismos cercanos en 96 h: el valor viene solo del cálculo estadístico del modelo."
    elif cercanos == 0:
        impulsor = "Sin sismos a menos de 200 km: el valor se apoya en actividad regional o de profundidad intermedia."
    else:
        impulsor = "Impulsado por actividad cercana (menos de 200 km) de las últimas 96 horas."
    return (
        f"📍 <b>{zona}</b>\n"
        f"{estado_txt}\n"
        f"💬 <i>{resumen_simple}</i>\n\n"
        f"⚠️ <b>Probabilidad de un terremoto fuerte (M6.5+) en los próximos 7 días:</b> {prob:.2f}%\n"
        f"• Lo normal aquí: {habitual:.2f}% ({frec_txt})\n"
        f"• Hoy: {comparacion}\n\n"
        f"🔎 <b>Últimas 96 horas:</b>\n"
        f"• Radio ≤ 200 km: {cerca_txt}\n"
        f"• Radio ≤ 500 km: {region_txt}\n"
        f"• Profundidad intermedia: {prof_txt}\n"
        f"🧭 <i>{impulsor}</i>\n\n"
        f"🎯 <i>Tu alerta: avisar si el riesgo llega a {umbral}x o más.</i>"
    )


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version_modelo": artefacto.get("version", "v1"),
        "metricas_test": artefacto["metricas_test"],
    }


@app.post("/predict")
def predecir(datos: ConsultaRiesgo):
    return calcular_prediccion(datos)


def enviar_lotes_telegram(tg_token: str, chat_id: str, notificaciones: list) -> None:
    cabecera = (
        "🚨 <b>Alerta sísmica</b>\n"
        "<i>Estimación estadística (datos USGS + modelo). No predice cuándo ocurrirá un terremoto: "
        "indica si el riesgo de esta semana está por encima o por debajo de lo normal en tu zona. "
        "M6.5+ = terremoto fuerte; \"1x\" = riesgo normal, \"2x\" = el doble.</i>\n\n"
    )
    pie = "\n\n🌐 Panel web: https://ultimo-modulo.vercel.app"

    lotes = []
    tam_lote = 4
    for i in range(0, len(notificaciones), tam_lote):
        bloque = "\n\n".join(notificaciones[i:i + tam_lote])
        if i == 0:
            bloque = cabecera + bloque
        if i + tam_lote >= len(notificaciones):
            bloque = bloque + pie
        lotes.append(bloque)

    for lote in lotes:
        r_tg = requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            json={"chat_id": chat_id, "text": lote[:4000], "parse_mode": "HTML"},
            timeout=15,
        )
        if not r_tg.ok:
            texto_plano = (
                lote.replace("<b>", "")
                .replace("</b>", "")
                .replace("<i>", "")
                .replace("</i>", "")
                .replace("&lt;", "<")
            )
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": chat_id, "text": texto_plano[:4000]},
                timeout=15,
            ).raise_for_status()


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
        fila = next((z for z in zonas_db if z["zona"] == zona), {})
        if fila.get("lat_centro") is not None and fila.get("lon_centro") is not None:
            clat, clon = float(fila["lat_centro"]), float(fila["lon_centro"])
        elif zona in ZONAS_COORDS:
            clat, clon = ZONAS_COORDS[zona]
        else:
            continue
        if zona not in artefacto["tasas_zona"]:
            continue
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
            "loc200_m45plus": max(0, loc200_con - loc200_sin45),
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
            "actualizado_en": datetime.now(timezone.utc).isoformat(),
        }
        update_data = {k: v for k, v in candidatos.items() if k in columnas_existentes}
        if update_data:
            sb.table("zonas_sismicas").update(update_data).eq("zona", zona).execute()

    alertas = sb.table("alertas_usuario").select("*").execute().data or []
    vinculos = {
        v["user_id"]: v["chat_id"]
        for v in (sb.table("vinculos_telegram").select("user_id,chat_id").execute().data or [])
        if v.get("chat_id")
    }
    por_chat = {}
    sin_vinculo = 0
    for a in alertas:
        zona = a["zona"]
        umbral = float(a.get("umbral_multiplicador", 2.0))
        info = resultados_zona.get(zona)
        det = detalles_zona.get(zona, {})
        if info and info["multiplicador_riesgo"] >= umbral:
            cercanos = det.get("loc200_con", 0)
            m45_plus = det.get("loc200_m45plus", 0)
            regionales = det.get("loc500_con", 0)
            profundos = det.get("prof70_150", 0)

            estado_txt, frec_txt, resumen_simple = describir_nivel_usuario(
                info["multiplicador_riesgo"],
                info["probabilidad_m65_7d"],
                info["tasa_historica_7d"],
                cercanos,
                profundos,
                regionales,
            )

            chat = vinculos.get(a.get("user_id"))
            if not chat:
                sin_vinculo += 1
                continue
            por_chat.setdefault(chat, []).append(construir_mensaje_alerta(
                zona, estado_txt, resumen_simple, frec_txt, info, umbral,
                cercanos, m45_plus, regionales, profundos,
            ))

    alerta_enviada = False
    for chat_id, mensajes in por_chat.items():
        enviar_lotes_telegram(tg_token, chat_id, mensajes)
        alerta_enviada = True

    return {
        "eventos_usgs_96h": len(eventos),
        "version_modelo": artefacto.get("version", "v1"),
        "zonas_actualizadas": len(resultados_zona),
        "alertas_evaluadas": len(alertas),
        "alerta_enviada_telegram": alerta_enviada,
        "chats_notificados": len(por_chat),
        "alertas_sin_telegram_vinculado": sin_vinculo,
    }


def enviar_progreso_tg(tg_token: str, chat_id: str, texto: str, msg_id: int | None = None):
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
            timeout=5,
        )
    except Exception:
        pass

    if msg_id:
        r = requests.post(
            f"https://api.telegram.org/bot{tg_token}/editMessageText",
            json={"chat_id": chat_id, "message_id": msg_id, "text": texto},
            timeout=10,
        )
        if r.ok:
            return msg_id
    r = requests.post(
        f"https://api.telegram.org/bot{tg_token}/sendMessage",
        json={"chat_id": chat_id, "text": texto},
        timeout=10,
    )
    if r.ok:
        return (r.json().get("result") or {}).get("message_id")
    return msg_id


def llamar_llm(prompt: str):
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    respuesta = ""
    ultimo_error = ""
    errores = []
    if openai_key:
        modelo_oa = os.getenv("OPENAI_MODEL", "gpt-6-luna")
        try:
            r_oa = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": modelo_oa,
                    "messages": [{"role": "user", "content": prompt}],
                    **({"reasoning_effort": os.getenv("OPENAI_REASONING", "low")}
                       if modelo_oa.startswith(("gpt-6", "gpt-5")) else {}),
                },
                timeout=40,
            )
            if r_oa.ok:
                respuesta = (r_oa.json()["choices"][0]["message"]["content"] or "").strip()
            else:
                errores.append(f"openai:{modelo_oa} ({r_oa.status_code})")
                print(f"[openai] {modelo_oa} ({r_oa.status_code}): {r_oa.text[:120]}", flush=True)
        except requests.RequestException as e:
            errores.append(f"openai:{modelo_oa} ({type(e).__name__})")
            print(f"[openai] {modelo_oa}: {type(e).__name__}", flush=True)

    modelos = [m.strip() for m in os.getenv("GEMINI_MODELS", "").split(",") if m.strip()] or [
        "gemini-flash-latest",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
    ]
    for modelo in (modelos if (gemini_key and not respuesta) else []):
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={gemini_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=40,
            )
        except requests.RequestException as e:
            ultimo_error = f"{modelo}: {type(e).__name__}"
            print(f"[gemini] {ultimo_error}", flush=True)
            errores.append(f"{modelo} ({type(e).__name__})")
            continue
        if r.ok:
            respuesta = (
                r.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No pude generar respuesta.")
            )
            break
        ultimo_error = f"{modelo} ({r.status_code}): {r.text[:120]}"
        print(f"[gemini] {ultimo_error}", flush=True)
        errores.append(f"{modelo} ({r.status_code})")
    return respuesta, errores


def enviar_respuesta_tg(tg_token: str, chat_id: str, msg_id, respuesta: str) -> None:
    if msg_id:
        r_final = requests.post(
            f"https://api.telegram.org/bot{tg_token}/editMessageText",
            json={
                "chat_id": chat_id,
                "message_id": msg_id,
                "text": respuesta,
                "parse_mode": "HTML",
            },
            timeout=15,
        )
        if not r_final.ok:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/editMessageText",
                json={"chat_id": chat_id, "message_id": msg_id, "text": respuesta},
                timeout=15,
            )
    else:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            json={"chat_id": chat_id, "text": respuesta, "parse_mode": "HTML"},
            timeout=15,
        )


HIST_INICIO = "2015-01-01"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
PALABRAS_HIST = re.compile(
    r"\b(19|20)\d{2}\b|hist[oó]ric|pasad|hace \d+|mayor(es)? (sismo|terremoto)|"
    r"(terremoto|sismo) (de|del|en)|[uú]ltim[oa]s? (mes|a[ñn]o|semana)|a[ñn]o pasado",
    re.I,
)


def _feature_a_fila(ev: dict):
    props = ev.get("properties") or {}
    coords = (ev.get("geometry") or {}).get("coordinates") or [None, None, None]
    if props.get("mag") is None or props.get("time") is None:
        return None
    return {
        "id": ev.get("id"),
        "fecha": datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc).isoformat(),
        "mag": props["mag"],
        "prof": coords[2],
        "lat": coords[1],
        "lon": coords[0],
        "lugar": props.get("place"),
    }


@app.api_route("/historial/cargar", methods=["GET", "POST"])
def cargar_historial(desde: str, hasta: str, min_mag: float = 4.5, key: str = Query(default="")):
    cron_secret = os.getenv("CRON_SECRET", "")
    if cron_secret and key != cron_secret:
        raise HTTPException(status_code=401, detail="Clave inválida")
    r = requests.get(
        USGS_URL,
        params={"format": "geojson", "starttime": desde, "endtime": hasta,
                "minmagnitude": min_mag, "limit": 20000, "orderby": "time"},
        timeout=90,
    )
    r.raise_for_status()
    filas = [f for f in (_feature_a_fila(e) for e in r.json().get("features", [])) if f and f["id"]]
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    for i in range(0, len(filas), 1000):
        sb.table("sismos_historicos").upsert(filas[i:i + 1000], on_conflict="id").execute()
    return {"desde": desde, "hasta": hasta, "cargados": len(filas)}


def extraer_filtros_historicos(texto: str):
    """Devuelve filtros si la pregunta pide sismos pasados concretos; si no, None."""
    if not PALABRAS_HIST.search(texto):
        return None
    prompt = (
        f"Hoy es {datetime.now(timezone.utc).date().isoformat()}. Si la pregunta pide sismos PASADOS concretos "
        "(por fecha, año, lugar o magnitud), responde SOLO este JSON sin texto extra: "
        '{"historico": true, "desde": "AAAA-MM-DD", "hasta": "AAAA-MM-DD", "min_mag": 6.0, '
        '"lugar": "palabra clave en inglés como aparece en USGS, p.ej. Chile, Japan, Mexico; vacío si no hay", "limite": 5}. '
        'Si no es una consulta de sismos pasados responde {"historico": false}. '
        f"Pregunta: {texto}"
    )
    salida, _ = llamar_llm(prompt)
    m = re.search(r"\{.*\}", salida or "", re.S)
    if not m:
        return None
    try:
        f = json.loads(m.group(0))
        if not f.get("historico"):
            return None
        return {
            "desde": str(f["desde"])[:10],
            "hasta": str(f["hasta"])[:10],
            "min_mag": max(4.5, float(f.get("min_mag") or 4.5)),
            "lugar": str(f.get("lugar") or "").strip(),
            "limite": max(1, min(int(f.get("limite") or 5), 10)),
        }
    except Exception:
        return None


def buscar_sismos_pasados(sb, f: dict):
    """Primero la base propia; si no hay datos (o el rango es anterior a la carga), USGS en vivo."""
    if f["desde"] >= HIST_INICIO:
        q = (sb.table("sismos_historicos").select("fecha,mag,lugar,prof")
             .gte("fecha", f["desde"]).lte("fecha", f["hasta"] + "T23:59:59")
             .gte("mag", f["min_mag"]))
        if f["lugar"]:
            q = q.ilike("lugar", f"%{f['lugar']}%")
        filas = q.order("mag", desc=True).limit(f["limite"]).execute().data or []
        if filas:
            return filas, "📚 Fuente: base de datos propia (USGS, M4.5+ desde 2015)"
    try:
        r = requests.get(
            USGS_URL,
            params={"format": "geojson", "starttime": f["desde"], "endtime": f["hasta"],
                    "minmagnitude": f["min_mag"], "orderby": "magnitude", "limit": 300},
            timeout=40,
        )
        r.raise_for_status()
        filas = [x for x in (_feature_a_fila(e) for e in r.json().get("features", [])) if x]
        if f["lugar"]:
            filas = [x for x in filas if f["lugar"].lower() in (x["lugar"] or "").lower()]
        return filas[: f["limite"]], "🌐 Fuente: USGS en tiempo real (no estaba en la base propia)"
    except requests.RequestException:
        return [], "🌐 Fuente: USGS en tiempo real (no disponible ahora)"


def responder_historico(sb, texto_usuario: str, f: dict) -> str:
    filas, fuente = buscar_sismos_pasados(sb, f)
    datos = "\n".join(
        f"- {x['fecha'][:10]}: M{x['mag']} · {x.get('lugar')} · profundidad {x.get('prof')} km" for x in filas
    ) or "(sin resultados)"
    ficha = "\n".join(f"- {c['contenido']}" for c in recuperar_conocimiento(texto_usuario, sb))
    prompt = (
        "Eres un asistente que explica sismos pasados a ciudadanos sin conocimientos técnicos. Responde en español "
        "con SOLO etiquetas HTML de Telegram (<b>, <i>); PROHIBIDO Markdown o asteriscos; usa &lt; para el símbolo menor que. "
        "Usa EXCLUSIVAMENTE estos datos (no inventes fechas, magnitudes ni lugares). Si no hay resultados, dilo y sugiere "
        "ampliar fechas o bajar la magnitud. Presenta los sismos como lista (fecha, magnitud, lugar) y una frase de contexto. "
        f"Termina EXACTAMENTE con esta línea: {fuente}\n\n"
        f"Sismos encontrados:\n{datos}\n"
        + (f"\nConocimiento de apoyo (opcional):\n{ficha}\n" if ficha else "")
        + f"\nPregunta: {texto_usuario}"
    )
    respuesta, errores = llamar_llm(prompt)
    if not respuesta:
        respuesta = "⚠️ No pude redactar la respuesta ahora. (Probé: " + ", ".join(errores) + ")\n" + fuente
    return respuesta


def procesar_consulta_telegram(tg_token: str, chat_id: str, texto_usuario: str, msg_id: int | None):
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    filtros = extraer_filtros_historicos(texto_usuario)
    if filtros:
        enviar_respuesta_tg(tg_token, chat_id, msg_id, responder_historico(sb, texto_usuario, filtros))
        return
    zonas_db = sb.table("zonas_sismicas").select("*").execute().data or []

    evaluaciones = []
    for z in zonas_db:
        nombre = z.get("zona")
        if nombre not in artefacto["tasas_zona"]:
            continue
        loc200_con = int(z.get("loc200_n96_con") or 0)
        loc200_sin45 = int(z.get("loc200_n96_sin45") or 0)
        loc200_m45plus = max(0, loc200_con - loc200_sin45)
        loc500_con = int(z.get("loc500_n96_con") or 0)
        prof70_150 = int(z.get("lat_n96_prof70_150") or 0)

        pred = calcular_prediccion(ConsultaRiesgo(
            zona=nombre,
            loc200_n96_con=loc200_con,
            loc200_n96_sin45=loc200_sin45,
            loc500_n96_con=loc500_con,
            lat_n96_prof70_150=prof70_150,
        ))
        prob_val = pred["probabilidad_m65_7d"]
        base_val = pred["tasa_historica_7d"]
        mult = pred["multiplicador_riesgo"]
        estado_txt, frec_txt, _ = describir_nivel_usuario(
            mult, prob_7d=prob_val, tasa_base=base_val, loc200_con=loc200_con, prof70_150=prof70_150,
            loc500_con=loc500_con,
        )

        evaluaciones.append((
            prob_val,
            f"- {nombre}: estado={estado_txt}, prob_7d={prob_val*100:.2f}%, "
            f"habitual_7d={base_val*100:.2f}% ({frec_txt}), "
            f"multiplicador={mult}x, sismos_cercanos_200km={loc200_con} ({loc200_m45plus} moderados M4.5+ y {loc200_sin45} leves <M4.5), "
            f"sismos_regionales_500km={loc500_con} (incluye los de 200 km), "
            f"sismos_prof_intermedia_70_150km_dentro_de_500km={prof70_150}"
        ))

    evaluaciones.sort(key=lambda x: x[0], reverse=True)
    resumen_zonas = [item[1] for item in evaluaciones]

    msg_id = enviar_progreso_tg(
        tg_token,
        chat_id,
        "🧠 2/2 Predicción calculada. Redactando explicación clara...",
        msg_id=msg_id,
    )

    contexto_rag = recuperar_conocimiento(texto_usuario, sb)
    bloque_rag = ""
    if contexto_rag:
        bloque_rag = (
            "\n\nConocimiento de apoyo (úsalo SOLO si ayuda a explicar un concepto o dar consejos de seguridad; "
            "no inventes nada fuera de esto y de los datos; si lo usas, añade al final una línea 📚 <i>Fuente: base de conocimiento</i>):\n"
            + "\n".join(f"- {f['contenido']}" for f in contexto_rag)
        )
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if gemini_key or openai_key:
        prompt = (
            "Eres un asistente que explica el riesgo sísmico a ciudadanos sin conocimientos técnicos. "
            "Responde en español claro usando EXCLUSIVAMENTE etiquetas HTML de Telegram (<b>negrita</b> e <i>cursiva</i>). "
            "PROHIBIDO usar asteriscos (*) o Markdown. Si escribes símbolos menor que, usa &lt;.\n\n"
            "REGLA CRÍTICA DE HONESTIDAD: Si 'sismos_cercanos_200km' es 0, NUNCA digas que hay un enjambre sísmico activo; "
            "explica que no hay temblores cercanos y que la variación se debe únicamente a sismos de profundidad intermedia, a actividad regional o al cálculo estadístico del modelo. "
            "NUNCA llames «profundos» a los sismos de 70–150 km: son de profundidad intermedia.\n\n"
            "Usa EXACTAMENTE esta estructura línea por línea (dejando una línea en blanco entre bloques):\n"
            "🔎 <b>Tu consulta:</b> [Resumen breve de lo que preguntó el usuario]\n"
            "📍 <b>Zona analizada:</b> [Nombre exacto de la zona]\n"
            "🚦 <b>Nivel de actividad actual:</b> [Copia exactamente el 'estado' proporcionado en los datos]\n"
            "⚠️ <b>Probabilidad de terremoto fuerte (M6.5+ esta semana):</b> <b>[prob_7d]</b> — En esta zona lo habitual en una semana tranquila es [habitual_7d] ([equivalencia en semanas]), por lo que hoy está a <b>[multiplicador]x</b> de su nivel normal.\n"
            "📊 <b>¿En qué se basa este resultado? (últimas 96 horas):</b>\n"
            "• Sismos cercanos (&lt;200 km): [sismos_cercanos_200km] ([moderados M4.5+] moderados M4.5+ y [leves] leves)\n"
            "• Sismos en la región (&lt;500 km, incluye los cercanos): [sismos_regionales_500km]\n"
            "• Sismos de profundidad intermedia (70–150 km, dentro de 500 km): [sismos_prof_intermedia_70_150km_dentro_de_500km]\n"
            "💡 <b>En palabras simples:</b> [1 o 2 frases muy claras y coherentes con los números reales]\n\n"
            "Si el usuario pregunta por varias zonas o por las zonas más activas, muestra el Top 3 ordenado por mayor probabilidad real (prob_7d) usando ese mismo formato.\n\n"
            "Datos calculados en tiempo real (ordenados de mayor a menor probabilidad real):\n"
            + "\n".join(resumen_zonas)
            + bloque_rag
            + f"\n\nPregunta del usuario: {texto_usuario}"
        )
        respuesta, errores = llamar_llm(prompt)

        if not respuesta:
            respuesta = "⚠️ El asistente de IA tardó demasiado en responder. Inténtalo de nuevo en un minuto. (Probé: " + ", ".join(errores) + ")"
    else:
        respuesta = "Falta configurar OPENAI_API_KEY o GEMINI_API_KEY en Render."

    enviar_respuesta_tg(tg_token, chat_id, msg_id, respuesta)


RAG_DIR = Path(__file__).resolve().parent.parent / "rag" / "conocimiento"
EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"


def generar_embedding(texto: str, tarea: str) -> list:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and os.getenv("RAG_EMBEDDINGS", "gemini").lower() == "openai":  # mismo proveedor para indexar y consultar (768 dim); reindexar si se cambia
        r = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_key}"},
            json={
                "model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                "input": texto[:8000],
                "dimensions": 768,
            },
            timeout=30,
        )
        if not r.ok:
            raise RuntimeError(f"embeddings openai {r.status_code}: {r.text[:200]}")
        return r.json()["data"][0]["embedding"]
    r = requests.post(
        EMBED_URL,
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={
            "content": {"parts": [{"text": texto[:8000]}]},
            "taskType": tarea,
            "outputDimensionality": 768,
        },
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"embeddings gemini {r.status_code}: {r.text[:200]}")
    return r.json()["embedding"]["values"]


def recuperar_conocimiento(pregunta: str, sb, k: int = 3, min_similitud: float = 0.55) -> list:
    """RAG: devuelve los fragmentos de la base de conocimiento más cercanos a la pregunta."""
    try:
        vec = generar_embedding(pregunta, "RETRIEVAL_QUERY")
        filas = sb.rpc("buscar_conocimiento", {"query_embedding": vec, "match_count": k}).execute().data or []
        return [f for f in filas if f.get("similitud", 0) >= min_similitud]
    except Exception:
        return []  # sin RAG el bot sigue funcionando con los datos en vivo


@app.api_route("/rag/ingestar", methods=["GET", "POST"])
def ingestar_conocimiento(key: str = Query(default="")):
    cron_secret = os.getenv("CRON_SECRET", "")
    if cron_secret and key != cron_secret:
        raise HTTPException(status_code=401, detail="Clave inválida")
    try:
        return _ingestar()
    except Exception as e:
        print(f"[rag] {type(e).__name__}: {e}", flush=True)
        return {"error": f"{type(e).__name__}: {e}"[:400]}


def _ingestar():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    total = 0
    for md in sorted(RAG_DIR.glob("*.md")):
        for trozo in re.split(r"^## ", md.read_text(encoding="utf-8"), flags=re.M)[1:]:
            titulo, _, cuerpo = trozo.partition("\n")
            contenido = f"{titulo.strip()}: {cuerpo.strip()}"
            sb.table("conocimiento").upsert(
                {
                    "fuente": md.name,
                    "titulo": titulo.strip(),
                    "contenido": contenido,
                    "embedding": generar_embedding(contenido, "RETRIEVAL_DOCUMENT"),
                },
                on_conflict="fuente,titulo",
            ).execute()
            total += 1
    return {"fragmentos_indexados": total}


def vincular_telegram(tg_token: str, chat_id: str, codigo: str) -> None:
    if codigo:
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        r = sb.table("vinculos_telegram").update({"chat_id": chat_id}).eq("codigo", codigo).execute()
        ok = bool(r.data)
        texto = ("✅ Telegram vinculado. Recibirás aquí tus alertas sísmicas. "
                 "También puedes preguntarme por cualquier zona.") if ok else                 "⚠️ Código no válido. Genera uno nuevo en el panel web."
    else:
        texto = "👋 Hola. Para recibir alertas, vincula este chat desde el panel web: https://ultimo-modulo.vercel.app"
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                  json={"chat_id": chat_id, "text": texto}, timeout=15)


@app.get("/bot-info")
def bot_info():
    r = requests.get(f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/getMe", timeout=10)
    return {"username": r.json().get("result", {}).get("username", "")}


@app.post("/telegram-webhook")
def telegram_webhook(update: dict, background_tasks: BackgroundTasks):
    message = update.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    texto_usuario = (message.get("text") or "").strip()

    if not texto_usuario or not chat_id:
        return {"ok": True}

    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]

    if texto_usuario.startswith("/start"):
        codigo = texto_usuario.partition(" ")[2].strip()
        vincular_telegram(tg_token, chat_id, codigo)
        return {"ok": True}

    msg_id = enviar_progreso_tg(
        tg_token,
        chat_id,
        "⏳ 1/2 Consultando la red sísmica y calculando probabilidades...",
    )

    background_tasks.add_task(
        procesar_consulta_telegram, tg_token, chat_id, texto_usuario, msg_id
    )
    return {"ok": True}