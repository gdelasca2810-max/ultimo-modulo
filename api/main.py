from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="API Riesgo Sísmico Relativo (USGS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "modelo_sismos.joblib"
artefacto = joblib.load(MODEL_PATH)


class ConsultaRiesgo(BaseModel):
    zona: str
    loc200_n96_con: int
    loc200_n96_sin45: int
    loc500_n96_con: int
    lat_n96_prof70_150: int


@app.get("/health")
def health():
    return {"status": "ok", "metricas_test": artefacto["metricas_test"]}


@app.post("/predict")
def predecir(datos: ConsultaRiesgo):
    if datos.zona not in artefacto["tasas_zona"]:
        raise HTTPException(status_code=404, detail=f"Zona no reconocida: {datos.zona}")

    tasa_base = artefacto["tasas_zona"][datos.zona]
    media_96h = artefacto["medias_96h"][datos.zona]

    fila = pd.DataFrame([{
        "loc200_n96_con": datos.loc200_n96_con,
        "loc200_n96_sin45": datos.loc200_n96_sin45,
        "loc500_n96_con": datos.loc500_n96_con,
        "lat_n96_prof70_150": datos.lat_n96_prof70_150,
        "media_96h_zona": media_96h,
    }])

    prob_7d = float(artefacto["modelo"].predict_proba(fila)[0][1])
    multiplicador = round(prob_7d / tasa_base, 2) if tasa_base > 0 else 1.0

    return {
        "zona": datos.zona,
        "probabilidad_m65_7d": round(prob_7d, 5),
        "tasa_historica_7d": round(tasa_base, 5),
        "multiplicador_riesgo": multiplicador,
        "alerta_enjambre": multiplicador >= 2.0,
    }