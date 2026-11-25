#!/usr/bin/env python
"""Descarga el catálogo crudo de USGS ComCat (API FDSN pública, sin clave) a CSV.

No hay datos crudos en el repo: este script permite regenerarlos. NO se ejecuta
automáticamente. Parámetros que el auditor debe revisar y fijar explícitamente
(los valores por defecto son una PROPUESTA, no la configuración original del CSV):

  --start / --end     Rango. El CSV del modelo cubre t_eval 2016-01-05 .. 2026-09-26; las
                      ventanas pasadas de 96 h y futuras de 30 d exigen descargar desde
                      2015-12-28 hasta 2026-10-31 como mínimo.
  --minmag            Umbral de magnitud. DESCONOCIDO para el CSV original (la API de
                      producción usa 2.5). Cámbielo hasta reproducir los conteos del CSV.
  --eventtype         earthquake (excluye explosiones/canteras).
  --max-por-consulta  USGS rechaza consultas con más de 20000 eventos; el script divide
                      el intervalo recursivamente hasta cumplirlo.

Ejemplo:
    python scripts/descargar_catalogo_usgs.py --start 2015-12-28 --end 2026-11-01 \
        --minmag 2.5 --out data/raw/usgs_catalogo.csv

Salida (UTF-8): id,time_utc,latitude,longitude,depth_km,mag,magType,place,type,status,updated_utc
Guarda también <out>.meta.json con parámetros, fecha de descarga y número de filas.
La carpeta data/raw/ queda fuera de git (el .gitignore ya excluye *.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"  # servicio público, sin credenciales
CAMPOS = ["id", "time_utc", "latitude", "longitude", "depth_km", "mag", "magType",
          "place", "type", "status", "updated_utc"]


def iso(ms: int | None) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def consultar(ini: datetime, fin: datetime, args) -> list[dict]:
    params = {
        "format": "geojson",
        "starttime": ini.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": fin.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": args.minmag,
        "eventtype": args.eventtype,
        "orderby": "time-asc",
        "limit": args.max_por_consulta,
    }
    for intento in range(1, args.reintentos + 1):
        try:
            r = requests.get(ENDPOINT, params=params, timeout=args.timeout)
            if r.status_code == 400 and "exceeds search limit" in r.text.lower():
                return []  # señal para dividir
            r.raise_for_status()
            return r.json().get("features", [])
        except requests.RequestException as e:
            print(f"  intento {intento}/{args.reintentos} falló: {e}", file=sys.stderr)
            time.sleep(2 * intento)
    raise RuntimeError(f"No se pudo descargar {params['starttime']}..{params['endtime']}")


def descargar_rango(ini: datetime, fin: datetime, args, vistos: set, escritor) -> int:
    feats = consultar(ini, fin, args)
    if len(feats) >= args.max_por_consulta or (not feats and (fin - ini) > timedelta(hours=1)):
        # Posible truncado o rechazo por tamaño: dividir a la mitad.
        if (fin - ini) <= timedelta(minutes=1):
            raise RuntimeError(f"Intervalo mínimo saturado en {ini}: suba --max-por-consulta o el minmag.")
        medio = ini + (fin - ini) / 2
        return (descargar_rango(ini, medio, args, vistos, escritor)
                + descargar_rango(medio, fin, args, vistos, escritor))
    n = 0
    for f in feats:
        if f["id"] in vistos:
            continue
        p, c = f["properties"], f["geometry"]["coordinates"]
        vistos.add(f["id"])
        escritor.writerow([f["id"], iso(p.get("time")), c[1], c[0], c[2], p.get("mag"),
                           p.get("magType"), p.get("place"), p.get("type"), p.get("status"),
                           iso(p.get("updated"))])
        n += 1
    time.sleep(args.pausa)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="AAAA-MM-DD (UTC, inclusive)")
    ap.add_argument("--end", required=True, help="AAAA-MM-DD (UTC, exclusive)")
    ap.add_argument("--minmag", type=float, default=2.5)
    ap.add_argument("--eventtype", default="earthquake")
    ap.add_argument("--max-por-consulta", type=int, default=20000)
    ap.add_argument("--chunk-dias", type=int, default=30)
    ap.add_argument("--pausa", type=float, default=0.5, help="segundos entre consultas")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--reintentos", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("data/raw/usgs_catalogo.csv"))
    args = ap.parse_args()

    ini = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    fin = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    if fin <= ini:
        print("--end debe ser posterior a --start", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    vistos: set = set()
    total = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CAMPOS)
        t = ini
        while t < fin:
            t2 = min(t + timedelta(days=args.chunk_dias), fin)
            n = descargar_rango(t, t2, args, vistos, w)
            total += n
            print(f"{t.date()} -> {t2.date()}: {n} eventos (acumulado {total})")
            t = t2
    meta = {"descargado_utc": datetime.now(timezone.utc).isoformat(), "endpoint": ENDPOINT,
            "start": args.start, "end": args.end, "minmag": args.minmag, "eventtype": args.eventtype,
            "filas": total, "columnas": CAMPOS}
    args.out.with_suffix(args.out.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Listo: {total} eventos en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
