#!/usr/bin/env python
"""Construye un dataset ML reproducible desde un catálogo USGS crudo.

Especificación v4: las features reproducen la semántica operativa de la API
(M>=2.5, 96 h pasadas, radios 200/500 km y profundidad 70--150 km a <=500 km).
El target explícito es un M>=6.5 a <=500 km en los siete días posteriores.
No modifica el CSV ni el joblib de producción.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "raw" / "usgs_catalogo_m25_2015_2026.csv"
DEFAULT_MAIN = ROOT / "api" / "main.py"
DEFAULT_OUT = ROOT / "data" / "candidate" / "zona_dia_usgs_v4.csv"


def extraer_zonas(ruta: Path) -> dict[str, tuple[float, float]]:
    """Extrae ZONAS_COORDS sin importar FastAPI, secretos ni el modelo."""
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and any(
            isinstance(destino, ast.Name) and destino.id == "ZONAS_COORDS" for destino in nodo.targets
        ):
            valor = ast.literal_eval(nodo.value)
            return {str(nombre): (float(coord[0]), float(coord[1])) for nombre, coord in valor.items()}
    raise ValueError(f"No encontré ZONAS_COORDS en {ruta}")


def cargar_zonas_csv(ruta: Path) -> dict[str, tuple[float, float]]:
    """Carga un export mínimo de Supabase: zona, lat_centro, lon_centro."""
    tabla = pd.read_csv(ruta)
    requeridas = {"zona", "lat_centro", "lon_centro"}
    if faltantes := (requeridas - set(tabla.columns)):
        raise ValueError(f"El CSV de zonas no contiene: {sorted(faltantes)}")
    tabla = tabla.dropna(subset=list(requeridas))
    if tabla["zona"].duplicated().any():
        raise ValueError("El CSV de zonas contiene nombres duplicados")
    return {
        str(fila.zona): (float(fila.lat_centro), float(fila.lon_centro))
        for fila in tabla[["zona", "lat_centro", "lon_centro"]].itertuples(index=False)
    }


def distancias_km(eventos: pd.DataFrame, lat_centro: float, lon_centro: float) -> np.ndarray:
    radio_tierra = 6371.0
    lat1 = np.radians(eventos["latitude"].to_numpy(dtype=float))
    lon1 = np.radians(eventos["longitude"].to_numpy(dtype=float))
    lat2, lon2 = np.radians([lat_centro, lon_centro])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * radio_tierra * np.arcsin(np.sqrt(a))


def contar_pasado(tiempos: np.ndarray, evaluaciones: np.ndarray, horas: int) -> np.ndarray:
    """Número de eventos en [t-horas, t), sin información futura."""
    return (
        np.searchsorted(tiempos, evaluaciones, side="left")
        - np.searchsorted(tiempos, evaluaciones - np.timedelta64(horas, "h"), side="left")
    ).astype(int)


def existe_futuro(tiempos: np.ndarray, evaluaciones: np.ndarray, dias: int) -> np.ndarray:
    """Target: existe evento en [t, t+días), sin incluir el propio instante de evaluación."""
    return (
        np.searchsorted(tiempos, evaluaciones + np.timedelta64(dias, "D"), side="left")
        - np.searchsorted(tiempos, evaluaciones, side="left")
        > 0
    ).astype(int)


def fechas_diarias(inicio: str, fin: str, hora: str) -> pd.DatetimeIndex:
    h, m = (int(valor) for valor in hora.split(":", maxsplit=1))
    t0 = datetime.fromisoformat(inicio).replace(tzinfo=timezone.utc) + timedelta(hours=h, minutes=m)
    t1 = datetime.fromisoformat(fin).replace(tzinfo=timezone.utc) + timedelta(hours=h, minutes=m)
    return pd.date_range(t0, t1, freq="D", tz="UTC")


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as flujo:
        for bloque in iter(lambda: flujo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def construir(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    zonas = cargar_zonas_csv(args.zonas_csv) if args.zonas_csv else extraer_zonas(args.main_py)
    crudo = pd.read_csv(args.raw)
    requeridas = {"id", "time_utc", "latitude", "longitude", "depth_km", "mag"}
    if faltantes := (requeridas - set(crudo.columns)):
        raise ValueError(f"Faltan columnas del catálogo: {sorted(faltantes)}")
    crudo["time_utc"] = pd.to_datetime(crudo["time_utc"], utc=True)
    crudo = crudo.dropna(subset=list(requeridas - {"id"})).drop_duplicates("id", keep="last")
    if "type" in crudo:
        crudo = crudo[crudo["type"].eq("earthquake")]
    crudo = crudo.sort_values("time_utc").reset_index(drop=True)

    evaluaciones = fechas_diarias(args.start, args.end, args.hora_utc)
    minimo = evaluaciones.min().to_pydatetime() - timedelta(hours=args.ventana_horas)
    maximo = evaluaciones.max().to_pydatetime() + timedelta(days=args.horizonte_dias)
    if crudo["time_utc"].min() > minimo or crudo["time_utc"].max() < maximo:
        raise ValueError(f"Cobertura insuficiente; se necesita {minimo}..{maximo}")
    evaluaciones_np = evaluaciones.to_numpy(dtype="datetime64[ns]")
    mascara_feature = crudo["mag"].ge(args.min_mag_feature).to_numpy()
    mascara_target_mag = crudo["mag"].ge(args.min_mag_target).to_numpy()
    filas: list[pd.DataFrame] = []

    for zona, (lat, lon) in sorted(zonas.items()):
        distancia = distancias_km(crudo, lat, lon)
        local_mask = mascara_feature & (distancia <= args.radio_local_km)
        regional_mask = mascara_feature & (distancia <= args.radio_regional_km)
        profundo_mask = regional_mask & crudo["depth_km"].between(
            args.prof_min_km, args.prof_max_km, inclusive="both"
        ).to_numpy()
        leve_mask = local_mask & crudo["mag"].lt(args.umbral_fuerte).to_numpy()
        target_mask = mascara_target_mag & (distancia <= args.radio_target_km)

        def tiempos(mascara: np.ndarray) -> np.ndarray:
            return crudo.loc[mascara, "time_utc"].to_numpy(dtype="datetime64[ns]")

        filas.append(pd.DataFrame({
            "zona": zona,
            "t_eval_utc": evaluaciones,
            "lat_centro": lat,
            "lon_centro": lon,
            "loc200_n96_con": contar_pasado(tiempos(local_mask), evaluaciones_np, args.ventana_horas),
            "loc200_n96_sin45": contar_pasado(tiempos(leve_mask), evaluaciones_np, args.ventana_horas),
            "loc500_n96_con": contar_pasado(tiempos(regional_mask), evaluaciones_np, args.ventana_horas),
            "lat_n96_prof70_150": contar_pasado(tiempos(profundo_mask), evaluaciones_np, args.ventana_horas),
            "y_M65_7d": existe_futuro(tiempos(target_mask), evaluaciones_np, args.horizonte_dias),
        }))

    dataset = pd.concat(filas, ignore_index=True)
    meta = {
        "version_dataset": "usgs_v4_operational_semantics",
        "creado_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(args.raw), "raw_sha256": sha256(args.raw),
        "raw_rows": int(len(crudo)),
        "rango_raw_utc": [str(crudo["time_utc"].min()), str(crudo["time_utc"].max())],
        "zonas": int(len(zonas)), "filas": int(len(dataset)), "positivos": int(dataset["y_M65_7d"].sum()),
        "fuente_zonas": str(args.zonas_csv) if args.zonas_csv else str(args.main_py),
        "rango_evaluacion_utc": [str(evaluaciones.min()), str(evaluaciones.max())],
        "features": ["loc200_n96_con", "loc200_n96_sin45", "loc500_n96_con", "lat_n96_prof70_150"],
        "target": "Existe M>=6.5 a <=500 km en [t_eval_utc, t_eval_utc + 7 días)",
        "parametros": vars(args),
    }
    meta["parametros"] = {clave: str(valor) if isinstance(valor, Path) else valor for clave, valor in meta["parametros"].items()}
    return dataset, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--main-py", type=Path, default=DEFAULT_MAIN)
    ap.add_argument("--zonas-csv", type=Path, default=None,
                    help="Export de zonas_sismicas con zona,lat_centro,lon_centro; prioriza producción")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start", default="2016-01-05")
    ap.add_argument("--end", default="2026-09-26")
    ap.add_argument("--hora-utc", default="00:37")
    ap.add_argument("--min-mag-feature", type=float, default=2.5)
    ap.add_argument("--min-mag-target", type=float, default=6.5)
    ap.add_argument("--ventana-horas", type=int, default=96)
    ap.add_argument("--horizonte-dias", type=int, default=7)
    ap.add_argument("--radio-local-km", type=float, default=200.0)
    ap.add_argument("--radio-regional-km", type=float, default=500.0)
    ap.add_argument("--radio-target-km", type=float, default=500.0)
    ap.add_argument("--prof-min-km", type=float, default=70.0)
    ap.add_argument("--prof-max-km", type=float, default=150.0)
    ap.add_argument("--umbral-fuerte", type=float, default=4.5)
    args = ap.parse_args()
    if args.radio_local_km > args.radio_regional_km:
        raise SystemExit("--radio-local-km no puede superar --radio-regional-km")
    dataset, metadata = construir(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False)
    meta = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset: {args.out} | filas={metadata['filas']} | positivos={metadata['positivos']}")
    print(f"Metadata: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
