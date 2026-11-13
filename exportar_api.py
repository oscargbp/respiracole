"""
RespiraCole · API de datos abiertos
===================================

Publica los resultados de RespiraCole como ficheros JSON estándar, para que
otras aplicaciones (plataformas escolares, apps de familias, la propia
Comunidad) puedan integrarlos sin trabajo adicional:

  api/estaciones_airquality.json
      Una entidad AirQualityObserved (Smart Data Models, formato key-values)
      por estación de las dos redes, con la métrica del semáforo del último
      día disponible y su nivel del Índice de Calidad del Aire. Se valida
      contra el esquema oficial del modelo.
  api/centros.geojson
      Los centros escolares como GeoJSON (FeatureCollection), con exposición
      media 2021-2025, fiabilidad de la estimación, estación de referencia y
      días en ámbar y rojo del curso 2024-25.
  api/indice.json
      Descripción de los recursos, licencia y fuentes.

Requiere haber ejecutado preparar_datos.py, analizar_equidad.py, predecir.py
y episodios.py.

Uso:
    pip install jsonschema requests
    python exportar_api.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import requests

from predecir import CONTS, nivel
from preparar_datos import CARPETA_SALIDA, BASE

ESQUEMA = ("https://raw.githubusercontent.com/smart-data-models/dataModel.Environment/"
           "master/AirQualityObserved/schema.json")
NIVELES = {1: "buena", 2: "razonablemente buena", 3: "regular", 4: "desfavorable",
           5: "muy desfavorable", 6: "extremadamente desfavorable"}
NOMBRE_SDM = {"NO2": "no2", "O3": "o3", "PM10": "pm10", "PM25": "pm25"}
CARPETA_API = BASE / "api"


def estaciones_airquality() -> list[dict]:
    est = pd.read_csv(CARPETA_SALIDA / "estaciones.csv", dtype={"codigo": str})
    m = pd.read_csv(CARPETA_SALIDA / "metricas_diarias.csv", dtype={"estacion_codigo": str},
                    parse_dates=["fecha"])
    fecha = m["fecha"].max()
    dia = m[m["fecha"] == fecha]
    entidades = []
    for _, e in est.iterrows():
        valores = dia[dia["estacion_codigo"] == e["codigo"]]
        if valores.empty:
            continue
        ent = {
            "id": f"urn:ngsi-ld:AirQualityObserved:RespiraCole:{e['codigo']}:{fecha:%Y-%m-%d}",
            "type": "AirQualityObserved",
            "dateObserved": f"{fecha:%Y-%m-%d}T00:00:00+01:00",
            "location": {"type": "Point", "coordinates": [round(e["lon"], 6), round(e["lat"], 6)]},
            "address": {"addressCountry": "ES", "addressRegion": "Comunidad de Madrid",
                        "addressLocality": e["municipio"]},
            "name": e["direccion"] if e["red"] == "ayuntamiento" else e["municipio"],
            "description": ("Métrica del semáforo RespiraCole: NO2 y O3, máximo horario de 9:00 "
                            "a 17:00; PM10 y PM2,5, media diaria. µg/m³."),
            "typeofLocation": "outdoor",
            "source": ("https://datos.madrid.es" if e["red"] == "ayuntamiento"
                       else "https://datos.comunidad.madrid"),
            "dataProvider": "RespiraCole",
        }
        peor = 0
        for _, v in valores.iterrows():
            ent[NOMBRE_SDM[v["contaminante"]]] = round(float(v["valor"]), 1)
            peor = max(peor, int(nivel(v["contaminante"], v["valor"])))
        ent["airQualityIndex"] = peor
        ent["airQualityLevel"] = NIVELES[peor]
        entidades.append(ent)
    return entidades


def validar(entidades: list[dict]) -> str:
    import jsonschema

    try:
        esquema = requests.get(ESQUEMA, timeout=60).json()
    except requests.RequestException as e:
        return f"No he podido descargar el esquema para validar: {e}"
    try:
        validador = jsonschema.Draft202012Validator(esquema, registry=_registro())
        errores = [(ent["id"], err.message) for ent in entidades
                   for err in validador.iter_errors(ent)]
    except Exception as e:  # una referencia del esquema no se pudo descargar
        return f"No he podido completar la validación: {e}"
    if errores:
        return "Errores de validación:\n" + "\n".join(f"  {i}: {m}" for i, m in errores[:10])
    return f"Validación AirQualityObserved: {len(entidades)} entidades correctas."


def _registro():
    """Resuelve las referencias del esquema ($ref) descargándolas de GitHub."""
    from referencing import Registry, Resource

    def recuperar(uri: str):
        # Las referencias a smart-data-models.github.io se sirven también desde GitHub
        uri = uri.replace("https://smart-data-models.github.io/data-models/",
                          "https://raw.githubusercontent.com/smart-data-models/data-models/master/")
        return Resource.from_contents(requests.get(uri, timeout=60).json())

    return Registry(retrieve=recuperar)


def centros_geojson() -> dict:
    eq = pd.read_csv(CARPETA_SALIDA / "centros_equidad.csv", dtype={"codigo": str})
    ep = pd.read_csv(CARPETA_SALIDA / "episodios_centros.csv", dtype={"codigo": str})
    ep = ep[ep["curso"] == "2024-25"][["codigo", "dias_ambar", "dias_rojo"]]
    df = eq.merge(ep, on="codigo", how="left")
    feats = []
    for _, c in df.iterrows():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(c["lon"], 6), round(c["lat"], 6)]},
            "properties": {
                "codigo": c["codigo"], "nombre": c["nombre"], "etapa": c["categoria"],
                "titularidad": c["titularidad"], "municipio": c["municipio"],
                "distrito": c["distrito"] if isinstance(c["distrito"], str) else None,
                "no2_media_2021_2025": c["NO2_media"], "pm10_media_2021_2025": c["PM10_media"],
                "pm25_media_2021_2025": c["PM25_media"],
                "estacion_no2": str(c["NO2_estacion"]).split(".")[0],
                "distancia_estacion_no2_km": c["NO2_distancia_km"],
                "fiabilidad": c["fiabilidad"],
                "dias_ambar_2024_25": None if pd.isna(c["dias_ambar"]) else int(c["dias_ambar"]),
                "dias_rojo_2024_25": None if pd.isna(c["dias_rojo"]) else int(c["dias_rojo"]),
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def main() -> int:
    CARPETA_API.mkdir(exist_ok=True)
    ent = estaciones_airquality()
    print(validar(ent))
    (CARPETA_API / "estaciones_airquality.json").write_text(
        json.dumps(ent, ensure_ascii=False, indent=1), encoding="utf-8")
    geo = centros_geojson()
    (CARPETA_API / "centros.geojson").write_text(
        json.dumps(geo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    indice = {
        "nombre": "RespiraCole · datos abiertos",
        "licencia": "CC BY 4.0. Cita: RespiraCole, a partir de datos de la Comunidad de Madrid "
                    "y del Ayuntamiento de Madrid.",
        "recursos": [
            {"ruta": "api/estaciones_airquality.json", "formato": "JSON (NGSI-LD key-values)",
             "modelo": "Smart Data Models · AirQualityObserved", "entidades": len(ent)},
            {"ruta": "api/centros.geojson", "formato": "GeoJSON", "entidades": len(geo["features"])},
        ],
        "fuentes": ["https://datos.comunidad.madrid", "https://datos.madrid.es"],
    }
    (CARPETA_API / "indice.json").write_text(json.dumps(indice, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"API: {len(ent)} estaciones y {len(geo['features']):,} centros exportados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
