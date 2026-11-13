"""Descarga la topología de Metro y Metro Ligero (CRTM, ArcGIS) y los GTFS."""
import json, sys, time, zipfile, io
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent / "raw" / "metro"
SRV = "https://services5.arcgis.com/UxADft6QPcvFyDU1/arcgis/rest/services"
CAPAS = {"M4_Red": range(6), "M10_Red": range(6)}
GTFS = {"metro": "5c7f2951962540d69ffe8f640d94c246", "ligero": "aaed26cc0ff64b0c947ac0bc3e033196"}


def capa(servicio, i):
    info = requests.get(f"{SRV}/{servicio}/FeatureServer/{i}", params={"f": "json"}, timeout=60).json()
    feats, off = [], 0
    while True:
        r = requests.get(f"{SRV}/{servicio}/FeatureServer/{i}/query", timeout=120, params={
            "where": "1=1", "outFields": "*", "f": "geojson", "outSR": 4326,
            "resultOffset": off, "resultRecordCount": 1000}).json()
        feats += r["features"]
        if len(r["features"]) < 1000:
            break
        off += 1000
    return info["name"], {"type": "FeatureCollection", "features": feats}


if __name__ == "__main__":
    (BASE / "topo").mkdir(parents=True, exist_ok=True)
    for s, idx in CAPAS.items():
        for i in idx:
            nombre, fc = capa(s, i)
            (BASE / "topo" / f"{nombre}.geojson").write_text(json.dumps(fc), encoding="utf-8")
            print(nombre, len(fc["features"]))
    for k, item in GTFS.items():
        z = requests.get(f"https://www.arcgis.com/sharing/rest/content/items/{item}/data", timeout=120).content
        zipfile.ZipFile(io.BytesIO(z)).extractall(BASE / f"gtfs_{k}")
        print("GTFS", k, len(z))
