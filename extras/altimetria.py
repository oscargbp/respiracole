"""Altitud de cada nudo de la red peatonal y ciclista, para el perfil de desniveles del camino al cole.

Fuente: Ayuntamiento de Madrid, cartografía municipal por distritos a escala 1:5000
        (capa 07_punto_acotado_a: puntos acotados con su cota en metros).
Entrada: datos/rutas_red.json (rutas_red.py)
Salida:  añade a datos/rutas_red.json el campo nodos_z (altitud en decímetros, entero)
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
CARTO = BASE / "raw" / "rutas" / "carto"
COTAS = CARTO / "cotas"
DATOS = BASE / "datos"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
CAPA = "07_punto_acotado_a"


def descargar():
    COTAS.mkdir(parents=True, exist_ok=True)
    cat = json.loads((CARTO / "c5000.json").read_text(encoding="utf-8"))
    for i, r in enumerate(x for x in cat["resources"] if x["format"].upper() == "ZIP"):
        destino = COTAS / f"{i}.gpkg"
        if destino.exists():
            continue
        for intento in range(5):
            try:
                z = requests.get(r["url"], headers=UA, timeout=300)
                z.raise_for_status()
                zf = zipfile.ZipFile(io.BytesIO(z.content))
                break
            except (requests.RequestException, zipfile.BadZipFile):
                time.sleep(5 * (intento + 1))
        else:
            print("no se pudo descargar", r["url"])
            continue
        nombres = [n for n in zf.namelist() if CAPA in n and not n.endswith(".xml")]
        if not nombres:
            continue
        tmp = COTAS / f"tmp_{i}"
        zf.extractall(tmp, nombres)
        shp = next(tmp.rglob(f"*{CAPA}.shp"))
        g = gpd.read_file(shp)[["TEXTO", "geometry"]]
        g.to_file(destino, driver="GPKG")
        for f in sorted(tmp.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        tmp.rmdir()
        print(i, len(g), "puntos acotados")
        time.sleep(2)


def cotas():
    trozos = [gpd.read_file(f) for f in sorted(COTAS.glob("*.gpkg"))]
    g = pd.concat(trozos, ignore_index=True)
    z = pd.to_numeric(g.TEXTO.astype(str).str.replace(",", "."), errors="coerce")
    # el rótulo está desplazado 3 m al nordeste del punto que acota
    p = pd.DataFrame({"x": g.geometry.x - 3, "y": g.geometry.y - 3, "z": z}).dropna()
    p = p[p.z.between(530, 900)].drop_duplicates(["x", "y"])
    # se descartan cotas que no encajan con su entorno (azoteas, pasos elevados, erratas)
    a = cKDTree(p[["x", "y"]].values)
    _, idx = a.query(p[["x", "y"]].values, k=9)
    med = np.median(p.z.values[idx[:, 1:]], axis=1)
    return p[np.abs(p.z.values - med) < 12]


def altitud_nodos(p, xs, ys, k=6):
    a = cKDTree(p[["x", "y"]].values)
    d, idx = a.query(np.c_[xs, ys], k=k)
    w = 1 / np.maximum(d, 5) ** 2
    z = (w * p.z.values[idx]).sum(axis=1) / w.sum(axis=1)
    return z, d[:, 0]


if __name__ == "__main__":
    if "--sin-descarga" not in sys.argv:
        descargar()
    p = cotas()
    red = json.loads((DATOS / "rutas_red.json").read_text(encoding="utf-8"))
    z, d0 = altitud_nodos(p, np.array(red["nodos_x"]), np.array(red["nodos_y"]))
    red["nodos_z"] = [int(round(v * 10)) for v in z]
    (DATOS / "rutas_red.json").write_text(json.dumps(red, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"puntos acotados: {len(p):,} | nudos: {len(z):,} | altitud {z.min():.0f}-{z.max():.0f} m "
          f"| distancia al punto acotado más cercano: mediana {np.median(d0):.0f} m, p95 {np.percentile(d0, 95):.0f} m")
