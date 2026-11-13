"""
RespiraCole · Red peatonal y ciclista de Madrid capital
=======================================================

Construye, solo con datos oficiales del Ayuntamiento de Madrid, la red por la que
se calculan los caminos al colegio a pie y en bici, y la prepara para la web.

Fuentes (Ayuntamiento de Madrid):
  - Callejero oficial · Subviales vigentes (geoportal.madrid.es): superficie de
    cada calle. Las calles no vienen como ejes, así que se obtienen sus ejes
    "adelgazando" esa superficie (esqueletización de una imagen de 4 m).
  - Cartografía municipal 1:5000 (datos.madrid.es): sendas de parques, caminos,
    rampas y pasarelas peatonales, y zonas ajardinadas.
  - Infraestructura ciclista (geoportal.madrid.es): carriles bici, ciclocarriles
    y Anillo Verde Ciclista.
  - Aforos de tráfico (IMD 2020) y accidentes con peatones 2023-2025 (datos.madrid.es).
  - Callejero oficial · direcciones vigentes, con coordenadas de cada portal.

Cada tramo de la red lleva: longitud, % por zona verde o senda, tráfico de la
calle (IMD), atropellos en sus proximidades y tipo de vía ciclista.

Genera:
  datos/rutas_red.json          Red de nodos y tramos (para el calculador de rutas)
  datos/rutas_direcciones.json  Portales de la capital, agrupados por calle
  datos/rutas_resumen.txt       Cifras de la red

Uso:
    pip install geopandas rasterio scikit-image sknw
    python rutas_red.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import LineString
from skimage.morphology import skeletonize

from preparar_datos import CARPETA_RAW, CARPETA_SALIDA

RES = 4.0                   # metros por píxel
BUFFER_SENDAS = 2.5         # anchura (semi) con la que se dibujan sendas y caminos
RADIO_AFORO = 30            # un tramo hereda el tráfico del aforo a menos de esta distancia
RADIO_ATROPELLO = 20
RADIO_BICI = 8
RAMA_MINIMA = 12            # se eliminan ramas muertas más cortas (artefactos del adelgazado)
EXCLUIR = re.compile(r"Autov|Autopista", re.I)

R = CARPETA_RAW / "rutas"


def capa_carto(nombre: str) -> gpd.GeoDataFrame:
    trozos = [gpd.read_file(f) for f in sorted((R / "carto" / "capas").glob(f"*__{nombre}.shp"))]
    g = pd.concat(trozos, ignore_index=True)
    return gpd.GeoDataFrame(g, geometry="geometry", crs=trozos[0].crs).to_crs(25830)


def cargar_fuentes():
    sub = gpd.read_file(R / "subviales" / "SUBVIALES.shp").to_crs(25830)
    sub = sub[~sub["VIAS"].fillna("").str.contains(EXCLUIR)]
    lineas = pd.concat([capa_carto(n)[["geometry"]] for n in
                        ("11_senda_l", "06_camino_pista_l", "06_rampa_pasarela_l")], ignore_index=True)
    verde = capa_carto("11_zona_ajardinada_p")[["geometry"]]
    bici = gpd.read_file(R / "bici" / "Infraestructura_Ciclista.shp").to_crs(25830)
    return sub, gpd.GeoDataFrame(lineas, crs=25830), verde, bici


def esqueleto(sub, lineas, bici):
    x0, y0, x1, y1 = sub.total_bounds
    x0, y0 = x0 - 50, y0 - 50
    ancho, alto = int((x1 + 50 - x0) / RES), int((y1 + 50 - y0) / RES)
    tr = from_origin(x0, y1 + 50, RES, RES)
    formas = [(g, 1) for g in sub.geometry]
    formas += [(g.buffer(BUFFER_SENDAS), 1) for g in lineas.geometry if g is not None]
    # El Anillo Verde y los carriles bici que van por parques también son transitables
    formas += [(g.buffer(BUFFER_SENDAS), 1) for g in bici.geometry if g is not None]
    mascara = features.rasterize(formas, out_shape=(alto, ancho), transform=tr, dtype="uint8")
    print(f"  imagen {ancho}×{alto}, transitable {mascara.mean():.1%}")
    esq = skeletonize(mascara.astype(bool))
    return esq, tr


def a_grafo(esq, tr) -> nx.MultiGraph:
    """Convierte el esqueleto (píxeles) en una red de nodos y tramos.

    Nodos: píxeles con 1 vecino (final) o 3 o más (cruce), agrupando los
    píxeles de cruce contiguos. Tramos: cadenas de píxeles entre dos nodos.
    """
    from scipy import ndimage

    esq = esq.astype(bool)
    alto, ancho = esq.shape
    vecinos = ndimage.convolve(esq.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    es_nodo = esq & (vecinos != 2)
    etiquetas, n_nodos = ndimage.label(es_nodo, structure=np.ones((3, 3)))
    filas, cols = np.nonzero(es_nodo)
    lab = etiquetas[filas, cols]
    cuenta = np.bincount(lab, minlength=n_nodos + 1)
    cf = np.bincount(lab, weights=filas, minlength=n_nodos + 1) / np.maximum(cuenta, 1)
    cc = np.bincount(lab, weights=cols, minlength=n_nodos + 1) / np.maximum(cuenta, 1)

    a, b, c, d, e, f = tr.a, tr.b, tr.c, tr.d, tr.e, tr.f

    def xy(fila, col):
        return (a * (col + .5) + b * (fila + .5) + c, d * (col + .5) + e * (fila + .5) + f)

    G = nx.MultiGraph()
    for i in range(1, n_nodos + 1):
        x, y = xy(cf[i], cc[i])
        G.add_node(i, x=x, y=y)

    plano = esq.ravel()
    nodo_plano = etiquetas.ravel()
    visitado = np.zeros(plano.size, dtype=bool)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    D = [dr * ancho + dc for dr, dc in offs]
    directos = set()

    def vecinos_de(p):
        r, col = divmod(p, ancho)
        for (dr, dc), dd in zip(offs, D):
            rr, c2 = r + dr, col + dc
            if 0 <= rr < alto and 0 <= c2 < ancho and plano[p + dd]:
                yield p + dd

    for p in (filas * ancho + cols):
        u = nodo_plano[p]
        for q in vecinos_de(p):
            v = nodo_plano[q]
            if v:
                if v != u and (min(u, v), max(u, v)) not in directos:
                    directos.add((min(u, v), max(u, v)))
                    G.add_edge(u, v, pix=[p, q])
                continue
            if visitado[q]:
                continue
            camino, prev, cur = [p, q], p, q
            visitado[q] = True
            fin = None
            while True:
                sig = None
                for s in vecinos_de(cur):
                    if s == prev:
                        continue
                    if nodo_plano[s] and nodo_plano[s] != 0:
                        if len(camino) > 2 or nodo_plano[s] != u:
                            fin = s
                            break
                    elif not visitado[s]:
                        sig = s
                if fin is not None:
                    camino.append(fin)
                    break
                if sig is None:
                    break
                visitado[sig] = True
                camino.append(sig)
                prev, cur = cur, sig
            if fin is not None:
                G.add_edge(u, nodo_plano[fin], pix=camino)

    # Geometría de cada tramo en coordenadas UTM
    for u, v, k, dat in list(G.edges(keys=True, data=True)):
        pix = np.array(dat.pop("pix"))
        r, col = np.divmod(pix, ancho)
        pts = np.column_stack(xy(r.astype(float), col.astype(float)))
        pts[0] = (G.nodes[u]["x"], G.nodes[u]["y"]) if _cerca(pts[0], G.nodes[u]) else (G.nodes[v]["x"], G.nodes[v]["y"])
        pts[-1] = (G.nodes[v]["x"], G.nodes[v]["y"]) if _cerca(pts[-1], G.nodes[v]) else (G.nodes[u]["x"], G.nodes[u]["y"])
        linea = LineString(pts)
        if linea.length < 0.5 and u == v:
            G.remove_edge(u, v, k)
            continue
        G.edges[u, v, k]["geom"] = linea
    return G


def _cerca(p, nodo) -> bool:
    return (p[0] - nodo["x"]) ** 2 + (p[1] - nodo["y"]) ** 2 < (3 * RES) ** 2


def simplificar(G: nx.MultiGraph) -> nx.MultiGraph:
    """Une los tramos que pasan por nodos de grado 2 y elimina bucles mínimos."""
    G.remove_edges_from([(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                         if u == v and d["geom"].length < 4 * RES])
    for n in list(G.nodes):
        if G.degree(n) != 2 or n not in G:
            continue
        ed = list(G.edges(n, keys=True, data=True))
        if len(ed) != 2:
            continue
        (_, a, k1, d1), (_, b, k2, d2) = ed
        if a == n or b == n:
            continue
        g1 = list(d1["geom"].coords)
        g2 = list(d2["geom"].coords)
        p = (G.nodes[n]["x"], G.nodes[n]["y"])
        # orientar: a → n → b
        if _d2(g1[0], p) < _d2(g1[-1], p):
            g1 = g1[::-1]
        if _d2(g2[-1], p) < _d2(g2[0], p):
            g2 = g2[::-1]
        G.remove_node(n)
        G.add_edge(a, b, geom=LineString(g1 + g2[1:]))
    for u, v, k, d in G.edges(keys=True, data=True):
        d["geom"] = d["geom"].simplify(2.0)
    return G


def _d2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def podar(G: nx.MultiGraph) -> nx.MultiGraph:
    for _ in range(3):
        quitar = [n for n in G.nodes if G.degree(n) == 1 and
                  next(iter(G.edges(n, data=True)))[2]["geom"].length < RAMA_MINIMA]
        G.remove_nodes_from(quitar)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    mayor = max(nx.connected_components(G), key=len)
    return G.subgraph(mayor).copy()


def atributos(G, lineas, verde, bici):
    edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)]
    tramos = gpd.GeoDataFrame({"i": range(len(edges))},
                              geometry=[d["geom"] for *_, d in edges], crs=25830)
    tramos["long"] = tramos.length

    # Verde: proporción de puntos del tramo (cada 8 m) dentro de una zona ajardinada
    # o a menos de 4 m de una senda o camino
    muestras = []
    for i, g in zip(tramos["i"], tramos.geometry):
        n = max(2, int(g.length // 8) + 1)
        for t in np.linspace(0, 1, n):
            muestras.append((i, g.interpolate(t, normalized=True)))
    pts = gpd.GeoDataFrame({"i": [m[0] for m in muestras]}, geometry=[m[1] for m in muestras], crs=25830)
    pts["k"] = range(len(pts))
    en_jardin = gpd.sjoin(pts[["k", "geometry"]], verde[["geometry"]],
                          predicate="within")["k"].unique()
    en_senda = gpd.sjoin_nearest(pts[["k", "geometry"]], lineas[["geometry"]], how="inner",
                                 max_distance=3)["k"].unique()
    pts["v"] = pts["k"].isin(en_jardin) | pts["k"].isin(en_senda)
    tramos["verde"] = tramos["i"].map(pts.groupby("i")["v"].mean()).fillna(0).clip(0, 1)

    # Bici: tipo de vía ciclista más cercana a menos de RADIO_BICI
    tipos = {"VÍA EXCLUSIVA BICI": 3, "ANILLO VERDE CICLISTA": 3, "VÍA PREFERENTE BICI": 2,
             "VÍA USO COMPARTIDO": 1}
    b = bici.assign(t=bici["d_COD_TIPO"].map(tipos).fillna(1))[["t", "geometry"]]
    medio = tramos.copy()
    medio["geometry"] = tramos.geometry.interpolate(0.5, normalized=True)
    j = gpd.sjoin_nearest(medio[["i", "geometry"]], b, how="left", max_distance=RADIO_BICI)
    tramos["bici"] = tramos["i"].map(j.groupby("i")["t"].max()).fillna(0).astype(int)

    # Tráfico: IMD del aforo más cercano al tramo
    import geopandas as _g
    carpeta = CARPETA_RAW / "factores" / "imd_tramos_2020"
    p = _g.read_file(next(carpeta.glob("*.shp")))
    p = p[p.geometry.notna()].reset_index(drop=True).set_crs(4326, allow_override=True).to_crs(25830)
    p["imd"] = pd.read_excel(next(carpeta.glob("*.xlsx")))["IMD"].to_numpy()
    # Cada aforo se asigna a la calle (subvial) en la que está, y la calle entera
    # hereda la mediana de sus aforos: una avenida con tráfico lo tiene en todo su recorrido.
    sub = gpd.read_file(R / "subviales" / "SUBVIALES.shp").to_crs(25830)[["SVIA_ID", "geometry"]]
    pa = gpd.sjoin_nearest(p[["imd", "geometry"]], sub, how="inner", max_distance=RADIO_AFORO)
    imd_calle = pa.groupby("SVIA_ID")["imd"].median()
    medio_t = tramos[["i"]].copy()
    medio_t["geometry"] = tramos.geometry.interpolate(0.5, normalized=True)
    medio_t = gpd.GeoDataFrame(medio_t, geometry="geometry", crs=25830)
    jc = gpd.sjoin(medio_t, sub, predicate="within", how="left").drop_duplicates("i")
    tramos["imd"] = tramos["i"].map(jc.set_index("i")["SVIA_ID"].map(imd_calle)).fillna(0)
    # Y junto al propio aforo, su valor si es mayor
    j = gpd.sjoin_nearest(tramos[["i", "geometry"]], p[["imd", "geometry"]], how="left",
                          max_distance=RADIO_AFORO)
    cerca = tramos["i"].map(j.groupby("i")["imd"].max()).fillna(0)
    tramos["imd"] = np.maximum(tramos["imd"], cerca).astype(int)
    tramos.loc[tramos["verde"] > 0.6, "imd"] = 0   # por dentro de un parque no hay tráfico

    # Atropellos de peatones cerca del tramo
    from camino_escolar import atropellos
    acc = atropellos()
    acc = gpd.GeoDataFrame(acc, geometry=gpd.points_from_xy(acc["x"], acc["y"]), crs=25830)
    j = gpd.sjoin(acc[["menor", "geometry"]], tramos.assign(geometry=tramos.buffer(RADIO_ATROPELLO))[["i", "geometry"]])
    tramos["atrop"] = tramos["i"].map(j.groupby("i").size()).fillna(0).astype(int)

    # Nombre de cada tramo, para las indicaciones paso a paso:
    # la calle del callejero en la que está o, fuera de las calles, el parque más cercano
    # El callejero apenas trae nombres en sus piezas de calle: cada pieza toma el nombre
    # más repetido entre los portales (direcciones oficiales) que dan a ella
    sub = gpd.read_file(R / "subviales" / "SUBVIALES.shp").to_crs(25830)[["SVIA_ID", "geometry"]]
    por = portales()
    jp_ = gpd.sjoin_nearest(gpd.GeoDataFrame(por[["calle"]], geometry=gpd.points_from_xy(por["x"], por["y"]),
                                             crs=25830), sub, how="inner", max_distance=25)
    nombre_pieza = jp_.groupby("SVIA_ID")["calle"].agg(lambda s: s.value_counts().index[0])
    sub["nombre"] = sub["SVIA_ID"].map(nombre_pieza).fillna("")
    jn = gpd.sjoin(medio_t, sub[["nombre", "geometry"]], predicate="within", how="left").drop_duplicates("i")
    tramos["nombre"] = tramos["i"].map(jn.set_index("i")["nombre"]).fillna("")
    parques = pd.read_csv(R / "parques.csv", sep=";", encoding="latin1", dtype=str)
    parques = parques.assign(x=pd.to_numeric(parques["COORDENADA-X"], errors="coerce"),
                             y=pd.to_numeric(parques["COORDENADA-Y"], errors="coerce")).dropna(subset=["x", "y"])
    parques = gpd.GeoDataFrame(parques[["NOMBRE"]], geometry=gpd.points_from_xy(parques["x"], parques["y"]),
                               crs=25830)
    sin_nombre = tramos["nombre"].eq("")
    jp = gpd.sjoin_nearest(medio_t[sin_nombre.values], parques, how="left", max_distance=600)
    jp = jp.drop_duplicates("i").set_index("i")["NOMBRE"]
    tramos.loc[sin_nombre, "nombre"] = tramos.loc[sin_nombre, "i"].map(jp).fillna("")
    tramos.loc[sin_nombre & tramos["nombre"].ne(""), "nombre"] = (
        "senda del " + tramos.loc[sin_nombre & tramos["nombre"].ne(""), "nombre"].str.strip())
    tramos.loc[tramos["nombre"].eq(""), "nombre"] = "camino peatonal"
    return edges, tramos


def exportar(G, edges, tramos):
    nodos = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodos)}
    xs = [round(G.nodes[n]["x"]) for n in nodos]
    ys = [round(G.nodes[n]["y"]) for n in nodos]
    nombres = sorted(tramos["nombre"].unique())
    idx_nombre = {n: i for i, n in enumerate(nombres)}
    tr = []
    for (u, v, k, d), (_, t) in zip(edges, tramos.iterrows()):
        coords = np.asarray(d["geom"].coords).round().astype(int)
        # La geometría empieza en el nodo a: si viene al revés, se invierte
        if (coords[0][0] - G.nodes[u]["x"]) ** 2 + (coords[0][1] - G.nodes[u]["y"]) ** 2 > \
           (coords[-1][0] - G.nodes[u]["x"]) ** 2 + (coords[-1][1] - G.nodes[u]["y"]) ** 2:
            coords = coords[::-1]
        # Geometría intermedia en desplazamientos relativos (más compacta)
        inter = coords[1:-1]
        delta = np.diff(np.vstack([coords[:1], inter]), axis=0).ravel().tolist() if len(inter) else []
        tr.append([idx[u], idx[v], round(t["long"], 1), round(float(t["verde"]), 2), int(t["bici"]),
                   int(t["imd"]), int(t["atrop"]), delta, idx_nombre[t["nombre"]]])
    # Atropellos de peatones con su posición, para marcarlos en el mapa y en el recorrido
    from camino_escolar import atropellos
    acc = atropellos()
    puntos = [[int(x), int(y), int(m), int(h), int(g)] for x, y, m, h, g in
              zip(acc["x"], acc["y"], acc["menor"], acc["horario_escolar"], acc["grave"])]
    red = {"crs": "EPSG:25830", "nodos_x": xs, "nodos_y": ys,
           "campos": ["a", "b", "long_m", "verde", "bici", "imd", "atropellos", "geom_delta", "nombre"],
           "nombres": nombres, "tramos": tr,
           "atropellos_campos": ["x", "y", "menor", "horario_escolar", "grave"], "atropellos": puntos}
    (CARPETA_SALIDA / "rutas_red.json").write_text(json.dumps(red, separators=(",", ":")),
                                                     encoding="utf-8")
    return red


def portales() -> pd.DataFrame:
    d = pd.read_csv(R / "direccionesvigentes_20261109.csv", sep=";", encoding="latin1", dtype=str)
    d = d[d["CLASE_APP"].str.strip().eq("NUMERO")]
    x = pd.to_numeric(d["UTMX_ETRS"].str.replace(",", "."), errors="coerce")
    y = pd.to_numeric(d["UTMY_ETRS"].str.replace(",", "."), errors="coerce")
    d = d.assign(x=x, y=y).dropna(subset=["x", "y"])
    d["calle"] = (d["VIA_CLASE"].str.strip().str.title() + " " +
                  d["VIA_PAR"].fillna("").str.strip().str.lower() + " " +
                  d["VIA_NOMBRE_ACENTOS"].str.strip().str.title()).str.replace(r"\s+", " ", regex=True)
    return d


def direcciones():
    d = pd.read_csv(R / "direccionesvigentes_20261109.csv", sep=";", encoding="latin1", dtype=str)
    d = d[d["CLASE_APP"].str.strip().eq("NUMERO")]
    x = pd.to_numeric(d["UTMX_ETRS"].str.replace(",", "."), errors="coerce")
    y = pd.to_numeric(d["UTMY_ETRS"].str.replace(",", "."), errors="coerce")
    d = d.assign(x=x.round(), y=y.round(), n=pd.to_numeric(d["NUMERO"], errors="coerce"))
    d = d.dropna(subset=["x", "y", "n"])
    d["calle"] = (d["VIA_CLASE"].str.strip().str.title() + " " +
                  d["VIA_PAR"].fillna("").str.strip().str.lower() + " " +
                  d["VIA_NOMBRE_ACENTOS"].str.strip().str.title()).str.replace(r"\s+", " ", regex=True)
    salida = []
    for (calle, cp), g in d.groupby(["calle", "COD_POSTAL"]):
        g = g.sort_values("n").drop_duplicates("n")
        salida.append([calle, cp, g["n"].astype(int).tolist(), g["x"].astype(int).tolist(),
                       g["y"].astype(int).tolist()])
    (CARPETA_SALIDA / "rutas_direcciones.json").write_text(
        json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return len(d), len(salida)


def main() -> int:
    print("Cargando fuentes…")
    sub, lineas, verde, bici = cargar_fuentes()
    cache = CARPETA_RAW / "rutas" / "red_cache.pkl"
    if cache.exists():
        import pickle
        G = pickle.loads(cache.read_bytes())
        print("Red de ejes leída de la caché")
    else:
        print("Construyendo los ejes de la red (esqueletización)…")
        esq, tr = esqueleto(sub, lineas, bici)
        G = simplificar(podar(a_grafo(esq, tr)))
        import pickle
        cache.write_bytes(pickle.dumps(G))
    print(f"  {G.number_of_nodes():,} nodos y {G.number_of_edges():,} tramos")
    print("Calculando atributos de cada tramo…")
    edges, tramos = atributos(G, lineas, verde, bici)
    exportar(G, edges, tramos)
    n_dir, n_calles = direcciones()
    km = tramos["long"].sum() / 1000
    texto = "\n".join([
        "RESPIRACOLE · RED PEATONAL Y CICLISTA DE MADRID CAPITAL", "=" * 55,
        f"Nodos: {G.number_of_nodes():,} · tramos: {G.number_of_edges():,} · {km:,.0f} km",
        f"  por zona verde o senda:      {(tramos['long'] * tramos['verde']).sum() / 1000:,.0f} km",
        f"  con vía ciclista:            {tramos.loc[tramos['bici'] > 0, 'long'].sum() / 1000:,.0f} km",
        f"  junto a aforo > 10.000 veh.: {tramos.loc[tramos['imd'] > 10000, 'long'].sum() / 1000:,.0f} km",
        f"  con atropellos cerca:        {tramos.loc[tramos['atrop'] > 0, 'long'].sum() / 1000:,.0f} km",
        f"Direcciones (portales): {n_dir:,} en {n_calles:,} calles y códigos postales",
    ]) + "\n"
    (CARPETA_SALIDA / "rutas_resumen.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    return 0


def parques(area_min=1500, tolerancia=4):
    """Zonas ajardinadas como superficies, para pintarlas de fondo en el mapa del camino al cole.

    Salida: datos/rutas_parques.json -> lista de anillos exteriores [x0, y0, dx, dy, dx, dy, ...] (UTM, metros)
    """
    verde = capa_carto("11_zona_ajardinada_p")
    verde = verde[verde.geometry.notna() & verde.is_valid]
    verde = verde[verde.area >= area_min].copy()
    verde["geometry"] = verde.simplify(tolerancia)
    anillos = []
    for geom in verde.geometry:
        if geom.is_empty:
            continue
        for pol in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            c = np.round(np.array(pol.exterior.coords)).astype(int)
            if len(c) >= 4:
                anillos.append(np.r_[c[:1], np.diff(c, axis=0)].ravel().tolist())
    (CARPETA_SALIDA / "rutas_parques.json").write_text(json.dumps(anillos, separators=(",", ":")), encoding="utf-8")
    print(f"Parques: {len(anillos):,} superficies")
    return anillos


if __name__ == "__main__":
    sys.exit(main())
