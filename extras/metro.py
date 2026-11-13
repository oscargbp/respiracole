"""Red de Metro y Metro Ligero para el camino al cole (datos del Consorcio Regional de Transportes).

Entrada (descargar_metro.py):
  raw/metro/topo/M4_*.geojson, M10_*.geojson   topología: estaciones, accesos, itinerarios y tramos
  raw/metro/gtfs_metro, raw/metro/gtfs_ligero  GTFS: frecuencias en hora punta
  datos/rutas_red.json                         red peatonal (rutas_red.py), para enganchar cada acceso a una calle

Salida: datos/metro_red.json
  grupos: estaciones o intercambiadores [nombre, x, y, accesibilidad T/P/N, [accesos]]
          acceso = [x, y, nombre, accesible 0/1, nodo de la red peatonal]
  lineas: itinerarios [linea, sublinea, color, destino, frecuencia_s, frecuencia_estimada,
                       [grupos], [segundos acumulados], geometría (dx, dy enteros)]
"""
import json
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import linemerge

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw" / "metro"
DATOS = BASE / "datos"
CRS = 25830
HORA = "08:00:00"          # frecuencia de referencia: entrada al colegio
ACCESIBLE = {"A", "S", "R"}  # ascensor, sin barreras, rampa
COLOR_L3 = "FFD000"


def norm(s):
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", s).strip()


def capa(modo, nombre):
    return gpd.read_file(RAW / "topo" / f"{modo}_{nombre}.geojson")


def frecuencias(carpeta, prefijo_modo):
    """Frecuencia (segundos) a las 8:00 de lunes a jueves para cada (línea, rama, sentido)."""
    r = lambda f: pd.read_csv(RAW / carpeta / f"{f}.txt", encoding="utf-8-sig", dtype=str)
    cal, trips, freq, rutas = r("calendar"), r("trips"), r("frequencies"), r("routes")
    serv = cal[(cal.monday == "1") & (cal.tuesday == "1")].service_id.tolist()
    t = trips[trips.service_id.isin(serv)].merge(rutas[["route_id", "route_short_name", "route_color"]])
    f = freq.merge(t, on="trip_id")
    f = f[(f.start_time <= HORA) & (f.end_time > HORA)]
    out, colores = {}, dict(zip(rutas.route_short_name, rutas.route_color))
    if f.empty:
        # sin frecuencias: se cuentan las salidas de cabecera entre las 7:30 y las 8:30
        stt = r("stop_times")
        ini = stt[stt.stop_sequence.astype(int) == stt.groupby("trip_id").stop_sequence.transform(lambda v: v.astype(int).min())]
        ini = ini.merge(t, on="trip_id")
        ini = ini[(ini.departure_time >= "07:30:00") & (ini.departure_time < "08:30:00")]
        for (ruta, sh, d), g in ini.groupby(["route_short_name", "shape_id", "direction_id"]):
            parte = sh.split("__")[1]
            rama = parte.split("_")[1] if "_" in parte else ""
            out[(ruta, rama.upper(), d)] = round(3600 / len(g))
        return out, colores
    for _, x in f.iterrows():
        # rama a partir del identificador del recorrido: 4__7_A___1__IT_1 -> A; 4__1____1__IT_1 -> ""
        parte = x.shape_id.split("__")[1]
        rama = parte.split("_")[1] if "_" in parte else ""
        out[(x.route_short_name, rama.upper(), x.direction_id)] = int(x.headway_secs)
    return out, colores


def construir():
    red = json.loads((DATOS / "rutas_red.json").read_text(encoding="utf-8"))
    arbol = cKDTree(np.c_[red["nodos_x"], red["nodos_y"]])

    fq_m, col_m = frecuencias("gtfs_metro", "4")
    fq_l, col_l = frecuencias("gtfs_ligero", "10")
    col_m.setdefault("3", COLOR_L3)

    estaciones, accesos, paradas, tramos = [], [], [], []
    for modo in ("M4", "M10"):
        e = capa(modo, "Estaciones").to_crs(CRS)
        e["modo"] = modo
        estaciones.append(e)
        a = capa(modo, "Accesos").to_crs(CRS)
        accesos.append(a)
        p = capa(modo, "ParadasPorItinerario")
        p["modo"] = modo
        paradas.append(p)
        tr = capa(modo, "Tramos").to_crs(CRS)
        tr["modo"] = modo
        tramos.append(tr)
    est = pd.concat(estaciones, ignore_index=True)
    est = est[est.IDESTACION.notna() & est.DENOMINACION.notna()]
    acc = pd.concat(accesos, ignore_index=True)
    par = pd.concat(paradas, ignore_index=True)
    tra = pd.concat(tramos, ignore_index=True)

    # --- Grupos: un intercambiador agrupa las estaciones de varias líneas
    def clave(f):
        return f"I{int(float(f.CODIGOINTERCAMBIADOR))}" if pd.notna(f.CODIGOINTERCAMBIADOR) and str(f.CODIGOINTERCAMBIADOR).strip() else f.IDESTACION
    est["grupo"] = est.apply(clave, axis=1)
    # una estación de Metro Ligero y otra de Metro con el mismo nombre a menos de 300 m también son transbordo
    claves = est.groupby("grupo")
    grupo_de = dict(zip(est.IDESTACION, est.grupo))
    orden_grupos = list(claves.groups)
    idx_grupo = {g: i for i, g in enumerate(orden_grupos)}

    grupos = []
    for g in orden_grupos:
        e = claves.get_group(g)
        nombre = e.DENOMINACION.mode().iloc[0]
        x, y = e.geometry.x.mean(), e.geometry.y.mean()
        grado = "T" if (e.GRADOACCESIBILIDAD == "T").all() else ("N" if (e.GRADOACCESIBILIDAD == "N").all() else "P")
        a = acc[acc.IDFESTACION.isin(e.IDESTACION)]
        lista = [[float(r.geometry.x), float(r.geometry.y), str(r.DENOMINACION or "").strip() or nombre.title(),
                  int(r.ACCESOMINUSVALIDOS in ACCESIBLE)] for r in a.itertuples()]
        if not lista:  # estaciones en superficie sin accesos publicados: la propia estación
            lista = [[float(x), float(y), nombre.title(), int(grado == "T")]]
        # enganche a la red peatonal (solo si hay calle a menos de 120 m)
        for l in lista:
            d, n = arbol.query([l[0], l[1]])
            l.append(int(n) if d <= 120 else -1)
            l[0], l[1] = round(l[0]), round(l[1])
        grupos.append([nombre_bonito(nombre), round(x), round(y), grado, lista])

    # --- Itinerarios
    lineas = []
    tra["orden"] = tra.NUMEROORDEN.astype(int)
    for (modo, it), p in par.groupby(["modo", "CODIGOITINERARIO"]):
        p = p.sort_values("NUMEROORDEN")
        if (p.TIPOITINERARIO.astype(str) != "1").any():
            continue
        lab = str(p.NUMEROLINEAUSUARIO.iloc[0])
        sentido = str(int(p.SENTIDO.iloc[0]))
        if modo == "M4":
            m = re.match(r"^(\d+|R)([aAbB]|-\d)?$", lab)
            num, suf = m.group(1), (m.group(2) or "")
            rama = suf.upper() if suf and suf[0] != "-" else (suf[1:] if suf else "")
            dirc = "0" if sentido == "1" else "1"
            if suf.startswith("-"):
                dirc = "0" if suf == "-1" else "1"
                rama = suf[1:]
            hw = fq_m.get((num, rama, dirc)) or fq_m.get((num, "", dirc))
            color = col_m.get(num, "888888")
            nombre_linea = f"Línea {num}" if num != "R" else "Ramal Ópera-Príncipe Pío"
        else:
            m = re.match(r"^(\d+)(-\d)?$", lab)
            num = m.group(1)
            dirc = "0" if (m.group(2) or "-1") == "-1" and sentido == "1" else "1"
            rama = (m.group(2) or "")[1:]
            hw = fq_l.get(("ML" + num, rama, dirc)) or fq_l.get(("ML" + num, rama, "0")) or fq_l.get(("ML" + num, "", "0"))
            color = col_l.get("ML" + num, "888888")
            nombre_linea = f"Metro Ligero {num}"
        estimada = hw is None
        ids = [g for g in p.IDFESTACION if g in grupo_de]
        if len(ids) < 2:
            continue
        seg = [0.0]
        for r in p.iloc[1:].itertuples():
            v = float(r.VELOCIDADTRAMOANTERIOR or 0) or 25.0
            seg.append(seg[-1] + float(r.LONGITUDTRAMOANTERIOR or 0) / (v / 3.6))
        # geometría: tramos del itinerario en orden
        t = tra[(tra.modo == modo) & (tra.CODIGOITINERARIO == it)].sort_values("orden")
        geom = linemerge(list(t.geometry)) if len(t) else None
        if geom is None or geom.geom_type != "LineString":
            pts = [est.set_index("IDESTACION").geometry[g] for g in ids]
            geom = LineString([(q.x, q.y) for q in pts])
        else:
            # orientar del primer al último punto del itinerario
            x0 = est.set_index("IDESTACION").geometry[ids[0]]
            if geom.coords[0][0] and np.hypot(geom.coords[-1][0] - x0.x, geom.coords[-1][1] - x0.y) < np.hypot(geom.coords[0][0] - x0.x, geom.coords[0][1] - x0.y):
                geom = LineString(geom.coords[::-1])
        geom = geom.simplify(6)
        c = np.round(np.array(geom.coords)).astype(int)
        delta = np.r_[c[:1], np.diff(c, axis=0)].ravel().tolist()
        destino = nombre_bonito(p.DENOMINACION.iloc[-1])
        lineas.append([nombre_linea, lab, "#" + color, destino, hw or 240, int(estimada),
                       [idx_grupo[grupo_de[g]] for g in ids], [round(s) for s in seg], delta])
    return {"grupos": grupos, "lineas": lineas,
            "fuente": "Consorcio Regional de Transportes de Madrid: topología de Metro y Metro Ligero (2026) y GTFS"}


ACENTOS = {k: v for v in """Ópera Príncipe Pío Chamartín Elíptica América Lucía Ramón García Begoña Bolívar Jesús María González
Fernández Martín Martínez Sánchez López Pérez Álvarez Méndez Menéndez José Concepción Asunción Estación República
Vía Ilustración Aviación Bambú Jardín Antón Pacífico Atocha Tetuán Álvaro Avión Paraíso Ibáñez Tomás Nicolás
Andrés Ángel Ángeles Alcalá Callao Ríos Rosales Simancas Suanzes Usera Ventas Villaverde Vinateros Óscar Hipódromo
Cortijo Ciudad Lineal Núñez Balboa Cuzco Cristóbal Colón Ríos Rosas Rubén Darío Gregorio Marañón Tribunal Velázquez
Íñigo Mártires Montaña Zarzaquemada Leganés Alcorcón Móstoles Fuenlabrada Getafe Universidad Juan Carlos Rey
Ifema Barajas Pueblo Aeropuerto Álvarez Villaamil Margarita Úrsula Lacoma Valdezarza Peñagrande Herrera Oria Sáinz Baranda
Ascensión Gómez Rodríguez Manoteras Hortaleza Concordia Pirámides Marqués Vadillo Embajadores Tirso Molina Sevilla
Gran Química Pavones Valdebernardo Vicálvaro Ensanche Vallecas Sureste Rivas Laguna Lucero Ópera Noviciado
Méndez Álvaro Arganzuela Delicias Palos Frontera Legazpi Almendrales Hospital Octubre Plaza Elíptica Abrantes Pan
Bendito San Fermín Orcasitas Puerta Arganda Valdecarros Sierra Guadalupe Belén Colonia Batán Lago Príncipe Somosaguas
Educación Física Física""".split() for k in [unicodedata.normalize("NFD", v).encode("ascii", "ignore").decode()] if k != v}
MINUS = {"de", "del", "la", "las", "los", "el", "y", "e"}


def nombre_bonito(s):
    s = str(s).strip().lower()
    pal = [w if (w in MINUS and i) else w[:1].upper() + w[1:] for i, w in enumerate(s.split())]
    s = " ".join(pal)
    s = re.sub(r"-(\w)", lambda m: "-" + m.group(1).upper(), s)
    s = re.sub(r"\b(Ii|Iii|Iv|Vi|Vii|Viii|Ix|Xi|Xii|Xiii)\b", lambda m: m.group(1).upper(), s)
    for w, t in ACENTOS.items():
        s = re.sub(rf"\b{w}\b", t, s)
    for a, b in {"Marañon": "Marañón", "Guzman": "Guzmán", "Joaquin": "Joaquín", "Julian": "Julián", "Sofia": "Sofía",
                 "Lavapies": "Lavapiés", "Hernandez": "Hernández", "Nuñez": "Núñez", "O'donnell": "O'Donnell",
                 "Catolicos": "Católicos", "Comunicacion": "Comunicación", "Bernabeu": "Bernabéu", "Belgica": "Bélgica",
                 "Ibañez": "Ibáñez", "Jardin": "Jardín", "Rio": "Río", "Mostoles": "Móstoles", "Alcorcon": "Alcorcón",
                 "Leganes": "Leganés", "Fuenlabrada": "Fuenlabrada", "Ramon": "Ramón", "Cajal": "Cajal"}.items():
        s = re.sub(rf"(?<!\w){re.escape(a)}(?!\w)", b, s)
    s = s.replace("Avda.", "Avenida").replace("Pza.", "Plaza")
    return s


if __name__ == "__main__":
    salida = construir()
    (DATOS / "metro_red.json").write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    g, l = salida["grupos"], salida["lineas"]
    print("grupos", len(g), "| itinerarios", len(l),
          "| accesos", sum(len(x[4]) for x in g), "enganchados", sum(a[4] >= 0 for x in g for a in x[4]),
          "| accesibles", sum(a[3] for x in g for a in x[4]))
    for x in l:
        print(f"{x[0]:>18} {x[1]:>4} {x[3][:24]:<24} paradas {len(x[6]):>2} tiempo {x[7][-1] / 60:5.1f} min  frec {x[4]}s{' (estimada)' if x[5] else ''}")
