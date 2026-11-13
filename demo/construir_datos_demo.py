"""Genera demo_datos.js con los datos que usa la demo de RespiraCole."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "datos"
HORARIO = Path(sys.argv[1])  # parquet con el horario combinado (ver preparar_datos.leer_historico)

# Bandas del Índice de Calidad del Aire (µg/m³): límite superior de cada nivel 1-5;
# por encima del último, nivel 6. NO2 y O3 con valores horarios, PM con medias de 24 h.
BANDAS = {
    "NO2": [40, 90, 120, 230, 340],
    "O3": [50, 100, 130, 240, 380],
    "PM10": [20, 40, 50, 100, 150],
    "PM25": [10, 20, 25, 50, 75],
}


def nivel(cont, v):
    if v is None or np.isnan(v):
        return None
    return int(np.searchsorted(BANDAS[cont], v, side="left") + 1)


h = pd.read_parquet(HORARIO)
h["fecha"] = pd.to_datetime(h["fecha"])

# --- Día normal: el día lectivo más reciente de 2025 con todas las estaciones en nivel <= 2
navidad = ((h.fecha.dt.month == 12) & (h.fecha.dt.day >= 20)) | ((h.fecha.dt.month == 1) & (h.fecha.dt.day <= 7))
# de lunes a jueves, para que "mañana" también sea lectivo
lect = h[(h.fecha.dt.year == 2025) & (h.fecha.dt.weekday < 4) & (~h.fecha.dt.month.isin([6, 7, 8]))
         & ~navidad]


def valores_dia(df):
    cole = df[df.hora.between(9, 16)]
    out = {}
    for cont in ("NO2", "O3"):
        s = cole[cole.contaminante == cont].groupby("estacion_codigo").valor.max()
        out[cont] = s
    for cont in ("PM10", "PM25"):
        s = df[df.contaminante == cont].groupby("estacion_codigo").valor.mean()
        out[cont] = s
    return out


candidatos = []
for fecha, g in lect.groupby("fecha"):
    v = valores_dia(g)
    peor = max((max((nivel(c, x) or 0) for x in s.values) if len(s) else 0) for c, s in v.items())
    candidatos.append((fecha, peor))
normales = [f for f, p in candidatos if p <= 2]
dia_normal = max(normales) if normales else max(candidatos, key=lambda t: -t[1])[0]

DIAS = [
    {"id": "normal", "fecha": dia_normal.strftime("%Y-%m-%d"), "nombre": "Un día normal"},
    {"id": "no2", "fecha": "2025-01-16", "nombre": "Episodio de NO₂"},
    {"id": "calima", "fecha": "2022-03-15", "nombre": "Calima"},
]

estaciones = pd.read_csv(DATOS / "estaciones.csv", dtype={"codigo": str})
cod_est = estaciones["codigo"].tolist()
idx_est = {c: i for i, c in enumerate(cod_est)}

met = pd.read_csv(DATOS / "metricas_diarias.csv", dtype={"estacion_codigo": str})
pred = pd.read_csv(DATOS / "prediccion_diaria.csv", dtype={"estacion_codigo": str})
pred["fecha"] = pred["fecha"].str[:10]
pred["fecha_obj"] = pred["fecha_obj"].str[:10]


def por_estacion(df, col):
    out = {}
    for cont, g in df.groupby("contaminante"):
        out[cont] = {idx_est[e]: round(float(x), 1) for e, x in zip(g.estacion_codigo, g[col])
                     if e in idx_est and pd.notna(x)}
    return out


dias_out = []
for d in DIAS:
    g = h[h.fecha == pd.Timestamp(d["fecha"])]
    v = None
    # horario completo por estación para el gráfico del día (NO2, O3, PM10)
    serie = {}
    for cont in ("NO2", "O3", "PM10", "PM25"):
        p = g[g.contaminante == cont].pivot_table(index="estacion_codigo", columns="hora", values="valor")
        serie[cont] = {idx_est[e]: [None if np.isnan(x) else round(float(x)) for x in p.loc[e].reindex(range(24)).values]
                       for e in p.index if e in idx_est}
    # Mismas métricas que el modelo de predicción (predecir.py)
    valores = por_estacion(met[met.fecha == d["fecha"]], "valor")
    pr = pred[pred.fecha == d["fecha"]]
    manana = {"fecha": pr.fecha_obj.iloc[0] if len(pr) else None,
              "pred": por_estacion(pr, "pred"), "real": por_estacion(pr, "y")}
    dias_out.append({**d, "valores": valores, "horas": serie, "manana": manana})

# --- Centros
ctx = pd.read_csv(DATOS / "centros_contexto.csv", dtype={"codigo": str})
eq = pd.read_csv(DATOS / "centros_equidad.csv", dtype={"codigo": str})
cols_eq = ["codigo", "lat", "lon", "fiabilidad", "quintil_renta", "area_territorial"]
for c in ("NO2", "PM10", "PM25", "O3"):
    cols_eq += [f"{c}_estacion", f"{c}_distancia_km", f"{c}_dias_supera_oms"]
df = ctx.merge(eq[cols_eq], on="codigo", how="left")
ep = pd.read_csv(DATOS / "episodios_centros.csv", dtype={"codigo": str})
CURSOS = ["2021-22", "2022-23", "2023-24", "2024-25"]
ep_map = {}
for cod, g in ep.groupby("codigo"):
    g = g.set_index("curso")
    ep_map[cod] = [[int(g.loc[c, "dias_ambar"]), int(g.loc[c, "dias_rojo"])] if c in g.index else None
                   for c in CURSOS]
camino = json.loads((DATOS / "camino_escolar.json").read_text(encoding="utf-8"))


def r(x, n=1):
    return None if pd.isna(x) else round(float(x), n)


centros = []
for _, f in df.iterrows():
    est = {}
    for c in ("NO2", "PM10", "PM25", "O3"):
        e = f.get(f"{c}_estacion")
        if pd.notna(e):
            est[c] = [idx_est[str(int(e))], r(f[f"{c}_distancia_km"], 2)]
    centros.append([
        f.codigo, f.nombre.title() if f.nombre.isupper() else f.nombre, f.categoria, f.titularidad,
        f.municipio, f.distrito if isinstance(f.distrito, str) else "",
        r(f.lat, 5), r(f.lon, 5),
        r(f.NO2_media), r(f.PM10_media), r(f.PM25_media),
        r(f.NO2_dias_supera_oms, 0), r(f.PM10_dias_supera_oms, 0), r(f.O3_dias_supera_oms, 0),
        f.fiabilidad, r(f.indice_renta, 0), None if pd.isna(f.quintil_renta) else int(f.quintil_renta),
        r(f.densidad_municipio_hab_km2, 0), r(f.vehiculos_por_1000_hab, 0),
        r(f.pct_vehiculos_sin_distintivo), r(f.densidad_distrito_hab_ha, 0),
        r(f.zonas_verdes_ha_por_10000_hab, 1),
        r(f.trafico_imd_max_200m, 0), f.trafico_calle if isinstance(f.trafico_calle, str) else None,
        r(f.trafico_distancia_m, 0), est, ep_map.get(f.codigo),
        (lambda k: [k["n"], k["menores"], k["horario"], k["graves"]] if k else None)(camino.get(f.codigo)),
    ])
CAMPOS = ["codigo", "nombre", "categoria", "titularidad", "municipio", "distrito", "lat", "lon",
          "no2", "pm10", "pm25", "dias_no2", "dias_pm10", "dias_o3", "fiabilidad", "renta",
          "quintil", "dens_muni", "veh_1000", "pct_sin_dist", "dens_distrito", "verde",
          "trafico", "calle", "trafico_dist", "est", "ep", "atrop"]

# Detalle del camino escolar (puntos), en un fichero aparte que se carga al abrir la ficha
(Path(__file__).parent / "demo_camino.js").write_text(
    "window.RC_CAMINO = " + json.dumps(camino, ensure_ascii=False, separators=(",", ":")) + ";",
    encoding="utf-8")

# Evaluación del modelo (2025, días lectivos), para mostrarla en la demo
ev_txt = (DATOS / "prediccion_evaluacion.txt").read_text(encoding="utf-8")
import re
fila = lambda nombre: [float(x) for x in re.search(nombre + r"\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", ev_txt).groups()]
evaluacion = {"persistencia": fila(r"Persistencia \(= hoy\)"), "modelo": fila("Modelo RespiraCole"),
              "n": int(re.search(r"evaluados: ([\d,]+)", ev_txt).group(1).replace(",", ""))}

# --- Estaciones y perfiles
ph = pd.read_csv(DATOS / "perfil_horario.csv", dtype={"estacion_codigo": str})
pm = pd.read_csv(DATOS / "perfil_mensual.csv", dtype={"estacion_codigo": str})
res = pd.read_csv(DATOS / "estaciones_resumen.csv", dtype={"estacion_codigo": str})
est_out = []
for i, e in estaciones.iterrows():
    perf_h = {c: g.sort_values("hora").media.round(0).tolist()
              for c, g in ph[ph.estacion_codigo == e.codigo].groupby("contaminante")}
    perf_m = {c: g.sort_values("mes").media.round(0).tolist()
              for c, g in pm[pm.estacion_codigo == e.codigo].groupby("contaminante")}
    medias = {row.contaminante: row.media_periodo for row in res[res.estacion_codigo == e.codigo].itertuples()}
    nombre = e.direccion.split(" (")[0] if e.red == "ayuntamiento" else e.municipio
    est_out.append({"c": e.codigo, "n": nombre, "m": e.municipio, "red": e.red, "tipo": e.tipo_estacion,
                    "lat": r(e.lat, 5), "lon": r(e.lon, 5), "ph": perf_h, "pm": perf_m, "med": medias})

# --- Equidad
q = pd.read_csv(DATOS / "equidad_quintiles.csv").to_dict("records")
dist = pd.read_csv(DATOS / "equidad_distritos.csv")[["distrito", "indice_renta", "NO2_media",
                                                      "PM10_media", "centros"]].to_dict("records")

salida = {"campos": CAMPOS, "centros": centros, "estaciones": est_out, "dias": dias_out,
          "bandas": BANDAS, "quintiles": q, "distritos": dist, "cursos": CURSOS,
          "evaluacion": evaluacion}
texto = "window.RC_DATOS = " + json.dumps(salida, ensure_ascii=False, separators=(",", ":")) + ";"
(Path(__file__).parent / "demo_datos.js").write_text(texto, encoding="utf-8")
print("Día normal elegido:", DIAS[0]["fecha"], "| tamaño:", round(len(texto) / 1024), "KB",
      "| centros:", len(centros), "| estaciones:", len(est_out))
for d in dias_out:
    lv = {c: [nivel(c, x) for x in v.values()] for c, v in d["valores"].items()}
    print(d["fecha"], {c: max(filter(None, l), default=None) for c, l in lv.items()})

# --- Red de calles y direcciones para el camino al cole (si se ha generado con rutas_red.py)
if (DATOS / "rutas_red.json").exists():
    red = json.loads((DATOS / "rutas_red.json").read_text(encoding="utf-8"))
    red["direcciones"] = json.loads((DATOS / "rutas_direcciones.json").read_text(encoding="utf-8"))
    red.pop("nodos_z", None)
    if (DATOS / "rutas_parques.json").exists():  # rutas_red.parques()
        red["parques"] = json.loads((DATOS / "rutas_parques.json").read_text(encoding="utf-8"))
    texto_r = "window.RC_RUTAS = " + json.dumps(red, ensure_ascii=False, separators=(",", ":")) + ";"
    (Path(__file__).parent / "demo_rutas.js").write_text(texto_r, encoding="utf-8")
    print("Red de rutas:", len(red["tramos"]), "tramos ·", round(len(texto_r) / 1024 / 1024, 1), "MB")

# --- Previsión a 5 días (pronostico.py)
if (DATOS / "pronostico.json").exists():
    texto_p = "window.RC_PRON = " + (DATOS / "pronostico.json").read_text(encoding="utf-8") + ";"
    (Path(__file__).parent / "demo_pronostico.js").write_text(texto_p, encoding="utf-8")
    print("Previsión:", round(len(texto_p) / 1024), "KB")


# --- Tiempo real (tiempo_real.py): semáforo de ahora y previsión real
if (DATOS / "tiempo_real.json").exists():
    texto_v = "window.RC_VIVO = " + (DATOS / "tiempo_real.json").read_text(encoding="utf-8") + ";"
    (Path(__file__).parent / "demo_vivo.js").write_text(texto_v, encoding="utf-8")
    print("Tiempo real:", round(len(texto_v) / 1024), "KB")
