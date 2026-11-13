"""
RespiraCole · Camino escolar seguro y saludable (Madrid capital)
================================================================

Para cada centro de la capital, describe el entorno de 400 m (unos 5 minutos a
pie) que recorre el alumnado al llegar y salir:
  - Atropellos de peatones de 2023 a 2025 (Ayuntamiento de Madrid, "Accidentes de
    tráfico de la ciudad de Madrid"), con los que afectaron a menores y los que
    ocurrieron en horario de entrada o salida.
  - Puntos de aforo con más de 10.000 vehículos al día (Ayuntamiento, IMD 2020).
  - Los cruces o calles con más atropellos del entorno.

No calcula rutas: para eso haría falta la red de calles y aceras, que sería
el siguiente paso con cada ayuntamiento cliente.

Requiere haber ejecutado capas_contexto.py (descarga los aforos).
Genera datos/camino_escolar.json.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from preparar_datos import CARPETA_RAW, CARPETA_SALIDA, descargar_ayto, recursos_ayto

DS_ACCIDENTES = "300228-0-accidentes-trafico-detalle"
ANOS_ACC = ["2023", "2024", "2025"]
RADIO = 400
MENORES = {"Menor de 5 años", "De 6 a 9 años", "De 10 a 14 años", "De 15 a 17 años"}
GRAVES = {"Ingreso superior a 24 horas", "Fallecido 24 horas"}
UMBRAL_AFORO = 10_000


def descargar_accidentes() -> list:
    carpeta = CARPETA_RAW / "ayuntamiento" / "accidentes"
    carpeta.mkdir(parents=True, exist_ok=True)
    falta = [a for a in ANOS_ACC if not (carpeta / f"accidentes_{a}.csv").exists()]
    if falta:
        recs = recursos_ayto(DS_ACCIDENTES)
        for a in falta:
            url = next(r["url"] for r in recs if (r.get("format") or "").upper() == "CSV"
                       and (r.get("description") or r.get("name") or "").strip().endswith(a))
            descargar_ayto(url, carpeta / f"accidentes_{a}.csv")
            time.sleep(2)
    return [carpeta / f"accidentes_{a}.csv" for a in ANOS_ACC]


def atropellos() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str)
                    for f in descargar_accidentes()], ignore_index=True)
    df["peaton"] = df["tipo_persona"].fillna("").str.startswith("Peatón")
    df = df[df["peaton"]].copy()
    df["menor"] = df["rango_edad"].isin(MENORES)
    df["grave"] = df["lesividad"].isin(GRAVES)
    fecha = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    hora = pd.to_datetime(df["hora"], format="%H:%M:%S", errors="coerce")
    minutos = hora.dt.hour * 60 + hora.dt.minute
    lectivo = (fecha.dt.weekday < 5) & (~fecha.dt.month.isin([7, 8]))
    entrada = minutos.between(8 * 60, 9 * 60 + 30)
    salida = minutos.between(13 * 60 + 30, 17 * 60 + 30)
    df["horario_escolar"] = lectivo & (entrada | salida)
    df["x"] = pd.to_numeric(df["coordenada_x_utm"].str.replace(",", "."), errors="coerce")
    df["y"] = pd.to_numeric(df["coordenada_y_utm"].str.replace(",", "."), errors="coerce")
    df = df[df["x"].between(420_000, 460_000) & df["y"].between(4_455_000, 4_500_000)]
    # Un registro por accidente (puede haber varios peatones en el mismo)
    acc = df.groupby("num_expediente").agg(
        x=("x", "first"), y=("y", "first"), lugar=("localizacion", "first"),
        menor=("menor", "any"), grave=("grave", "any"),
        horario_escolar=("horario_escolar", "any")).reset_index()
    return acc


def aforos() -> pd.DataFrame:
    import geopandas as gpd

    carpeta = CARPETA_RAW / "factores" / "imd_tramos_2020"
    p = gpd.read_file(next(carpeta.glob("*.shp")))
    p = p[p.geometry.notna()].reset_index(drop=True).set_crs(4326, allow_override=True)
    p = p.to_crs(25830)
    x = pd.read_excel(next(carpeta.glob("*.xlsx")))
    return pd.DataFrame({"x": p.geometry.x, "y": p.geometry.y, "imd": x["IMD"].to_numpy(),
                         "calle": x["NOMBRE"].astype(str).str.strip().str.lstrip("- ")})


def limpiar_lugar(s: str) -> str:
    return " ".join(str(s).replace("CALL.", "Calle").replace("AVDA.", "Avenida")
                    .replace("PASEO", "Paseo").replace("PLAZA", "Plaza").split()).title()


def main() -> int:
    centros = pd.read_csv(CARPETA_SALIDA / "centros.csv", dtype={"codigo": str})
    cap = centros[centros["municipio"] == "Madrid"].reset_index(drop=True)
    acc, af = atropellos(), aforos()
    af = af[af["imd"] >= UMBRAL_AFORO].reset_index(drop=True)
    print(f"Atropellos 2023-2025: {len(acc):,} · aforos de más de {UMBRAL_AFORO:,} veh./día: {len(af):,}")

    ax, ay = acc["x"].to_numpy(), acc["y"].to_numpy()
    fx, fy = af["x"].to_numpy(), af["y"].to_numpy()
    salida = {}
    for _, c in cap.iterrows():
        dxa, dya = ax - c["utm_x"], ay - c["utm_y"]
        sel = np.hypot(dxa, dya) <= RADIO
        sub = acc[sel]
        dxf, dyf = fx - c["utm_x"], fy - c["utm_y"]
        self_ = np.hypot(dxf, dyf) <= RADIO
        lugares = (sub.groupby("lugar").size().sort_values(ascending=False).head(3))
        salida[c["codigo"]] = {
            "n": int(len(sub)),
            "menores": int(sub["menor"].sum()),
            "horario": int(sub["horario_escolar"].sum()),
            "graves": int(sub["grave"].sum()),
            # posición relativa al centro, en metros (x hacia el este, y hacia el norte)
            "pts": [[int(a), int(b), int(m), int(h)] for a, b, m, h in
                    zip(dxa[sel], dya[sel], sub["menor"], sub["horario_escolar"])],
            "vias": [[int(a), int(b), int(v), cal] for a, b, v, cal in
                     zip(dxf[self_], dyf[self_], af["imd"][self_], af["calle"][self_])],
            "lugares": [[limpiar_lugar(l), int(n)] for l, n in lugares.items() if n >= 2],
        }
    (CARPETA_SALIDA / "camino_escolar.json").write_text(
        json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    n = pd.Series({k: v["n"] for k, v in salida.items()})
    m = pd.Series({k: v["menores"] for k, v in salida.items()})
    print(f"Centros de la capital: {len(salida):,}")
    print(f"Atropellos a < {RADIO} m (2023-2025): mediana {n.median():.0f}, máximo {n.max()}")
    print(f"Centros con algún atropello de menores a < {RADIO} m: {(m > 0).mean():.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
