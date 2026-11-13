"""Entorno de cada centro para las familias: vivienda, sanidad, bibliotecas, deporte y áreas infantiles.

Fuentes (datos abiertos)
  Comunidad de Madrid
    - Alquiler €/m² mensual de vivienda habitual por código postal (Madrid capital) y por municipio (>20.000 hab.)
    - Valor tasado de la vivienda libre €/m² por municipio (>25.000 hab.)
    - Centros, servicios y establecimientos sanitarios (registro, con coordenadas)
    - Oficinas de farmacia (con coordenadas)
    - Directorio de bibliotecas (para contar las de cada municipio)
  Ayuntamiento de Madrid
    - Centros de atención médica (hospitales y centros de salud con nombre), bibliotecas,
      centros deportivos municipales, instalaciones deportivas básicas, piscinas y áreas infantiles

Salida: datos/entorno.json
  puntos:   {categoría: [[x, y, nombre], ...]}  (UTM 30N, metros)
  vivienda: {cp: {cp: [alquiler €/m², año]}, muni: {municipio: {alquiler: [€/m², año], compra: [€/m², año]}}}
  cp_centro: {código de centro: código postal}
  bibliotecas_muni: {municipio: número de bibliotecas públicas}
"""
import json
import re
from pathlib import Path

import pandas as pd

from preparar_datos import normalizar_municipio

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw" / "entorno"
DATOS = BASE / "datos"


def clave_muni(s):
    s = re.sub(r"^(.*) \((El|La|Los|Las)\)$", r"\2 \1", str(s).strip())
    return normalizar_municipio(s)


def leer_cm(nombre):
    d = pd.read_csv(RAW / f"{nombre}.csv", sep=";", encoding="utf-8-sig", dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d = d[d.Valor.notna() & (d.Valor != "-")].copy()
    d["Valor"] = d.Valor.str.replace(",", ".").astype(float)
    d["Año"] = d["Año"].astype(int)
    return d


def ultimo(d, clave):
    d = d.sort_values("Año").groupby(clave).tail(1)
    return {k: [round(v, 1), int(a)] for k, v, a in zip(d[clave], d.Valor, d["Año"])}


def vivienda(nombres_app):
    app = {clave_muni(m): m for m in nombres_app}
    alq = leer_cm("alquiler_metro_cuadrado_mensual_viviendas_arrendadas_valor_catastral_municipios_mas_20000")
    alq = alq[(alq["Tipo territorio"] == "Municipios") & (alq["Tipo de vivienda"] == "Viviendas habituales")]
    alq["m"] = alq.Territorio.map(lambda t: app.get(clave_muni(t)))
    compra = leer_cm("1004050")
    compra = compra[compra["Tipo territorio"] == "Municipios"]
    compra["m"] = compra.Territorio.map(lambda t: app.get(clave_muni(t)))
    muni = {}
    for m, v in ultimo(alq[alq.m.notna()], "m").items():
        muni.setdefault(m, {})["alquiler"] = v
    for m, v in ultimo(compra[compra.m.notna()], "m").items():
        muni.setdefault(m, {})["compra"] = v
    cp = leer_cm("1934259")
    cp["cp"] = cp.Concepto.str.extract(r"código postal (\d{5})")[0]
    cp["zona"] = cp.Concepto.str.extract(r"código postal \d{5}-(.*)$")[0]
    cps = ultimo(cp[cp.cp.notna()], "cp")
    zonas = dict(zip(cp.cp, cp.zona))
    return {"cp": {k: v + [zonas.get(k, "")] for k, v in cps.items()}, "muni": muni}


def xy(df, cx, cy):
    x = pd.to_numeric(df[cx].astype(str).str.replace(",", "."), errors="coerce")
    y = pd.to_numeric(df[cy].astype(str).str.replace(",", "."), errors="coerce")
    return x, y


def ayto(fichero, filtro=None):
    d = pd.read_csv(RAW / f"ayto_{fichero}.csv", sep=";", encoding="latin-1", dtype=str)
    if filtro is not None:
        d = d[filtro(d)]
    x, y = xy(d, "COORDENADA-X", "COORDENADA-Y")
    return [[round(a), round(b), str(n).strip()] for a, b, n in zip(x, y, d.NOMBRE) if pd.notna(a) and a > 0]


def titulo(s):
    s = str(s or "").strip()
    return s[:1].upper() + s[1:].lower() if s.isupper() else s


def puntos():
    out = {}
    # Sanidad: con nombre en Madrid capital (Ayuntamiento); registro de la Comunidad para el resto
    med = pd.read_csv(RAW / "ayto_212769-0-atencion-medica.csv", sep=";", encoding="latin-1", dtype=str)
    x, y = xy(med, "COORDENADA-X", "COORDENADA-Y")
    med["x"], med["y"] = x, y
    hosp_cap = med[med.TIPO.str.contains("Hospitales", na=False)]
    cs_cap = med[med.TIPO.str.endswith("/CentrosSalud", na=False)]
    reg = pd.read_csv(RAW / "centros_servicios_establecimientos_sanitarios.csv", sep=";", encoding="utf-8-sig", dtype=str)
    reg = reg.drop_duplicates("centro_nro_registro")
    reg = reg[reg.municipio_nombre.map(clave_muni) != "MADRID"]
    rx, ry = xy(reg, "localizacion_coordenada_x", "localizacion_coordenada_y")
    reg["x"], reg["y"] = rx, ry
    publico = reg.dependecia_funcional.str.contains("Institutos de Salud|Públicos|Municipio", na=False)
    dire = lambda r: f"{titulo(r.direccion_vial_tipo)} {r.direccion_vial_nombre or ''} {r.direccion_vial_nro or ''}".strip()
    h = reg[reg.centro_tipo.isin(["Hospital general", "Hospital especializado"])]
    c = reg[reg.centro_tipo.isin(["Centro de salud", "Consultorio de atención primaria"])]
    out["hosp"] = [[round(r.x), round(r.y), str(r.NOMBRE).strip()] for r in hosp_cap.itertuples() if r.x > 0] + \
                  [[round(r.x), round(r.y), f"{r.centro_tipo}{' público' if p else ' privado'} · {dire(r)}, {r.municipio_nombre}"]
                   for r, p in zip(h.itertuples(), publico[h.index]) if r.x > 0]
    out["cs"] = [[round(r.x), round(r.y), str(r.NOMBRE).strip()] for r in cs_cap.itertuples() if r.x > 0] + \
                [[round(r.x), round(r.y), f"{r.centro_tipo} · {dire(r)}, {r.municipio_nombre}"]
                 for r in c[publico[c.index]].itertuples() if r.x > 0]
    f = pd.read_csv(RAW / "oficinas_farmacia.csv", sep=";", encoding="utf-8-sig", dtype=str)
    fx, fy = xy(f, "LOCALIZACION_COORDENADA_X", "LOCALIZACION_COORDENADA_Y")
    out["farm"] = [[round(a), round(b), f"Farmacia · {titulo(t)} {n or ''} {k or ''}".strip()]
                   for a, b, t, n, k in zip(fx, fy, f.DIRECCION_VIAL_TIPO, f.DIRECCION_VIAL_NOMBRE, f.DIRECCION_VIAL_NRO) if pd.notna(a) and a > 0]
    out["bib"] = ayto("201747-0-bibliobuses-bibliotecas", lambda d: ~d.NOMBRE.str.contains("Bibliob", na=False))
    out["dep"] = ayto("200186-0-polideportivos")
    out["depb"] = ayto("200215-0-instalaciones-deportivas")
    out["pisc"] = ayto("210227-0-piscinas-publicas")
    inf = pd.read_csv(RAW / "ayto_200652-0-areas-infantiles.csv", sep=";", encoding="latin-1", dtype=str)
    ix, iy = xy(inf, "UTMX", "UTMY")
    out["inf"] = [[round(a), round(b), titulo(d)] for a, b, d in zip(ix, iy, inf.DIRECCION) if pd.notna(a) and a > 0]
    return out


def bibliotecas_muni(nombres_app):
    app = {clave_muni(m): m for m in nombres_app}
    b = pd.read_csv(RAW / "spacm_listado_bibliotecas_servicios_lectura.csv", sep=";", encoding="latin-1", dtype=str)
    b = b[b.tipo_biblioteca.str.contains("Pública", na=False) & ~b.tipo_biblioteca.str.contains("móvil|Bibliob", case=False, na=False)]
    b["m"] = b.localidad.map(lambda t: app.get(clave_muni(t)))
    return b.groupby("m").size().astype(int).to_dict()


if __name__ == "__main__":
    cen = pd.read_csv(DATOS / "centros.csv", dtype={"codigo": str, "codigo_postal": str})
    nombres = cen.municipio.unique()
    salida = {"puntos": puntos(), "vivienda": vivienda(nombres),
              "cp_centro": {c: str(p).zfill(5) for c, p in zip(cen.codigo, cen.codigo_postal) if pd.notna(p)},
              "bibliotecas_muni": bibliotecas_muni(nombres)}
    (DATOS / "entorno.json").write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print({k: len(v) for k, v in salida["puntos"].items()})
    print("alquiler CP:", len(salida["vivienda"]["cp"]), "| municipios con vivienda:", len(salida["vivienda"]["muni"]),
          "| con compra:", sum("compra" in v for v in salida["vivienda"]["muni"].values()))
    print("Madrid:", salida["vivienda"]["muni"].get("Madrid"), "| 28026:", salida["vivienda"]["cp"].get("28026"))
    print("bibliotecas por municipio:", len(salida["bibliotecas_muni"]), list(salida["bibliotecas_muni"].items())[:5])
