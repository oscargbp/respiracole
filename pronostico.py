"""Previsión a 5 días: tiempo (AEMET, por municipio) y calidad del aire (Copernicus CAMS, por estación).

Fuentes
  AEMET  https://www.aemet.es/xml/municipios/localidad_28XXX.xml   predicción oficial por municipio, sin clave
  CAMS   https://air-quality-api.open-meteo.com/v1/air-quality    previsión europea de Copernicus (CAMS Europe, ~10 km)

Salida: datos/pronostico.json
  dias:        fechas (hoy y los 4 siguientes)
  municipios:  {municipio: [[cielo_cod, cielo_txt, tmax, tmin, prob_lluvia_mañana, prob_lluvia_tarde,
                             viento_dir, viento_kmh, racha_kmh, uv, humedad_max] por día]}
  estaciones:  {índice de estación: [[NO2 máx 9-16 h, O3 máx 9-16 h, PM10 media, PM2,5 media, polvo máx] por día]}
Se ejecuta cada tarde en producción; la demo usa la última ejecución.
"""
import json
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
RAW = BASE / "raw"
DIAS = 5
UA = {"User-Agent": "Mozilla/5.0 (RespiraCole DATAMAD 2026)"}


def municipios():
    """Municipio (como aparece en la app) -> código INE de 3 cifras.

    La tabla se guarda en datos/municipios_ine.csv; si no existe, se construye con el directorio de centros.
    """
    tabla = DATOS / "municipios_ine.csv"
    if tabla.exists():
        t = pd.read_csv(tabla, dtype=str)
        return dict(zip(t.municipio, t.codigo_ine))
    raw = pd.read_csv(RAW / "centros_educativos.csv", sep=None, engine="python", dtype=str)
    raw.columns = [c.lstrip("\ufeff") for c in raw.columns]
    cen = pd.read_csv(DATOS / "centros.csv", dtype={"codigo": str})
    m = cen.merge(raw[["CODIGO", "CDMUNI"]], left_on="codigo", right_on="CODIGO")
    d = m.groupby("municipio").CDMUNI.agg(lambda s: s.mode().iloc[0]).to_dict()
    pd.DataFrame({"municipio": list(d), "codigo_ine": list(d.values())}).to_csv(tabla, index=False)
    return d


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def aemet(cod, dias):
    url = f"https://www.aemet.es/xml/municipios/localidad_28{cod}.xml"
    for intento in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException:
            time.sleep(2 + 3 * intento)
    else:
        return None
    raiz = ET.fromstring(r.content)
    out = {}
    for d in raiz.find("prediccion").findall("dia"):
        f = d.get("fecha")
        if f not in dias:
            continue
        # los primeros días vienen por periodos; los últimos, un solo dato sin periodo
        per = lambda tag, p: next((e for e in d.findall(tag) if e.get("periodo", "00-24") == p), None)

        def primero(tag, periodos):
            for p in periodos:
                e = per(tag, p)
                if e is not None and ((e.text or "").strip() or len(e) and (e[0].text or "").strip()):
                    return e
            return None
        cielo = primero("estado_cielo", ["06-12", "00-12", "00-24", "12-18", "12-24"])
        llm = primero("prob_precipitacion", ["06-12", "00-12", "00-24"])
        llt = primero("prob_precipitacion", ["12-18", "12-24", "00-24"])
        vie = primero("viento", ["06-12", "00-12", "00-24", "12-18", "12-24"])
        rac = primero("racha_max", ["06-12", "00-12", "00-24", "12-18", "12-24"])
        t, h = d.find("temperatura"), d.find("humedad_relativa")
        uv = d.find("uv_max")
        out[f] = [
            (cielo.text or "").strip().rstrip("n") if cielo is not None else None,
            cielo.get("descripcion") if cielo is not None else None,
            num(t.findtext("maxima")), num(t.findtext("minima")),
            num(llm.text) if llm is not None else None, num(llt.text) if llt is not None else None,
            vie.findtext("direccion") if vie is not None else None, num(vie.findtext("velocidad")) if vie is not None else None,
            num(rac.text) if rac is not None else None,
            num(uv.text) if uv is not None else None,
            num(h.findtext("maxima")) if h is not None else None,
        ]
    return [out.get(f) for f in dias], raiz.findtext("elaborado")


def cams(estaciones, dias):
    lat = ",".join(str(x) for x in estaciones.lat)
    lon = ",".join(str(x) for x in estaciones.lon)
    for intento in range(5):
        try:
            r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=120, params={
                "latitude": lat, "longitude": lon, "domains": "cams_europe", "timezone": "Europe/Madrid",
                "hourly": "nitrogen_dioxide,ozone,pm10,pm2_5,dust", "start_date": dias[0], "end_date": dias[-1]})
            r.raise_for_status()
            break
        except requests.RequestException:
            if intento == 4:
                raise
            time.sleep(5 * (intento + 1))
    res = r.json()
    res = res if isinstance(res, list) else [res]
    out = {}
    for i, x in enumerate(res):
        h = pd.DataFrame(x["hourly"])
        h["t"] = pd.to_datetime(h.time)
        h["f"], h["hora"] = h.t.dt.strftime("%Y-%m-%d"), h.t.dt.hour
        filas = []
        for f in dias:
            g, cole = h[h.f == f], h[(h.f == f) & h.hora.between(9, 16)]
            r1 = lambda v: None if v is None or pd.isna(v) else round(float(v), 1)
            filas.append([r1(cole.nitrogen_dioxide.max()), r1(cole.ozone.max()), r1(g.pm10.mean()),
                          r1(g.pm2_5.mean()), r1(g.dust.max())])
        out[i] = filas
    return out


if __name__ == "__main__":
    hoy = date.today()
    dias = [(hoy + timedelta(d)).isoformat() for d in range(DIAS)]
    muni = municipios()
    res, elaborado = {}, None
    for nombre, cod in sorted(muni.items()):
        x = aemet(cod, dias)
        if x:
            res[nombre], elaborado = x[0], x[1] or elaborado
        time.sleep(0.15)
    est = pd.read_csv(DATOS / "estaciones.csv", dtype={"codigo": str})
    aire = cams(est, dias)
    salida = {"generado": datetime.now().strftime("%Y-%m-%dT%H:%M"), "aemet_elaborado": elaborado, "dias": dias,
              "municipios": res, "estaciones": aire,
              "fuentes": {"tiempo": "AEMET, predicción por municipios", "aire": "Copernicus Atmosphere Monitoring Service (CAMS Europe), vía Open-Meteo"}}
    texto = json.dumps(salida, ensure_ascii=False, separators=(",", ":"))
    (DATOS / "pronostico.json").write_text(texto, encoding="utf-8")
    for web in ("demo", "docs"):  # la demo y la web publicada leen este fichero
        if (BASE / web).is_dir():
            (BASE / web / "demo_pronostico.js").write_text(f"window.RC_PRON = {texto};", encoding="utf-8")
    print("municipios", len(res), "de", len(muni), "| estaciones", len(aire), "| días", dias)
    print("Madrid:", res.get("Madrid"))
    print("Estación 0:", aire[0])
