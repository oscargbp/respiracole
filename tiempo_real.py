"""
RespiraCole · Tiempo real
=========================

Descarga los datos de la última hora de las redes de calidad del aire y calcula:
  - el semáforo de AHORA de cada estación (con las mismas reglas que el resto de RespiraCole),
  - la previsión para MAÑANA con el modelo de predecir.py (a partir de las 17:00, cuando ya
    ha terminado el horario escolar), y
  - lo que se previó ayer para hoy, para compararlo con lo que está pasando.

Fuentes
  Comunidad de Madrid: "Red de Calidad del Aire. Datos del día en curso" (cada hora),
                       "Datos del mes en curso" y "Datos meteorológicos del mes en curso"
  Ayuntamiento de Madrid: "Calidad del aire. Datos en tiempo real" (cada hora)

Los datos del día son provisionales (marca "T"); se validan después (marca "V").

Salida: datos/tiempo_real.json
Uso:    python tiempo_real.py            (en producción, cada hora)
        python tiempo_real.py --entrenar  (vuelve a entrenar el modelo con todo el histórico)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

import predecir
from preparar_datos import CABECERAS, CABECERAS_AYTO, CARPETA_RAW, CARPETA_SALIDA, a_numero, url_csv

VIVO = CARPETA_RAW / "vivo"
BASE = Path(__file__).resolve().parent
MODELO = CARPETA_SALIDA / "modelo_prediccion.joblib"
MAGNITUDES = {8: "NO2", 10: "PM10", 9: "PM25", 14: "O3"}
URL_AYTO = "https://ciudadesabiertas.madrid.es/dynamicAPI/API/query/{}.csv?pageSize=5000"
HORA_PREVISION = 17  # a partir de esta hora se publica la previsión de mañana
MADRID = ZoneInfo("Europe/Madrid")


# ---------------------------------------------------------------------------
# Descarga y lectura
# ---------------------------------------------------------------------------

def bajar(url: str, destino: Path, cabeceras: dict) -> Path | None:
    try:
        r = requests.get(url, headers=cabeceras, timeout=60)
        r.raise_for_status()
        destino.write_bytes(r.content)
        return destino
    except requests.RequestException as e:
        print(f"  aviso: no he podido descargar {url.split('/')[-1][:50]} ({type(e).__name__})")
        return None


def leer_horario(ruta: Path, magnitudes: dict) -> pd.DataFrame:
    """Formato de las dos redes (una fila por día, columnas h01..h24 y v01..v24) a una fila por hora."""
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    df["magnitud"] = pd.to_numeric(df["magnitud"], errors="coerce")
    df = df[df["magnitud"].isin(magnitudes)].copy()
    df["estacion_codigo"] = (df["provincia"].str.zfill(2) + df["municipio"].str.zfill(3)
                             + df["estacion"].str.zfill(3))
    df["fecha"] = pd.to_datetime(df["ano"].str.strip() + "-" + df["mes"].str.zfill(2) + "-"
                                 + df["dia"].str.zfill(2), errors="coerce")
    trozos = []
    for h in range(1, 25):
        t = df[["estacion_codigo", "magnitud", "fecha", "estacion"]].copy()
        t["hora"] = h - 1  # h01 = de 00:00 a 01:00
        t["valor"] = a_numero(df[f"h{h:02d}"])
        t["ok"] = df[f"v{h:02d}"].str.strip().isin(["V", "T"])  # validado o provisional
        trozos.append(t)
    largo = pd.concat(trozos, ignore_index=True)
    largo = largo[largo["ok"] & largo["valor"].notna() & largo["fecha"].notna()]
    largo["var"] = largo["magnitud"].map(magnitudes)
    return largo[["estacion_codigo", "estacion", "var", "fecha", "hora", "valor"]]


def descargar_todo() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Horario de contaminantes (últimos días), horario meteorológico y qué redes han respondido."""
    VIVO.mkdir(parents=True, exist_ok=True)
    redes = {"comunidad": False, "ayuntamiento": False}
    aire, meteo = [], []
    for ds, nombre in (("calidad_aire_datos_mes", "cm_mes"), ("calidad_aire_datos_dia", "cm_dia")):
        f = bajar(url_csv(ds), VIVO / f"{nombre}.csv", CABECERAS)
        if f:
            aire.append(leer_horario(f, MAGNITUDES))
            redes["comunidad"] = True
    f = bajar(url_csv("calidad_aire_datos_meteo_mes"), VIVO / "cm_meteo_mes.csv", CABECERAS)
    if f:
        meteo.append(leer_horario(f, predecir.METEO))
    for consulta in ("calair_tiemporeal_ult", "calair_tiemporeal"):
        f = bajar(URL_AYTO.format(consulta), VIVO / f"{consulta}.csv", CABECERAS_AYTO)
        if f:
            try:
                aire.append(leer_horario(f, MAGNITUDES))
                redes["ayuntamiento"] = True
            except (KeyError, ValueError) as e:
                print(f"  aviso: formato inesperado en {consulta}: {e}")
    cols = ["estacion_codigo", "var", "fecha", "hora"]
    nuevo = pd.concat(aire, ignore_index=True).drop_duplicates(cols, keep="last") if aire else pd.DataFrame()
    # Se guardan los últimos días para cubrir el cambio de mes
    cache = CARPETA_SALIDA / "ultimos_dias.parquet"
    if cache.exists():
        nuevo = pd.concat([pd.read_parquet(cache), nuevo], ignore_index=True).drop_duplicates(cols, keep="last")
    if len(nuevo):
        nuevo = nuevo[nuevo["fecha"] >= nuevo["fecha"].max() - pd.Timedelta(days=10)]
        nuevo.to_parquet(cache, index=False)
    met = pd.concat(meteo, ignore_index=True) if meteo else pd.DataFrame()
    return nuevo, met, redes


# ---------------------------------------------------------------------------
# Métricas del semáforo
# ---------------------------------------------------------------------------

def metricas_dia(h: pd.DataFrame, dia: pd.Timestamp, completo: bool) -> pd.DataFrame:
    """Métrica del semáforo por estación y contaminante para un día.

    NO2 y O3: máximo horario entre las 9:00 y las 17:00. PM10 y PM2,5: media del día.
    Si el día no ha terminado (completo=False): máximo de lo que va de horario escolar
    (o la última hora, si aún no ha empezado) y media de las últimas 24 horas.
    """
    g = h[h["fecha"] == dia]
    filas = []
    for (e, c), x in g.groupby(["estacion_codigo", "var"]):
        if c in ("NO2", "O3"):
            cole = x[x["hora"].between(9, 16)]
            if completo:
                v = cole["valor"].max() if len(cole) >= 6 else np.nan
            else:
                v = cole["valor"].max() if len(cole) else x.sort_values("hora")["valor"].iloc[-1]
        else:
            if completo:
                v = x["valor"].mean() if len(x) >= 18 else np.nan
            else:
                ayer = h[(h["fecha"] == dia - pd.Timedelta(days=1)) & (h["estacion_codigo"] == e) & (h["var"] == c)]
                ult = x["hora"].max()
                ventana = pd.concat([ayer[ayer["hora"] > ult], x])
                v = ventana["valor"].mean() if len(ventana) >= 12 else np.nan
        filas.append((e, c, v, x["valor"].mean()))
    return pd.DataFrame(filas, columns=["estacion_codigo", "contaminante", "valor", "media24"]).dropna(subset=["valor"])


def meteo_dia(met: pd.DataFrame) -> pd.DataFrame:
    """Meteorología regional por día, con las mismas variables que usa el modelo."""
    if met.empty:
        return pd.DataFrame(columns=["fecha"])
    agg = {"viento": ["mean", "max"], "temp": ["max", "min"], "humedad": ["mean"],
           "presion": ["mean"], "radiacion": ["sum"], "lluvia": ["sum"]}
    partes = []
    for var, funcs in agg.items():
        sub = met[met["var"] == var]
        regional = sub.groupby(["fecha", "estacion"])["valor"].agg(funcs).groupby("fecha").mean()
        regional.columns = [f"{var}_{f}" for f in regional.columns]
        partes.append(regional)
    out = pd.concat(partes, axis=1).sort_index()
    out["presion_cambio"] = out["presion_mean"].diff()
    return out.reset_index()


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def entrenar() -> dict:
    """Entrena el modelo definitivo con todo el histórico (2021-2025) y lo guarda."""
    print("Entrenando el modelo con todo el histórico…")
    d = pd.read_csv(CARPETA_SALIDA / "metricas_diarias.csv", dtype={"estacion_codigo": str}, parse_dates=["fecha"])
    meteo = predecir.meteo_regional()
    paquete = {}
    for cont in predecir.CONTS:
        t = predecir.preparar_tabla(d, meteo, cont)
        todo = pd.Series(True, index=t.index)
        t = predecir.con_clima(t, todo)
        train = t.dropna(subset=["y", "hoy"])
        modelo = predecir.nuevo_modelo().fit(train[predecir.VARIABLES], train["y"])
        clima = train.groupby(["estacion_codigo", "mes"])["y"].mean().rename("clima").reset_index()
        paquete[cont] = {"modelo": modelo, "clima": clima,
                         "estaciones": sorted(d[d["contaminante"] == cont]["estacion_codigo"].unique())}
    joblib.dump(paquete, MODELO)
    return paquete


def predecir_dia(paquete: dict, base: pd.DataFrame, anterior: pd.DataFrame, meteo: pd.DataFrame,
                 dia_base: pd.Timestamp) -> pd.DataFrame:
    """Predicción para el día siguiente a dia_base, con las métricas de dia_base y del día anterior."""
    objetivo = dia_base + pd.Timedelta(days=1)
    m = meteo[meteo["fecha"] == dia_base] if "fecha" in meteo and len(meteo) else pd.DataFrame()
    salida = []
    for cont, p in paquete.items():
        x = base[base["contaminante"] == cont].rename(columns={"valor": "hoy", "media24": "hoy_media24"})
        x = x[x["estacion_codigo"].isin(p["estaciones"])].copy()
        if x.empty:
            continue
        ay = anterior[anterior["contaminante"] == cont][["estacion_codigo", "valor"]].rename(columns={"valor": "ayer"})
        x = x.merge(ay, on="estacion_codigo", how="left")
        x["mes"], x["dia_semana"] = objetivo.month, objetivo.weekday()
        x = x.merge(p["clima"], on=["estacion_codigo", "mes"], how="left")
        x["estacion_id"] = x["estacion_codigo"].map({e: i for i, e in enumerate(p["estaciones"])})
        for v in predecir.VARIABLES:
            if v not in x:
                x[v] = float(m[v].iloc[0]) if len(m) and v in m and pd.notna(m[v].iloc[0]) else np.nan
        x["pred"] = p["modelo"].predict(x[predecir.VARIABLES]).clip(min=0)
        x["contaminante"] = cont
        salida.append(x[["estacion_codigo", "contaminante", "pred"]])
    return pd.concat(salida, ignore_index=True) if salida else pd.DataFrame(columns=["estacion_codigo", "contaminante", "pred"])


# ---------------------------------------------------------------------------

def por_estacion(df: pd.DataFrame, col: str, idx: dict) -> dict:
    out = {c: {} for c in predecir.CONTS}
    for e, c, v in zip(df["estacion_codigo"], df["contaminante"], df[col]):
        if e in idx and pd.notna(v):
            out[c][idx[e]] = round(float(v), 1)
    return out


def main() -> int:
    paquete = entrenar() if "--entrenar" in sys.argv or not MODELO.exists() else joblib.load(MODELO)
    print("Descargando los datos de la última hora…")
    h, met, redes = descargar_todo()
    if h.empty:
        print("Sin datos: no se actualiza.")
        return 1
    ahora = datetime.now(MADRID)
    hoy = pd.Timestamp(ahora.date())
    if h["fecha"].max() < hoy:  # todavía no hay ninguna hora de hoy
        hoy = h["fecha"].max()
    ayer, anteayer = hoy - pd.Timedelta(days=1), hoy - pd.Timedelta(days=2)
    ult_hora = int(h[h["fecha"] == hoy]["hora"].max())
    meteo = meteo_dia(met)

    m_hoy = metricas_dia(h, hoy, completo=False)
    m_ayer, m_anteayer = metricas_dia(h, ayer, True), metricas_dia(h, anteayer, True)
    estaciones = pd.read_csv(CARPETA_SALIDA / "estaciones.csv", dtype={"codigo": str})
    idx = {c: i for i, c in enumerate(estaciones["codigo"])}

    previsto_hoy = predecir_dia(paquete, m_ayer, m_anteayer, meteo, ayer)
    hay_prevision = ult_hora + 1 >= HORA_PREVISION
    previsto_manana = predecir_dia(paquete, m_hoy, m_ayer, meteo, hoy) if hay_prevision else None

    horas = {c: {} for c in predecir.CONTS}
    for (e, c), x in h[h["fecha"] == hoy].groupby(["estacion_codigo", "var"]):
        if e in idx:
            s = x.set_index("hora")["valor"].reindex(range(24))
            horas[c][idx[e]] = [None if pd.isna(v) else round(float(v)) for v in s]

    salida = {
        "generado": ahora.strftime("%Y-%m-%dT%H:%M"), "fecha": hoy.strftime("%Y-%m-%d"),
        "hora_dato": ult_hora + 1,  # los datos llegan hasta esta hora (la hora 9 es de 8:00 a 9:00)
        "manana": (hoy + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "redes": redes, "ahora": por_estacion(m_hoy, "valor", idx), "horas": horas,
        "previsto_hoy": por_estacion(previsto_hoy, "pred", idx),
        "previsto_manana": por_estacion(previsto_manana, "pred", idx) if previsto_manana is not None else None,
        "hora_prevision": HORA_PREVISION,
    }
    texto = json.dumps(salida, ensure_ascii=False, separators=(",", ":"))
    (CARPETA_SALIDA / "tiempo_real.json").write_text(texto, encoding="utf-8")
    for web in ("demo", "docs"):  # la demo y la web publicada leen este fichero
        if (BASE / web).is_dir():
            (BASE / web / "demo_vivo.js").write_text(f"window.RC_VIVO = {texto};", encoding="utf-8")
    n_est = len({i for c in salida["ahora"].values() for i in c})
    print(f"Datos de {salida['fecha']} hasta las {salida['hora_dato']}:00 · estaciones con dato: {n_est} de {len(idx)} · redes: {redes}")
    for c in predecir.CONTS:
        a, p = salida["ahora"][c], salida["previsto_hoy"][c]
        print(f"  {c:<5} ahora: {len(a):>2} estaciones, máx {max(a.values(), default=0):6.1f} | previsto ayer para hoy: {len(p):>2}, máx {max(p.values(), default=0):6.1f}")
    print("Previsión de mañana:", "publicada" if hay_prevision else f"se publica a las {HORA_PREVISION}:00")
    return 0


if __name__ == "__main__":
    sys.exit(main())
