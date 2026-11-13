"""
RespiraCole · Predicción para mañana
====================================

Predice, la víspera, el valor de cada contaminante en cada estación para el día
siguiente, con el que se calcula el semáforo de mañana de cada centro.

Qué se predice (las mismas métricas del semáforo de "Hoy"):
  - NO2 y O3: valor horario más alto entre las 9:00 y las 17:00
  - PM10 y PM2,5: media del día

Con qué datos (todo de la Comunidad y del Ayuntamiento de Madrid):
  - Histórico horario de calidad del aire de las dos redes (2021-2025)
  - Datos meteorológicos horarios de la Red de Calidad del Aire de la Comunidad
    (viento, temperatura, humedad, presión, radiación y lluvia), como media regional

Modelo: un modelo de árboles de decisión potenciados (HistGradientBoosting) por
contaminante, entrenado con 2021-2024 y evaluado con 2025 frente a la
"persistencia" (suponer que mañana será igual que hoy), que es la referencia
habitual en predicción de calidad del aire.

Limitación: solo usa lo que ya ha pasado. En producción se añadiría la
predicción meteorológica de AEMET y la de polvo sahariano de Copernicus (CAMS),
sin las que un episodio de calima que empieza de golpe no se puede anticipar.

Genera en "datos/":
  prediccion_evaluacion.txt       Acierto del modelo frente a la persistencia
  prediccion_diaria.csv           Valor real y predicho por estación, contaminante y día
                                  (siempre fuera de muestra; ver entrenar_y_evaluar)
  metricas_diarias.csv            Métricas diarias por estación (usadas por otros scripts)

Uso:
    pip install scikit-learn
    python predecir.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

from preparar_datos import (
    ANOS,
    CABECERAS,
    CARPETA_RAW,
    CARPETA_SALIDA,
    a_numero,
    descargar,
    descargar_red_ayuntamiento,
    leer_csv,
    leer_historico,
    recursos,
    url_csv,
)

DS_METEO = "calidad_aire_datos_meteo_historico"
METEO = {81: "viento", 83: "temp", 86: "humedad", 87: "presion", 88: "radiacion", 89: "lluvia"}

# Bandas del Índice de Calidad del Aire (µg/m³): límite superior de los niveles 1 a 5
BANDAS = {
    "NO2": [40, 90, 120, 230, 340],
    "O3": [50, 100, 130, 240, 380],
    "PM10": [20, 40, 50, 100, 150],
    "PM25": [10, 20, 25, 50, 75],
}
CONTS = list(BANDAS)
ANO_TEST = 2025


def nivel(cont: str, v):
    """Nivel 1-6 del Índice de Calidad del Aire."""
    return np.searchsorted(BANDAS[cont], v, side="left") + 1


def color(niv):
    """0 = verde (niveles 1-2), 1 = ámbar (3), 2 = rojo (4 o más)."""
    return np.select([niv <= 2, niv == 3], [0, 1], 2)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def metricas_diarias() -> pd.DataFrame:
    """Métrica del semáforo por estación, contaminante y día."""
    ficheros = sorted(CARPETA_RAW.glob("historico_*.csv"))
    _, f_ayto = descargar_red_ayuntamiento()
    h = pd.concat([leer_historico(f) for f in ficheros + f_ayto], ignore_index=True)
    h = h.drop_duplicates(["estacion_codigo", "contaminante", "fecha", "hora"])

    lect = h[h["hora"].between(9, 16)]
    pico = (lect[lect["contaminante"].isin(["NO2", "O3"])]
            .groupby(["estacion_codigo", "contaminante", "fecha"])["valor"]
            .agg(["max", "count"]).reset_index())
    pico = pico[pico["count"] >= 6].rename(columns={"max": "valor"})

    media = (h[h["contaminante"].isin(["PM10", "PM25"])]
             .groupby(["estacion_codigo", "contaminante", "fecha"])["valor"]
             .agg(["mean", "count"]).reset_index())
    media = media[media["count"] >= 18].rename(columns={"mean": "valor"})

    # Media de todo el día también para NO2 y O3 (variable explicativa)
    media24 = (h.groupby(["estacion_codigo", "contaminante", "fecha"])["valor"]
               .mean().rename("media24").reset_index())

    d = pd.concat([pico, media], ignore_index=True)[["estacion_codigo", "contaminante",
                                                      "fecha", "valor"]]
    d = d.merge(media24, on=["estacion_codigo", "contaminante", "fecha"], how="left")
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d


def meteo_regional() -> pd.DataFrame:
    """Meteorología diaria, media de las estaciones de la Comunidad."""
    trozos = []
    for ano in ANOS:
        f = descargar(url_csv(DS_METEO, str(ano)), CARPETA_RAW / "meteo" / f"meteo_{ano}.csv")
        df = leer_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        df["magnitud"] = pd.to_numeric(df["magnitud"], errors="coerce")
        df = df[df["magnitud"].isin(METEO)]
        df["fecha"] = pd.to_datetime(df["ano"] + "-" + df["mes"].str.zfill(2) + "-"
                                     + df["dia"].str.zfill(2), errors="coerce")
        horas = []
        for h in range(1, 25):
            t = df[["magnitud", "fecha", "estacion"]].copy()
            t["valor"] = a_numero(df[f"h{h:02d}"])
            t["ok"] = df[f"v{h:02d}"].str.strip() == "V"
            horas.append(t)
        largo = pd.concat(horas)
        trozos.append(largo[largo["ok"] & largo["valor"].notna()])
    m = pd.concat(trozos)
    m["var"] = m["magnitud"].map(METEO)

    agg = {"viento": ["mean", "max"], "temp": ["max", "min"], "humedad": ["mean"],
           "presion": ["mean"], "radiacion": ["sum"], "lluvia": ["sum"]}
    filas = []
    for var, funcs in agg.items():
        sub = m[m["var"] == var]
        # primero por estación y día, luego media regional
        por_est = sub.groupby(["fecha", "estacion"])["valor"].agg(funcs)
        regional = por_est.groupby("fecha").mean()
        regional.columns = [f"{var}_{f}" for f in regional.columns]
        filas.append(regional)
    out = pd.concat(filas, axis=1).sort_index()
    out["presion_cambio"] = out["presion_mean"].diff()
    return out.reset_index()


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def preparar_tabla(d: pd.DataFrame, meteo: pd.DataFrame, cont: str) -> pd.DataFrame:
    x = d[d["contaminante"] == cont].sort_values(["estacion_codigo", "fecha"]).copy()
    x = x.set_index(["estacion_codigo", "fecha"])
    idx = x.index
    # Valor de hoy, de ayer y media de hoy, para predecir mañana
    hoy = x[["valor", "media24"]].rename(columns={"valor": "hoy", "media24": "hoy_media24"})
    manana = x["valor"].rename("y")
    ayer = x["valor"].rename("ayer")

    t = hoy.reset_index()
    t["fecha_obj"] = t["fecha"] + pd.Timedelta(days=1)
    t = t.merge(manana.reset_index().rename(columns={"fecha": "fecha_obj"}),
                on=["estacion_codigo", "fecha_obj"], how="left")
    t = t.merge(ayer.reset_index().assign(fecha=lambda z: z["fecha"] + pd.Timedelta(days=1)),
                on=["estacion_codigo", "fecha"], how="left")
    t = t.merge(meteo, on="fecha", how="left")
    t["mes"] = t["fecha_obj"].dt.month
    t["dia_semana"] = t["fecha_obj"].dt.weekday
    t["estacion_id"] = t["estacion_codigo"].astype("category").cat.codes
    return t


def con_clima(t: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Añade la climatología de la estación y el mes, calculada solo con el entrenamiento."""
    clim = (t[train_mask].groupby(["estacion_codigo", "mes"])["y"].mean()
            .rename("clima").reset_index())
    return t.drop(columns="clima", errors="ignore").merge(clim, on=["estacion_codigo", "mes"],
                                                          how="left")


def nuevo_modelo():
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=0.6, max_iter=400, learning_rate=0.05,
        categorical_features=[VARIABLES.index("estacion_id")], random_state=0)


VARIABLES = ["hoy", "hoy_media24", "ayer", "clima", "mes", "dia_semana", "estacion_id",
             "viento_mean", "viento_max", "temp_max", "temp_min", "humedad_mean",
             "presion_mean", "presion_cambio", "radiacion_sum", "lluvia_sum"]


def entrenar_y_evaluar(d: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """Predicción fuera de muestra para todos los años.

    - 2025 (evaluación oficial): modelo entrenado solo con 2021-2024.
    - Años anteriores (para la demo): modelo entrenado con el resto de años,
      de modo que ningún día se predice con un modelo que lo haya visto.
    """
    predicciones = []
    for cont in CONTS:
        base = preparar_tabla(d, meteo, cont)
        for ano in ANOS:
            if ano == ANO_TEST:
                train_mask = base["fecha_obj"].dt.year < ANO_TEST
            else:
                train_mask = base["fecha_obj"].dt.year != ano
            t = con_clima(base, train_mask)
            train = t[train_mask].dropna(subset=["y", "hoy"])
            test = t[t["fecha_obj"].dt.year == ano].dropna(subset=["hoy"]).copy()
            modelo = nuevo_modelo().fit(train[VARIABLES], train["y"])
            test["pred"] = modelo.predict(test[VARIABLES]).clip(min=0)
            test["contaminante"] = cont
            predicciones.append(test[["estacion_codigo", "contaminante", "fecha", "fecha_obj",
                                      "hoy", "y", "pred"]])
    return pd.concat(predicciones, ignore_index=True)


def evaluar(pred: pd.DataFrame) -> str:
    """Acierto del semáforo de cada estación (peor contaminante) en días lectivos."""
    p = pred[(pred["fecha_obj"].dt.year == ANO_TEST) & pred["y"].notna()].copy()
    for col, dest in (("y", "n_real"), ("pred", "n_pred"), ("hoy", "n_pers")):
        p[dest] = [int(nivel(c, v)) for c, v in zip(p["contaminante"], p[col])]
    lectivo = (p["fecha_obj"].dt.weekday < 5) & (~p["fecha_obj"].dt.month.isin([7, 8]))
    p = p[lectivo]
    est = p.groupby(["estacion_codigo", "fecha_obj"])[["n_real", "n_pred", "n_pers"]].max()
    real, mod, pers = color(est["n_real"]), color(est["n_pred"]), color(est["n_pers"])

    def resumen(nombre, pr):
        acierto = (pr == real).mean()
        alerta_real, alerta_pr = real >= 1, pr >= 1
        tp = (alerta_real & alerta_pr).sum()
        recall = tp / max(alerta_real.sum(), 1)
        precision = tp / max(alerta_pr.sum(), 1)
        rojo_real, rojo_pr = real == 2, pr == 2
        recall_rojo = (rojo_real & rojo_pr).sum() / max(rojo_real.sum(), 1)
        return (f"  {nombre:<26}{acierto:>9.1%}{recall:>12.1%}{precision:>12.1%}"
                f"{recall_rojo:>12.1%}")

    L = [f"RESPIRACOLE · EVALUACIÓN DE LA PREDICCIÓN PARA MAÑANA ({ANO_TEST}, días lectivos)",
         "=" * 78, "",
         f"Estación-días evaluados: {len(est):,}",
         f"  verde {np.mean(real == 0):.1%} · ámbar {np.mean(real == 1):.1%} · "
         f"rojo {np.mean(real == 2):.1%}", "",
         f"  {'':26}{'acierto':>9}{'detecta':>12}{'precisión':>12}{'detecta':>12}",
         f"  {'':26}{'color':>9}{'ámbar/rojo':>12}{'avisos':>12}{'rojo':>12}",
         resumen("Persistencia (= hoy)", pers),
         resumen("Modelo RespiraCole", mod), "",
         "Error medio absoluto (µg/m³), días lectivos:"]
    for cont in CONTS:
        s = p[p["contaminante"] == cont]
        L.append(f"  {cont:<6} modelo {np.mean(np.abs(s['pred'] - s['y'])):6.1f}"
                 f"   persistencia {np.mean(np.abs(s['hoy'] - s['y'])):6.1f}")
    L += ["", "Notas:",
          "  · 'Detecta ámbar/rojo': de los días que acabaron en ámbar o rojo, cuántos se avisaron.",
          "  · 'Precisión avisos': de los avisos dados, cuántos acertaron.",
          "  · Sin predicción meteorológica ni de polvo: una calima repentina no se anticipa."]
    return "\n".join(L) + "\n"


def main() -> int:
    print("Calculando métricas diarias de las dos redes…")
    d = metricas_diarias()
    d.to_csv(CARPETA_SALIDA / "metricas_diarias.csv", index=False)
    print("Preparando la meteorología regional…")
    (CARPETA_RAW / "meteo").mkdir(parents=True, exist_ok=True)
    meteo = meteo_regional()
    print("Entrenando y evaluando…")
    pred = entrenar_y_evaluar(d, meteo)
    pred.to_csv(CARPETA_SALIDA / "prediccion_diaria.csv", index=False)
    texto = evaluar(pred)
    (CARPETA_SALIDA / "prediccion_evaluacion.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.RequestException as e:
        print(f"No he podido conectar con el portal de datos: {e}")
        sys.exit(1)
