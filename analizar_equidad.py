"""
RespiraCole · Análisis de equidad
=================================

¿Respiran peor aire los alumnos de las zonas con menos renta?

Requiere haber ejecutado antes preparar_datos.py (usa datos/centros_exposicion.csv).

Fuentes de renta:
  - Comunidad de Madrid: "Indicador de Renta Disponible Bruta Municipal",
    renta per cápita y porcentaje sobre la media regional (2023).
  - Ayuntamiento de Madrid: "Panel de indicadores de distritos y barrios",
    renta media disponible por persona por distrito (2023).

Como las dos fuentes usan metodologías distintas, no se mezclan euros: se
construye un ÍNDICE DE RENTA con la media de la Comunidad = 100.
  - Municipios: el "porcentaje sobre la media" que publica la Comunidad.
  - Distritos de Madrid capital: el índice de la ciudad (Comunidad) escalado
    por la posición de cada distrito dentro de la ciudad (Ayuntamiento):
        índice_distrito = índice_Madrid × renta_distrito / renta_ciudad

Genera en "datos/":
  centros_equidad.csv        Cada centro con su índice de renta y quintil
  equidad_quintiles.csv      Exposición media por quintil de renta
  equidad_titularidad.csv    Exposición por titularidad dentro de cada quintil
  equidad_distritos.csv      Distritos de Madrid: renta y exposición
  equidad_resumen.txt        Hallazgos clave

Uso:
    python analizar_equidad.py
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from preparar_datos import (
    CARPETA_RAW,
    CARPETA_SALIDA,
    GUIA,
    descargar,
    descargar_ayto,
    normalizar_municipio,
    recursos_ayto,
)

ANO_RENTA = 2023
URL_RENTA_CM = ("https://datos.comunidad.madrid/dataset/1715068e-0527-4632-aba1-37da355b346f/"
                "resource/fc9d4336-3c4a-4dd3-8f8f-dac0a4e29033/download/"
                "indicador-de-renta-disponible-bruta-municipal.-municipios.csv")
DS_AYTO_PANEL = "300087-0-indicadores-distritos"
INDICADOR_AYTO = "Renta media disponible por persona"

CONTS = ["NO2", "PM10", "PM25"]
ETIQUETA_Q = {1: "Q1 (renta más baja)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (renta más alta)"}


# ---------------------------------------------------------------------------
# Renta
# ---------------------------------------------------------------------------

def nombre_renta_a_centros(nombre: str) -> str:
    """'Acebeda (La)' -> 'Acebeda, La' (formato del directorio de centros)."""
    m = re.match(r"^(.*) \((El|La|Los|Las)\)$", nombre.strip())
    return f"{m.group(1)}, {m.group(2)}" if m else nombre.strip()


def indice_municipios() -> pd.DataFrame:
    ruta = descargar(URL_RENTA_CM, CARPETA_RAW / "renta_municipal.csv")
    df = pd.read_csv(ruta, sep=";", encoding="latin1", dtype=str)
    df = df[(df["Año"] == str(ANO_RENTA)) & (df["Tipo territorio"] == "Municipios")]
    piv = df.pivot_table(index="Territorio", columns="Medida", values="Valor", aggfunc="first")
    salida = pd.DataFrame({
        "municipio_renta": piv.index,
        "renta_pc_euros": pd.to_numeric(piv["Per cápita"], errors="coerce").to_numpy(),
        "indice_renta": pd.to_numeric(piv["Porcentaje sobre la media"].str.replace(",", "."),
                                      errors="coerce").to_numpy(),
    })
    salida["clave"] = salida["municipio_renta"].map(nombre_renta_a_centros).map(normalizar_municipio)
    return salida


def renta_distritos_madrid() -> pd.DataFrame:
    destino = CARPETA_RAW / "ayuntamiento" / "panel_indicadores_acumulado.csv"
    if not destino.exists():
        url = next(r["url"] for r in recursos_ayto(DS_AYTO_PANEL)
                   if r["url"].endswith("acumulado.csv"))
        destino.parent.mkdir(parents=True, exist_ok=True)
        descargar_ayto(url, destino)
    df = pd.read_csv(destino, sep=";", encoding="utf-8-sig", dtype=str)
    df = df[(df["indicador_completo"] == INDICADOR_AYTO) & (df["año"] == str(ANO_RENTA))
            & df["barrio"].isna()]
    df = df.assign(valor=pd.to_numeric(
        df["valor_indicador"].str.replace(".", "", regex=False).str.replace(",", "."),
        errors="coerce"))
    ciudad = df[df["distrito"].isna()]["valor"].iloc[0]
    distritos = (df[df["distrito"].notna()].groupby("distrito")["valor"].first()
                 .reset_index(name="renta_persona_ayto"))
    distritos.attrs["ciudad"] = ciudad
    return distritos


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def asignar_renta(expo: pd.DataFrame, munis: pd.DataFrame, distritos: pd.DataFrame) -> pd.DataFrame:
    df = expo.copy()
    df["clave"] = df["municipio"].map(normalizar_municipio)
    df = df.merge(munis[["clave", "indice_renta"]], on="clave", how="left")

    indice_madrid = munis.loc[munis["clave"] == "MADRID", "indice_renta"].iloc[0]
    ciudad = distritos.attrs["ciudad"]
    distritos = distritos.assign(
        indice_distrito=(indice_madrid * distritos["renta_persona_ayto"] / ciudad).round(1))
    df = df.merge(distritos[["distrito", "indice_distrito"]], on="distrito", how="left")

    capital = df["clave"] == "MADRID"
    # Centros de la capital: índice del distrito (si falta el distrito, el de la ciudad)
    df.loc[capital, "indice_renta"] = df.loc[capital, "indice_distrito"].fillna(indice_madrid)
    df["ambito_renta"] = np.where(capital & df["indice_distrito"].notna(), "distrito",
                                  "municipio")
    df = df.drop(columns=["clave", "indice_distrito"])

    con = df["indice_renta"].notna()
    df.loc[con, "quintil_renta"] = pd.qcut(df.loc[con, "indice_renta"].rank(method="first"),
                                           5, labels=[1, 2, 3, 4, 5]).astype(int)
    return df


def tabla_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for q, g in df.dropna(subset=["quintil_renta"]).groupby("quintil_renta"):
        fila = {"quintil": ETIQUETA_Q[int(q)], "centros": len(g),
                "indice_renta_min": g["indice_renta"].min(),
                "indice_renta_max": g["indice_renta"].max()}
        for c in CONTS:
            fila[f"{c}_media"] = round(g[f"{c}_media"].mean(), 1)
            fila[f"{c}_veces_oms"] = round(g[f"{c}_media"].mean() / GUIA[c]["oms_anual"], 2)
        fila["NO2_dias_supera_oms"] = round(g["NO2_dias_supera_oms"].mean(), 0)
        fila["PM10_dias_supera_oms"] = round(g["PM10_dias_supera_oms"].mean(), 0)
        fila["pct_centros_NO2_mas_del_doble_oms"] = round(
            100 * (g["NO2_media"] > 2 * GUIA["NO2"]["oms_anual"]).mean(), 1)
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_titularidad(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=["quintil_renta"])
    g = g[g["titularidad"].isin(["PÚBLICO", "PRIVADO CONCERTADO", "PRIVADO"])]
    t = (g.groupby(["quintil_renta", "titularidad"])
         .agg(centros=("codigo", "size"), NO2_media=("NO2_media", "mean"))
         .reset_index())
    t["NO2_media"] = t["NO2_media"].round(1)
    t["quintil"] = t["quintil_renta"].astype(int).map(ETIQUETA_Q)
    return t[["quintil", "titularidad", "centros", "NO2_media"]]


def tabla_distritos(df: pd.DataFrame, distritos: pd.DataFrame) -> pd.DataFrame:
    cap = df[(df["municipio"] == "Madrid") & (df["ambito_renta"] == "distrito")]
    t = (cap.groupby("distrito")
         .agg(centros=("codigo", "size"), indice_renta=("indice_renta", "first"),
              NO2_media=("NO2_media", "mean"), PM10_media=("PM10_media", "mean"),
              publicos=("titularidad", lambda s: (s == "PÚBLICO").sum()))
         .reset_index())
    t = t.merge(distritos, on="distrito", how="left")
    for c in ("NO2_media", "PM10_media"):
        t[c] = t[c].round(1)
    return t.sort_values("indice_renta")


def spearman(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])


def construir_resumen(df, q, tit, dist) -> str:
    L = ["RESPIRACOLE · ANÁLISIS DE EQUIDAD", "=" * 50, "",
         f"Renta: índice con la media de la Comunidad de Madrid = 100 (año {ANO_RENTA}).",
         f"Centros con índice de renta: {df['indice_renta'].notna().sum():,} de {len(df):,}",
         f"  (capital por distrito: {(df['ambito_renta'] == 'distrito').sum():,};"
         f" resto por municipio)", ""]

    L.append("EXPOSICIÓN MEDIA POR QUINTIL DE RENTA (µg/m³; entre paréntesis, veces la guía OMS)")
    L.append(f"  {'':22}{'índice':>14}{'NO2':>14}{'PM10':>14}{'PM2,5':>14}{'centros':>9}")
    for _, f in q.iterrows():
        rango = f"{f['indice_renta_min']:.0f}-{f['indice_renta_max']:.0f}"
        L.append(f"  {f['quintil']:<22}{rango:>14}"
                 + "".join(f"{f[c + '_media']:>8.1f} ({f[c + '_veces_oms']:.1f}x)" for c in CONTS)
                 + f"{f['centros']:>9}")
    q1, q5 = q.iloc[0], q.iloc[-1]
    L += ["",
          f"Brecha NO2 Q1 vs Q5: {q1['NO2_media']:.1f} vs {q5['NO2_media']:.1f} µg/m³ "
          f"({(q1['NO2_media'] / q5['NO2_media'] - 1):+.0%})",
          f"Brecha PM10 Q1 vs Q5: {q1['PM10_media']:.1f} vs {q5['PM10_media']:.1f} µg/m³ "
          f"({(q1['PM10_media'] / q5['PM10_media'] - 1):+.0%})",
          f"Centros con NO2 > doble de la guía OMS: Q1 {q1['pct_centros_NO2_mas_del_doble_oms']:.0f} %"
          f" · Q5 {q5['pct_centros_NO2_mas_del_doble_oms']:.0f} %",
          "",
          "CORRELACIÓN DE SPEARMAN ENTRE RENTA Y EXPOSICIÓN (negativa = menos renta, más contaminación)"]
    for nombre, sub in [("Toda la Comunidad", df),
                        ("Madrid capital (distritos)", df[df["ambito_renta"] == "distrito"]),
                        ("Resto de municipios", df[df["municipio"] != "Madrid"])]:
        vals = "  ".join(f"{c} {spearman(sub['indice_renta'], sub[c + '_media']):+.2f}" for c in CONTS)
        L.append(f"  {nombre:<28}{vals}")

    L += ["", "MADRID CAPITAL: LOS 5 DISTRITOS CON MENOS Y MÁS RENTA"]
    L.append(f"  {'distrito':<22}{'índice':>8}{'NO2':>8}{'PM10':>8}{'centros':>9}")
    for _, f in pd.concat([dist.head(5), dist.tail(5)]).iterrows():
        L.append(f"  {f['distrito']:<22}{f['indice_renta']:>8.0f}{f['NO2_media']:>8.1f}"
                 f"{f['PM10_media']:>8.1f}{f['centros']:>9}")

    L += ["", "NO2 MEDIO POR TITULARIDAD DENTRO DE CADA QUINTIL"]
    piv = tit.pivot(index="quintil", columns="titularidad", values="NO2_media")
    L.append(f"  {'':22}" + "".join(f"{c[:18]:>20}" for c in piv.columns))
    for idx, fila in piv.iterrows():
        L.append(f"  {idx:<22}" + "".join(f"{v:>20.1f}" for v in fila.to_numpy()))
    L += ["", "LIMITACIONES",
          "  · Cada centro cuenta igual: no hay datos públicos de alumnado por centro.",
          "  · La renta es la del municipio o distrito, no la de las familias del centro.",
          "  · La exposición es la de la estación más cercana, no una medición en el centro."]
    return "\n".join(L) + "\n"


def main() -> int:
    expo = pd.read_csv(CARPETA_SALIDA / "centros_exposicion.csv", dtype={"codigo": str})
    print("Descargando datos de renta…")
    munis = indice_municipios()
    distritos = renta_distritos_madrid()

    print("Analizando…")
    df = asignar_renta(expo, munis, distritos)
    q = tabla_quintiles(df)
    tit = tabla_titularidad(df)
    dist = tabla_distritos(df, distritos)

    df.to_csv(CARPETA_SALIDA / "centros_equidad.csv", index=False)
    q.to_csv(CARPETA_SALIDA / "equidad_quintiles.csv", index=False)
    tit.to_csv(CARPETA_SALIDA / "equidad_titularidad.csv", index=False)
    dist.to_csv(CARPETA_SALIDA / "equidad_distritos.csv", index=False)
    texto = construir_resumen(df, q, tit, dist)
    (CARPETA_SALIDA / "equidad_resumen.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
