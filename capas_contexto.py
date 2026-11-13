"""
RespiraCole · Capas de contexto
===============================

Añade a cada centro información que ayuda a explicar su calidad del aire.
Requiere haber ejecutado antes preparar_datos.py y analizar_equidad.py.

Capas (en todos los centros, por municipio, datos de la Comunidad de Madrid):
  - Densidad de población (hab/km²), 2025.
  - Parque de vehículos por distintivo ambiental, 2025: vehículos por cada
    1.000 habitantes y % de vehículos sin distintivo (los más contaminantes).
    Población del padrón municipal 2025.

Capas solo en Madrid capital (por distrito o por calle, datos del Ayuntamiento):
  - Densidad de población del distrito (hab/ha) y superficie de zonas verdes
    y parques de distrito por cada 10.000 habitantes (panel de indicadores).
  - Tráfico en calles cercanas: la intensidad media diaria (IMD, 2020) más alta
    entre los puntos de aforo a menos de 200 m del centro, con el nombre de la
    calle y la distancia.
    ATENCIÓN: los aforos son de la red municipal. No incluyen autovías ni
    circunvalaciones estatales (M-30 parcialmente, M-40, A-4, A-42…), así que
    este dato mide el tráfico en CALLES CERCANAS, no el tráfico total del entorno.

Genera en "datos/":
  centros_contexto.csv     Una fila por centro con todas las capas
  contexto_resumen.txt     Cifras clave y relación de cada capa con el NO2

Uso:
    pip install geopandas openpyxl
    python capas_contexto.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analizar_equidad import nombre_renta_a_centros, spearman
from preparar_datos import (
    CARPETA_RAW,
    CARPETA_SALIDA,
    descargar,
    descargar_ayto,
    normalizar_municipio,
    recursos_ayto,
    url_csv,
)

ANO_MUNI = 2025          # vehículos, padrón y densidad municipal
RADIO_TRAFICO_M = 200    # distancia máxima a un punto de aforo
UMBRAL_VIA_INTENSA = 20_000  # vehículos/día para considerar una vía "intensa"

DS_VEHICULOS = "parque_vehiculos_por_distintivo"
DS_PADRON = "padron_por_sexo"
DS_DENSIDAD = "1930748"
DS_AYTO_IMD = "203962-0-trafico-imd"
ANO_IMD = "2020"

IND_DENSIDAD_DISTRITO = "Población densidad (hab./ha.)"
IND_VERDE_DISTRITO = ("Superficie de zonas verdes y parques de distrito (ha.) "
                      "entre número de habitantes *10.000")

CARPETA_F = CARPETA_RAW / "factores"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def leer_estadistica_cm(ruta) -> pd.DataFrame:
    """Ficheros del Instituto de Estadística de la Comunidad (latin1, ';')."""
    df = pd.read_csv(ruta, sep=";", encoding="latin1", dtype=str)
    df = df[df["Tipo territorio"] == "Municipios"].copy()
    df["valor"] = pd.to_numeric(df["Valor"].str.replace(",", "."), errors="coerce")
    df["clave"] = df["Territorio"].map(nombre_renta_a_centros).map(normalizar_municipio)
    return df


def descargar_cm(id_dataset: str, nombre: str):
    CARPETA_F.mkdir(parents=True, exist_ok=True)
    return descargar(url_csv(id_dataset), CARPETA_F / nombre)


# ---------------------------------------------------------------------------
# Capas municipales (Comunidad de Madrid)
# ---------------------------------------------------------------------------

def capas_municipales() -> pd.DataFrame:
    veh = leer_estadistica_cm(descargar_cm(DS_VEHICULOS, "vehiculos_distintivo.csv"))
    veh = veh[veh["Año"] == str(ANO_MUNI)]
    piv = veh.pivot_table(index="clave", columns="Distintivo", values="valor", aggfunc="sum")
    total = piv.sum(axis=1)

    pad = leer_estadistica_cm(descargar_cm(DS_PADRON, "padron_por_sexo.csv"))
    pad = pad[(pad["Año"] == str(ANO_MUNI)) & (pad["Sexo"] == "Total")]
    poblacion = pad.set_index("clave")["valor"]

    den = leer_estadistica_cm(descargar_cm(DS_DENSIDAD, "densidad_municipios.csv"))
    den = den[den["Año"] == str(ANO_MUNI)].set_index("clave")["valor"]

    salida = pd.DataFrame({
        "poblacion_municipio": poblacion,
        "densidad_municipio_hab_km2": den,
        "vehiculos_por_1000_hab": (1000 * total / poblacion),
        "pct_vehiculos_sin_distintivo": (100 * piv.get("Sin distintivo") / total),
    }).round(1)
    # Municipios donde se domicilian flotas (renting, alquiler) inflan el dato
    # (hasta 100.000 vehículos por 1.000 hab.) y, al ser coches nuevos, también
    # rebajan el % sin distintivo: en ellos dejamos vacíos los datos de vehículos.
    salida["flag_flotas"] = salida["vehiculos_por_1000_hab"] > 1500
    salida.loc[salida["flag_flotas"], ["vehiculos_por_1000_hab",
                                       "pct_vehiculos_sin_distintivo"]] = np.nan
    return salida.reset_index(names="clave")


# ---------------------------------------------------------------------------
# Capas de distrito (Ayuntamiento de Madrid)
# ---------------------------------------------------------------------------

def capas_distritos() -> pd.DataFrame:
    ruta = CARPETA_RAW / "ayuntamiento" / "panel_indicadores_acumulado.csv"
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", dtype=str)
    df = df[df["barrio"].isna() & df["distrito"].notna()]
    df = df.assign(valor=pd.to_numeric(
        df["valor_indicador"].str.replace(".", "", regex=False).str.replace(",", "."),
        errors="coerce"))

    def ultimo(indicador: str, columna: str) -> pd.Series:
        sub = df[df["indicador_completo"] == indicador].copy()
        sub["año_n"] = pd.to_numeric(sub["año"], errors="coerce")
        sub = sub.sort_values("año_n").groupby("distrito").tail(1)
        return sub.set_index("distrito")["valor"].rename(columna)

    salida = pd.concat([
        ultimo(IND_DENSIDAD_DISTRITO, "densidad_distrito_hab_ha"),
        ultimo(IND_VERDE_DISTRITO, "zonas_verdes_ha_por_10000_hab"),
    ], axis=1)
    return salida.reset_index(names="distrito")


def trafico_calles_cercanas(centros_capital: pd.DataFrame) -> pd.DataFrame:
    """Vía con más tráfico (IMD) a menos de RADIO_TRAFICO_M de cada centro."""
    import zipfile

    import geopandas as gpd

    carpeta = CARPETA_F / f"imd_tramos_{ANO_IMD}"
    if not carpeta.exists():
        url = next(r["url"] for r in recursos_ayto(DS_AYTO_IMD)
                   if (r.get("format") or "").upper() == "ZIP"
                   and (r.get("description") or "").strip().endswith(ANO_IMD))
        f_zip = descargar_ayto(url, CARPETA_F / f"imd_tramos_{ANO_IMD}.zip")
        with zipfile.ZipFile(f_zip) as z:
            z.extractall(carpeta)

    # El shapefile trae las coordenadas, pero sus cifras tienen los separadores
    # de miles mal leídos; el Excel trae las cifras limpias en el mismo orden.
    puntos = gpd.read_file(next(carpeta.glob("*.shp")))
    puntos = puntos[puntos.geometry.notna()].reset_index(drop=True)
    excel = pd.read_excel(next(carpeta.glob("*.xlsx")))
    if len(excel) != len(puntos):
        raise RuntimeError("El Excel y el shapefile de tráfico no tienen las mismas filas")
    puntos = puntos.set_crs(4326, allow_override=True).to_crs(25830)
    puntos["imd"] = excel["IMD"].to_numpy()
    puntos["calle"] = excel["NOMBRE"].astype(str).str.strip().to_numpy()

    cx = centros_capital["utm_x"].to_numpy(float)[:, None]
    cy = centros_capital["utm_y"].to_numpy(float)[:, None]
    dist = np.hypot(cx - puntos.geometry.x.to_numpy()[None, :],
                    cy - puntos.geometry.y.to_numpy()[None, :])
    imd = np.where(dist <= RADIO_TRAFICO_M, puntos["imd"].to_numpy()[None, :], -1)
    idx = imd.argmax(axis=1)
    hay = imd[np.arange(len(idx)), idx] >= 0

    salida = pd.DataFrame({"codigo": centros_capital["codigo"].to_numpy()})
    salida["trafico_imd_max_200m"] = np.where(hay, puntos["imd"].to_numpy()[idx], np.nan)
    salida["trafico_calle"] = np.where(hay, puntos["calle"].to_numpy()[idx], None)
    salida["trafico_distancia_m"] = np.where(hay, dist[np.arange(len(idx)), idx].round(0), np.nan)
    salida["via_intensa_200m"] = salida["trafico_imd_max_200m"] >= UMBRAL_VIA_INTENSA
    return salida


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main() -> int:
    eq = pd.read_csv(CARPETA_SALIDA / "centros_equidad.csv", dtype={"codigo": str})
    pos = pd.read_csv(CARPETA_SALIDA / "centros.csv", dtype={"codigo": str})[
        ["codigo", "utm_x", "utm_y"]]
    df = eq.merge(pos, on="codigo", how="left")

    print("Capas municipales (Comunidad de Madrid)…")
    df["clave"] = df["municipio"].map(normalizar_municipio)
    df = df.merge(capas_municipales(), on="clave", how="left").drop(columns="clave")

    print("Capas de distrito y tráfico (Ayuntamiento de Madrid)…")
    capital = df["municipio"] == "Madrid"
    df = df.merge(capas_distritos(), on="distrito", how="left")
    df.loc[~capital, ["densidad_distrito_hab_ha", "zonas_verdes_ha_por_10000_hab"]] = np.nan
    df = df.merge(trafico_calles_cercanas(df[capital]), on="codigo", how="left")

    columnas = ["codigo", "nombre", "categoria", "municipio", "distrito", "titularidad",
                "NO2_media", "PM10_media", "PM25_media", "indice_renta",
                "poblacion_municipio", "densidad_municipio_hab_km2", "vehiculos_por_1000_hab",
                "pct_vehiculos_sin_distintivo", "flag_flotas",
                "densidad_distrito_hab_ha", "zonas_verdes_ha_por_10000_hab",
                "trafico_imd_max_200m", "trafico_calle", "trafico_distancia_m",
                "via_intensa_200m"]
    salida = df[columnas]
    salida.to_csv(CARPETA_SALIDA / "centros_contexto.csv", index=False)

    texto = construir_resumen(salida)
    (CARPETA_SALIDA / "contexto_resumen.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    return 0


def construir_resumen(df: pd.DataFrame) -> str:
    cap = df[df["municipio"] == "Madrid"]
    resto = df[df["municipio"] != "Madrid"]
    L = ["RESPIRACOLE · CAPAS DE CONTEXTO", "=" * 50, "",
         "COBERTURA",
         f"  Centros con densidad del municipio: "
         f"{df['densidad_municipio_hab_km2'].notna().sum():,} de {len(df):,}",
         f"  Centros con datos de vehículos del municipio: "
         f"{df['vehiculos_por_1000_hab'].notna().sum():,} de {len(df):,}",
         f"  Centros de la capital con densidad y zonas verdes de distrito: "
         f"{cap['zonas_verdes_ha_por_10000_hab'].notna().sum():,} de {len(cap):,}",
         f"  Centros de la capital con un punto de aforo a < {RADIO_TRAFICO_M} m: "
         f"{cap['trafico_imd_max_200m'].notna().sum():,} de {len(cap):,}",
         f"  … junto a una calle de más de {UMBRAL_VIA_INTENSA:,} vehículos/día: "
         f"{int(cap['via_intensa_200m'].fillna(False).sum()):,}",
         f"  Municipios sin dato de vehículos por flotas domiciliadas (>1.500 veh./1.000 hab.): "
         f"{df.loc[df['flag_flotas'].fillna(False).astype(bool), 'municipio'].nunique()}",
         "",
         "RELACIÓN CON EL NO2 (correlación de Spearman; + = más de esto, más NO2)"]
    pares = [
        ("Densidad del municipio (fuera de la capital)", resto, "densidad_municipio_hab_km2"),
        ("Vehículos por 1.000 hab. (fuera de la capital)", resto, "vehiculos_por_1000_hab"),
        ("% vehículos sin distintivo (fuera de la capital)", resto,
         "pct_vehiculos_sin_distintivo"),
        ("Densidad del distrito (capital)", cap, "densidad_distrito_hab_ha"),
        ("Zonas verdes por habitante (capital)", cap, "zonas_verdes_ha_por_10000_hab"),
        ("Tráfico en calles cercanas (capital)", cap, "trafico_imd_max_200m"),
    ]
    for nombre, sub, col in pares:
        L.append(f"  {nombre:<60}{spearman(sub[col], sub['NO2_media']):+.2f}")
    L += ["",
          "NOTAS",
          "  · La exposición al NO2 viene de la estación más cercana; la densidad y el",
          "    tráfico se miden en el entorno, así que sirven para explicar y contextualizar,",
          "    no para recalcular la exposición.",
          "  · El tráfico de calles cercanas no incluye autovías ni circunvalaciones estatales.",
          "  · El parque de vehículos se cuenta por municipio de domicilio, no de circulación."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
