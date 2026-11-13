"""
RespiraCole · Servidor MCP
==========================

Conecta los datos de RespiraCole con cualquier asistente de IA compatible con
MCP (Claude, ChatGPT, Gemini CLI, Cursor…), para que familias, docentes y
equipos directivos pregunten en lenguaje natural:
  «¿Cómo estará mañana el aire en el colegio de mi hija?»
  «¿Qué colegios públicos de Usera tienen más NO2?»

Herramientas:
  buscar_centros        Busca centros por nombre, municipio o distrito
  ficha_centro          Exposición media, fiabilidad, entorno y episodios de un centro
  semaforo_centro       Semáforo de un día concreto (datos medidos)
  prediccion_centro     Semáforo previsto para el día siguiente a una fecha
  centros_prioritarios  Centros con peor aire de un municipio o distrito

Requiere haber ejecutado el resto de scripts (lee la carpeta datos/).

Uso en Claude Desktop (Ajustes → Desarrollador → Editar configuración):
    {"mcpServers": {"respiracole": {"command": "uv", "args": ["run", "--with", "mcp",
     "--with", "pandas", "python", "/ruta/a/respiracole/servidor_mcp.py"]}}}
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
try:  # SDK de MCP 2.x
    from mcp.server.mcpserver import MCPServer as Servidor
except ImportError:  # SDK de MCP 1.x
    from mcp.server.fastmcp import FastMCP as Servidor

DATOS = Path(__file__).resolve().parent / "datos"
BANDAS = {"NO2": [40, 90, 120, 230, 340], "O3": [50, 100, 130, 240, 380],
          "PM10": [20, 40, 50, 100, 150], "PM25": [10, 20, 25, 50, 75]}
NIVELES = ["buena", "razonablemente buena", "regular", "desfavorable", "muy desfavorable",
           "extremadamente desfavorable"]
OMS = {"NO2": 10, "PM10": 15, "PM25": 5}
COLOR = {1: "verde", 2: "verde", 3: "ámbar"}

mcp = Servidor("respiracole")


@lru_cache
def _centros() -> pd.DataFrame:
    eq = pd.read_csv(DATOS / "centros_equidad.csv", dtype={"codigo": str})
    ctx = pd.read_csv(DATOS / "centros_contexto.csv", dtype={"codigo": str})[
        ["codigo", "trafico_imd_max_200m", "trafico_calle", "zonas_verdes_ha_por_10000_hab"]]
    df = eq.merge(ctx, on="codigo", how="left")
    df["_busca"] = (df["nombre"] + " " + df["municipio"] + " " + df["distrito"].fillna("")).map(_norm)
    return df


@lru_cache
def _metricas() -> pd.DataFrame:
    return pd.read_csv(DATOS / "metricas_diarias.csv", dtype={"estacion_codigo": str})


@lru_cache
def _prediccion() -> pd.DataFrame:
    return pd.read_csv(DATOS / "prediccion_diaria.csv", dtype={"estacion_codigo": str})


@lru_cache
def _episodios() -> pd.DataFrame:
    return pd.read_csv(DATOS / "episodios_centros.csv", dtype={"codigo": str})


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _nivel(cont: str, v: float) -> int:
    return int(np.searchsorted(BANDAS[cont], v, side="left") + 1)


def _centro(codigo: str) -> pd.Series:
    df = _centros()
    fila = df[df["codigo"] == str(codigo).strip()]
    if fila.empty:
        raise ValueError(f"No hay ningún centro con el código {codigo}. Usa buscar_centros.")
    return fila.iloc[0]


def _semaforo(c: pd.Series, tabla: pd.DataFrame, fecha_col: str, valor_col: str, fecha: str):
    dia = tabla[tabla[fecha_col] == fecha]
    if dia.empty:
        raise ValueError(f"No hay datos para {fecha}. El histórico va de 2021 a 2025.")
    detalle, peor = {}, 0
    for cont in BANDAS:
        est = str(c.get(f"{cont}_estacion", "")).split(".")[0]
        fila = dia[(dia["estacion_codigo"] == est) & (dia["contaminante"] == cont)]
        if fila.empty or pd.isna(fila[valor_col].iloc[0]):
            continue
        v = float(fila[valor_col].iloc[0])
        n = _nivel(cont, v)
        detalle[cont] = {"valor_ug_m3": round(v, 1), "nivel": NIVELES[n - 1]}
        peor = max(peor, n)
    if not peor:
        raise ValueError(f"No hay mediciones de las estaciones de este centro el {fecha}.")
    return {"color": COLOR.get(peor, "rojo"), "nivel_peor": NIVELES[peor - 1],
            "contaminantes": detalle,
            "nota": "NO2 y O3: máximo horario de 9:00 a 17:00. Partículas: media del día."}


@mcp.tool()
def buscar_centros(texto: str, limite: int = 10) -> list[dict]:
    """Busca centros escolares de la Comunidad de Madrid por nombre, municipio o distrito.
    Devuelve código, nombre, etapa, titularidad y ubicación de cada coincidencia."""
    palabras = _norm(texto).split()
    df = _centros()
    sel = df[df["_busca"].map(lambda t: all(p in t for p in palabras))].head(max(1, min(limite, 25)))
    return [{"codigo": r.codigo, "nombre": r.nombre, "etapa": r.categoria,
             "titularidad": r.titularidad, "municipio": r.municipio,
             "distrito": r.distrito if isinstance(r.distrito, str) else None}
            for r in sel.itertuples()]


@mcp.tool()
def ficha_centro(codigo: str) -> dict:
    """Ficha de calidad del aire de un centro: exposición media 2021-2025 frente a la guía
    de la OMS, fiabilidad de la estimación, entorno y días en ámbar y rojo por curso."""
    c = _centro(codigo)
    ep = _episodios()
    ep = ep[ep["codigo"] == c["codigo"]][["curso", "dias_ambar", "dias_rojo"]]
    return {
        "codigo": c["codigo"], "nombre": c["nombre"], "etapa": c["categoria"],
        "municipio": c["municipio"], "distrito": c["distrito"] if isinstance(c["distrito"], str) else None,
        "exposicion_media_2021_2025": {
            k: {"ug_m3": c[f"{k}_media"], "veces_guia_oms": round(c[f"{k}_media"] / OMS[k], 1)}
            for k in OMS},
        "fiabilidad": c["fiabilidad"], "distancia_estacion_no2_km": c["NO2_distancia_km"],
        "indice_renta_zona": c["indice_renta"],
        "calle_mas_trafico_200m": (None if pd.isna(c["trafico_imd_max_200m"]) else
                                   {"calle": c["trafico_calle"], "vehiculos_dia": int(c["trafico_imd_max_200m"])}),
        "episodios_por_curso": ep.to_dict("records"),
        "fuentes": "Comunidad de Madrid y Ayuntamiento de Madrid (datos abiertos).",
    }


@mcp.tool()
def semaforo_centro(codigo: str, fecha: str) -> dict:
    """Semáforo de un centro en una fecha (AAAA-MM-DD) entre 2021 y 2025, con los datos medidos."""
    c = _centro(codigo)
    return {"centro": c["nombre"], "fecha": fecha,
            **_semaforo(c, _metricas(), "fecha", "valor", fecha)}


@mcp.tool()
def prediccion_centro(codigo: str, fecha: str) -> dict:
    """Semáforo que RespiraCole habría previsto la víspera para el día siguiente a la fecha
    indicada (AAAA-MM-DD), junto con lo que se midió después."""
    c = _centro(codigo)
    p = _prediccion()
    p = p.assign(fecha=p["fecha"].astype(str).str[:10])
    prevision = _semaforo(c, p, "fecha", "pred", fecha)
    try:
        real = _semaforo(c, p, "fecha", "y", fecha)
    except ValueError:
        real = None
    return {"centro": c["nombre"], "prevision_hecha_el": fecha, "prevision": prevision,
            "lo_que_se_midio": real}


@mcp.tool()
def centros_prioritarios(zona: str, limite: int = 10) -> list[dict]:
    """Centros con más NO2 de un municipio o de un distrito de Madrid capital,
    ordenados de peor a mejor, con su índice de renta (media regional = 100)."""
    z = _norm(zona)
    df = _centros()
    sel = df[(df["municipio"].map(_norm) == z) | (df["distrito"].fillna("").map(_norm) == z)]
    if sel.empty:
        raise ValueError(f"No encuentro el municipio o distrito «{zona}».")
    sel = sel.sort_values("NO2_media", ascending=False).head(max(1, min(limite, 25)))
    return [{"codigo": r.codigo, "nombre": r.nombre, "etapa": r.categoria,
             "no2_media": r.NO2_media, "pm10_media": r.PM10_media,
             "indice_renta": r.indice_renta, "fiabilidad": r.fiabilidad}
            for r in sel.itertuples()]


if __name__ == "__main__":
    mcp.run()
