"""
RespiraCole · Histórico de episodios por centro
===============================================

Cuenta, para cada centro y curso escolar, cuántos días lectivos habría tenido
el semáforo en ámbar y en rojo. Es la base del panel para redes de centros y
del informe anual.

Requiere haber ejecutado predecir.py (datos/metricas_diarias.csv) y
analizar_equidad.py (datos/centros_equidad.csv).

Reglas:
  - Día lectivo: de lunes a viernes, de septiembre a junio (no se descuentan
    festivos ni vacaciones de Navidad y Semana Santa, que no están en los datos).
  - Color del centro: el peor nivel del Índice de Calidad del Aire entre los
    contaminantes, cada uno tomado de la estación más cercana que lo mide.
    Verde: niveles 1-2 · Ámbar: nivel 3 · Rojo: nivel 4 o más.

Genera datos/episodios_centros.csv: una fila por centro y curso.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from predecir import BANDAS, CONTS, nivel
from preparar_datos import CARPETA_SALIDA


def curso(fecha: pd.Series) -> pd.Series:
    inicio = np.where(fecha.dt.month >= 9, fecha.dt.year, fecha.dt.year - 1)
    return pd.Series([f"{a}-{str(a + 1)[2:]}" for a in inicio], index=fecha.index)


def main() -> int:
    m = pd.read_csv(CARPETA_SALIDA / "metricas_diarias.csv", dtype={"estacion_codigo": str},
                    parse_dates=["fecha"])
    m = m[(m["fecha"].dt.weekday < 5) & (~m["fecha"].dt.month.isin([7, 8]))]
    m["nivel"] = [int(nivel(c, v)) for c, v in zip(m["contaminante"], m["valor"])]
    tabla = m.pivot_table(index="fecha", columns=["contaminante", "estacion_codigo"],
                          values="nivel", aggfunc="max")

    c = pd.read_csv(CARPETA_SALIDA / "centros_equidad.csv", dtype=str)
    cols = [f"{k}_estacion" for k in CONTS]
    combos = c[cols].drop_duplicates()
    fechas = tabla.index.to_series()
    cursos = curso(fechas)

    filas = []
    for _, combo in combos.iterrows():
        niveles = []
        for k in CONTS:
            e = combo[f"{k}_estacion"]
            if isinstance(e, str) and (k, e.split(".")[0]) in tabla.columns:
                niveles.append(tabla[(k, e.split(".")[0])])
        if not niveles:
            continue
        peor = pd.concat(niveles, axis=1).max(axis=1)
        df = pd.DataFrame({"curso": cursos, "nivel": peor}).dropna()
        res = df.groupby("curso")["nivel"].agg(
            dias_con_dato="size",
            dias_ambar=lambda s: int((s == 3).sum()),
            dias_rojo=lambda s: int((s >= 4).sum())).reset_index()
        for k in CONTS:
            res[f"{k}_estacion"] = combo[f"{k}_estacion"]
        filas.append(res)
    por_combo = pd.concat(filas, ignore_index=True)

    salida = c[["codigo"] + cols].merge(por_combo, on=cols, how="left")[
        ["codigo", "curso", "dias_con_dato", "dias_ambar", "dias_rojo"]]
    # Solo cursos completos dentro del periodo de datos
    salida = salida[salida["curso"].isin(["2021-22", "2022-23", "2023-24", "2024-25"])]
    salida.to_csv(CARPETA_SALIDA / "episodios_centros.csv", index=False)

    resumen = salida.groupby("curso")[["dias_ambar", "dias_rojo"]].median()
    print("Mediana de días lectivos por centro y curso:")
    print(resumen.to_string())
    ultimo = salida[salida["curso"] == "2024-25"]
    print(f"\nCurso 2024-25: centros con al menos un día rojo: "
          f"{(ultimo['dias_rojo'] > 0).mean():.0%} · máximo de días rojos: "
          f"{ultimo['dias_rojo'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
