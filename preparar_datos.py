"""
RespiraCole · Preparación de datos
==================================

Descarga datos de dos portales de datos abiertos:
  - Comunidad de Madrid (https://datos.comunidad.madrid, CC BY 4.0): centros
    educativos, zonas y Red de Calidad del Aire regional (28 estaciones).
  - Ayuntamiento de Madrid (https://datos.madrid.es): red municipal de calidad
    del aire (24 estaciones), que cubre la ciudad de Madrid.

Y genera, en "datos/", los ficheros que usará la web:

  centros.csv              Centros educativos en alta, con latitud/longitud
  estaciones.csv           Estaciones de la Red de Calidad del Aire
  zonas.csv                Municipios y su zona de calidad del aire
  estaciones_anual.csv     Media anual por estación y contaminante (+ cobertura)
  estaciones_resumen.csv   Media del periodo y días por encima de las guías OMS
  perfil_horario.csv       Media por hora del día (estación y contaminante)
  perfil_mensual.csv       Media por mes (estación y contaminante)
  centros_exposicion.csv   Cada centro con la estación más cercana que mide
                           cada contaminante, la distancia y la exposición estimada
  resumen.txt              Cifras clave para comprobar que todo está bien

Uso:
    pip install requests pandas numpy pyproj
    python preparar_datos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

PORTAL = "https://datos.comunidad.madrid"
API_CKAN = PORTAL + "/api/3/action/package_show"
CABECERAS = {"User-Agent": "Mozilla/5.0 (RespiraCole; DATAMAD 2026)"}

DS_CENTROS = "centros_educativos"
DS_ESTACIONES = "calidad_aire_estaciones"
DS_ZONAS = "calidad_aire_zonas"
DS_HISTORICO = "calidad_aire_datos_historico"

# Años completos que usamos para la exposición media (los 5 últimos)
ANOS = [2021, 2022, 2023, 2024, 2025]

# Códigos de magnitud del histórico de la Comunidad -> nombre corto
CONTAMINANTES = {8: "NO2", 10: "PM10", 9: "PM25", 14: "O3"}

# Valores guía de la OMS (2021) y límites/objetivos vigentes de la UE, en µg/m³
GUIA = {
    #        OMS anual, OMS diario, UE anual
    "NO2":  {"oms_anual": 10, "oms_diario": 25, "ue_anual": 40},
    "PM10": {"oms_anual": 15, "oms_diario": 45, "ue_anual": 40},
    "PM25": {"oms_anual": 5,  "oms_diario": 15, "ue_anual": 25},
    # O3: la referencia es el máximo diario de la media móvil de 8 horas
    "O3":   {"oms_8h": 100, "ue_8h": 120},
}

# Las coordenadas UTM están en el huso 30 Norte, ETRS89 (EPSG:25830)
EPSG_UTM = "EPSG:25830"

# Fiabilidad orientativa según la distancia a la estación (km)
CORTES_FIABILIDAD = [-1, 3, 8, np.inf]
NIVELES_FIABILIDAD = ["alta", "media", "baja"]

# Red del Ayuntamiento de Madrid (https://datos.madrid.es), que cubre la capital
PORTAL_AYTO = "https://datos.madrid.es"
API_CKAN_AYTO = PORTAL_AYTO + "/api/3/action/package_show"
DS_AYTO_ESTACIONES = "212629-0-estaciones-control-aire"
DS_AYTO_HORARIO = "201200-0-calidad-aire-horario"
# datos.madrid.es rechaza a veces peticiones seguidas: usamos un navegador
# "normal" como User-Agent y reintentamos con pausas crecientes
CABECERAS_AYTO = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

# Tipos de centro que entran en RespiraCole (alumnado de 0 a 18 años y FP),
# agrupados por etapa. El directorio incluye también escuelas de música, de
# idiomas y de adultos, equipos de orientación, oficinas, conservatorios, etc.,
# que se excluyen.
CATEGORIAS = {
    "Escuela infantil (0-3)": ["EIPR", "EEI", "EEI-CN", "CPR INF"],
    "Infantil y Primaria": ["CP INF-PRI", "CPR INF-PRI", "CP PRI", "CPR PRI"],
    "Infantil a Secundaria": ["CP INF-PRI-SEC", "CPR INF-PRI-SEC", "CPR PRI-SEC",
                              "CPR INF-SEC", "CIMELPR PRSEC"],
    "Secundaria": ["IES", "CPR ES", "CIMPR SEC"],
    "Formación Profesional": ["CPR FPE", "CP FPE", "CP IFP"],
    "Educación especial": ["CP EE", "CPR EE", "CET"],
    "Centro extranjero": ["CEXTRAN ESP"],
}
TIPO_A_CATEGORIA = {tipo: cat for cat, tipos in CATEGORIAS.items() for tipo in tipos}

BASE = Path(__file__).resolve().parent
CARPETA_RAW = BASE / "raw"     # descargas originales (caché)
CARPETA_SALIDA = BASE / "datos"


# ---------------------------------------------------------------------------
# Descargas
# ---------------------------------------------------------------------------

def recursos(id_dataset: str) -> list[dict]:
    r = requests.get(API_CKAN, params={"id": id_dataset}, headers=CABECERAS, timeout=60)
    r.raise_for_status()
    return r.json()["result"]["resources"]


def url_csv(id_dataset: str, nombre: str | None = None) -> str:
    for rec in recursos(id_dataset):
        if (rec.get("format") or "").upper() != "CSV":
            continue
        if nombre is None or rec.get("name") == nombre:
            return rec["url"]
    raise RuntimeError(f"No hay CSV '{nombre or ''}' en el dataset {id_dataset}")


def descargar(url: str, destino: Path) -> Path:
    """Descarga un fichero salvo que ya esté en la caché."""
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    print(f"  · descargando {destino.name}")
    with requests.get(url, headers=CABECERAS, timeout=300, stream=True) as r:
        r.raise_for_status()
        tmp = destino.with_suffix(".part")
        with open(tmp, "wb") as f:
            for trozo in r.iter_content(1 << 20):
                f.write(trozo)
        tmp.rename(destino)
    return destino


def pedir_con_reintentos(url: str, params: dict | None = None, intentos: int = 5,
                         timeout: int = 300) -> requests.Response:
    import time

    for i in range(intentos):
        r = requests.get(url, params=params, headers=CABECERAS_AYTO, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 0:
            return r
        espera = 10 * (i + 1)
        print(f"    (el servidor respondió {r.status_code}; reintento en {espera} s)")
        time.sleep(espera)
    r.raise_for_status()
    return r


def recursos_ayto(id_dataset: str) -> list[dict]:
    r = pedir_con_reintentos(API_CKAN_AYTO, params={"id": id_dataset}, timeout=60)
    return r.json()["result"]["resources"]


def descargar_ayto(url: str, destino: Path) -> Path:
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    print(f"  · descargando {destino.name}")
    r = pedir_con_reintentos(url)
    destino.write_bytes(r.content)
    return destino


def descargar_red_ayuntamiento() -> tuple[Path, list[Path]]:
    """Descarga las estaciones y el histórico horario de la red municipal.

    Devuelve la ruta del CSV de estaciones y la lista de CSV mensuales/anuales
    del histórico (ya descomprimidos).
    """
    import time
    import zipfile

    carpeta = CARPETA_RAW / "ayuntamiento"
    carpeta.mkdir(parents=True, exist_ok=True)

    est = next(r["url"] for r in recursos_ayto(DS_AYTO_ESTACIONES)
               if (r.get("format") or "").upper() == "CSV")
    f_est = descargar_ayto(est, carpeta / "estaciones.csv")
    time.sleep(2)

    ficheros: list[Path] = []
    recs = recursos_ayto(DS_AYTO_HORARIO)
    for ano in ANOS:
        # Hasta 2024: un ZIP por año. Desde 2025: un único CSV acumulado.
        zip_rec = next((r for r in recs if (r.get("format") or "").upper() == "ZIP"
                        and (r.get("description") or "").strip().endswith(str(ano))), None)
        if zip_rec:
            f_zip = descargar_ayto(zip_rec["url"], carpeta / f"horario_{ano}.zip")
            destino = carpeta / f"horario_{ano}"
            if not destino.exists():
                with zipfile.ZipFile(f_zip) as z:
                    z.extractall(destino)
            ficheros += sorted(destino.rglob("*.csv"))
            time.sleep(2)
    acumulado = next((r for r in recs if (r.get("format") or "").upper() == "CSV"
                      and "desde" in (r.get("description") or "").lower()), None)
    if acumulado and any(a >= 2025 for a in ANOS):
        ficheros.append(descargar_ayto(acumulado["url"], carpeta / "horario_desde_2025.csv"))
    return f_est, ficheros


def leer_csv(ruta: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(ruta, sep=";", encoding="utf-8-sig", dtype=str, **kw)


def a_numero(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


# ---------------------------------------------------------------------------
# Centros, estaciones y zonas
# ---------------------------------------------------------------------------

def utm_a_latlon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    t = Transformer.from_crs(EPSG_UTM, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return np.round(lat, 6), np.round(lon, 6)


def preparar_centros(ruta: Path) -> tuple[pd.DataFrame, dict]:
    df = leer_csv(ruta)
    total = len(df)
    col_sit = next(c for c in df.columns if c.upper().startswith("SITUACI"))
    df = df[df[col_sit].str.strip() == "ALTA"].copy()
    en_alta = len(df)

    # Solo centros con alumnado de 0 a 18 años y FP (ver CATEGORIAS)
    df["categoria"] = df["TIPO_ABRV"].str.strip().map(TIPO_A_CATEGORIA)
    excluidos = int(df["categoria"].isna().sum())
    df = df[df["categoria"].notna()].copy()
    escolares = len(df)

    df["utm_x"], df["utm_y"] = a_numero(df["UTM_X"]), a_numero(df["UTM_Y"])
    dentro = df["utm_x"].between(350_000, 500_000) & df["utm_y"].between(4_400_000, 4_580_000)
    df = df[dentro].copy()
    df["lat"], df["lon"] = utm_a_latlon(df["utm_x"].to_numpy(), df["utm_y"].to_numpy())

    salida = pd.DataFrame({
        "codigo": df["CODIGO"].str.strip(),
        "nombre": df["CENTRO"].str.strip(),
        "tipo": df["TIPO_ABRV"].str.strip(),
        "tipo_descripcion": df["TIPO_EXT"].str.strip(),
        "categoria": df["categoria"],
        "titularidad": df["TITULARIDAD"].str.strip(),
        "area_territorial": df["DAT"].str.strip(),
        "domicilio": (df["CDTPVIA"].fillna("") + " " + df["DOMICILIO"].fillna("") + " "
                      + df["NMVIAL"].fillna("")).str.strip(),
        "codigo_postal": df["CDPOSTAL"].str.strip(),
        "municipio": df["MUNICIPIO"].str.strip(),
        "distrito": df["DISTRITO"].fillna("").str.strip(),
        "web": df["WEB"].fillna("").str.strip(),
        "utm_x": df["utm_x"].round(0).astype(int),
        "utm_y": df["utm_y"].round(0).astype(int),
        "lat": df["lat"],
        "lon": df["lon"],
    }).sort_values(["municipio", "nombre"]).reset_index(drop=True)

    cifras = {"directorio": total, "en_alta": en_alta, "excluidos": excluidos,
              "escolares": escolares, "sin_coordenadas": escolares - len(salida)}
    return salida, cifras


def preparar_estaciones(ruta: Path) -> pd.DataFrame:
    df = leer_csv(ruta)
    analiz = {
        "NO2": "estacion_analizador_NO2",
        "PM10": "estacion_analizador_PM10",
        "PM25": "estacion_analizador_PM2_5",
        "O3": "estacion_analizador_O3",
    }
    salida = pd.DataFrame({
        "codigo": df["estacion_codigo"].str.strip(),
        "municipio": df["estacion_municipio"].str.strip(),
        "zona": df["zona_calidad_aire_descripcion"].str.strip(),
        "tipo_area": df["estacion_tipo_area"].str.strip(),
        "tipo_estacion": df["estacion_tipo_estacion"].str.strip(),
        "direccion": df["estacion_direccion_postal"].str.strip(),
        "utm_x": a_numero(df["estacion_coord_UTM_ETRS89_x"]),
        "utm_y": a_numero(df["estacion_coord_UTM_ETRS89_y"]),
        "lat": a_numero(df["estacion_coord_latitud"]),
        "lon": a_numero(df["estacion_coord_longitud"]),
        "altitud": a_numero(df["estacion_altitud"]),
    })
    for nombre, col in analiz.items():
        salida[f"mide_{nombre}"] = df[col].str.strip().str.lower().isin(["si", "sí"])
    return salida.sort_values("municipio").reset_index(drop=True)


def preparar_estaciones_ayto(ruta: Path) -> pd.DataFrame:
    """Estaciones de la red del Ayuntamiento, con el mismo esquema que las de la Comunidad."""
    df = leer_csv(ruta)
    df.columns = [c.strip() for c in df.columns]
    marca = lambda col: df[col].fillna("").str.strip().str.upper() == "X"
    salida = pd.DataFrame({
        "codigo": df["CODIGO"].str.strip(),
        "municipio": "Madrid",
        "zona": "MADRID",
        "tipo_area": "Urbana",
        "tipo_estacion": df["NOM_TIPO"].str.strip(),
        "direccion": (df["ESTACION"].str.strip() + " (" + df["DIRECCION"].str.strip() + ")"),
        "utm_x": a_numero(df["COORDENADA_X_ETRS89"]),
        "utm_y": a_numero(df["COORDENADA_Y_ETRS89"]),
        "lat": a_numero(df["LATITUD"]),
        "lon": a_numero(df["LONGITUD"]),
        "altitud": a_numero(df["ALTITUD"]),
        "mide_NO2": marca("NO2"),
        "mide_PM10": marca("PM10"),
        "mide_PM25": marca("PM2_5"),
        "mide_O3": marca("O3"),
    })
    return salida


def preparar_zonas(ruta: Path) -> pd.DataFrame:
    df = leer_csv(ruta)
    return pd.DataFrame({
        "zona_codigo": df["zona_calidad_aire_codigo"].str.strip(),
        "zona": df["zona_calidad_aire_descripcion"].str.strip(),
        "municipio": df["zona_calidad_aire_municipio"].str.strip(),
    })


# ---------------------------------------------------------------------------
# Histórico horario
# ---------------------------------------------------------------------------

def leer_historico(ruta: Path) -> pd.DataFrame:
    """Lee un año del histórico y lo pasa a formato largo: una fila por hora.

    Solo conserva NO2, PM10, PM2.5 y O3, y solo los valores validados ("V").
    """
    df = leer_csv(ruta)
    # La Comunidad usa minúsculas y el Ayuntamiento mayúsculas: unificamos
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    df = df[pd.to_numeric(df["ano"], errors="coerce").isin(ANOS)]
    df["magnitud"] = pd.to_numeric(df["magnitud"], errors="coerce")
    df = df[df["magnitud"].isin(CONTAMINANTES)].copy()
    df["estacion_codigo"] = (df["provincia"].str.zfill(2) + df["municipio"].str.zfill(3)
                             + df["estacion"].str.zfill(3))
    df["fecha"] = pd.to_datetime(
        df["ano"].str.strip() + "-" + df["mes"].str.zfill(2) + "-" + df["dia"].str.zfill(2),
        errors="coerce",
    )

    trozos = []
    for h in range(1, 25):
        t = df[["estacion_codigo", "magnitud", "fecha"]].copy()
        t["hora"] = h - 1  # h01 = de 00:00 a 01:00
        t["valor"] = a_numero(df[f"h{h:02d}"])
        t["valido"] = df[f"v{h:02d}"].str.strip() == "V"
        trozos.append(t)
    largo = pd.concat(trozos, ignore_index=True)
    largo = largo[largo["valido"] & largo["valor"].notna() & largo["fecha"].notna()]
    largo["contaminante"] = largo["magnitud"].map(CONTAMINANTES)
    return largo[["estacion_codigo", "contaminante", "fecha", "hora", "valor"]]


def indicadores(horario: pd.DataFrame) -> dict[str, pd.DataFrame]:
    h = horario.copy()
    h["ano"] = h["fecha"].dt.year
    h["mes"] = h["fecha"].dt.month

    # Medias diarias (NO2, PM10, PM2.5). Exigimos al menos 18 horas válidas.
    diario = (h[h["contaminante"] != "O3"]
              .groupby(["estacion_codigo", "contaminante", "fecha"])["valor"]
              .agg(["mean", "count"]).reset_index())
    diario = diario[diario["count"] >= 18].rename(columns={"mean": "valor"})

    # O3: máximo diario de la media móvil de 8 horas
    o3 = h[h["contaminante"] == "O3"].sort_values(["estacion_codigo", "fecha", "hora"])
    o3 = o3.set_index(pd.to_datetime(o3["fecha"]) + pd.to_timedelta(o3["hora"], unit="h"))
    movil = (o3.groupby("estacion_codigo")["valor"]
             .rolling("8h", min_periods=6).mean().reset_index())
    movil.columns = ["estacion_codigo", "momento", "valor"]
    movil["fecha"] = movil["momento"].dt.normalize()
    o3_diario = (movil.groupby(["estacion_codigo", "fecha"])["valor"].max().reset_index())
    o3_diario["contaminante"] = "O3"

    # --- Media anual por estación y contaminante, con cobertura de datos ---
    anual = (h.groupby(["estacion_codigo", "contaminante", "ano"])["valor"]
             .agg(media="mean", horas="count").reset_index())
    horas_ano = np.where(pd.Series(anual["ano"]).map(lambda a: pd.Timestamp(a, 12, 31).dayofyear) == 366,
                         8784, 8760)
    anual["cobertura"] = (anual["horas"] / horas_ano).round(3)
    anual["media"] = anual["media"].round(1)

    # --- Días al año por encima de la guía diaria de la OMS ---
    dias = []
    for cont in ("NO2", "PM10", "PM25"):
        d = diario[diario["contaminante"] == cont].copy()
        d["supera"] = d["valor"] > GUIA[cont]["oms_diario"]
        dias.append(d)
    d = o3_diario.copy()
    d["supera"] = d["valor"] > GUIA["O3"]["oms_8h"]
    dias.append(d)
    dias = pd.concat(dias, ignore_index=True)
    dias["ano"] = pd.to_datetime(dias["fecha"]).dt.year
    dias_ano = (dias.groupby(["estacion_codigo", "contaminante", "ano"])
                .agg(dias_con_dato=("supera", "size"), dias_supera_oms=("supera", "sum"))
                .reset_index())
    anual = anual.merge(dias_ano, on=["estacion_codigo", "contaminante", "ano"], how="left")

    # --- Resumen del periodo: solo años con cobertura suficiente (>= 75 %) ---
    buenos = anual[anual["cobertura"] >= 0.75]
    resumen = (buenos.groupby(["estacion_codigo", "contaminante"])
               .agg(media_periodo=("media", "mean"),
                    anos_validos=("ano", "nunique"),
                    dias_supera_oms_ano=("dias_supera_oms", "mean"))
               .reset_index())
    resumen["media_periodo"] = resumen["media_periodo"].round(1)
    resumen["dias_supera_oms_ano"] = resumen["dias_supera_oms_ano"].round(0)

    def ref(cont: str, clave: str):
        return GUIA[cont].get(clave, np.nan)

    resumen["oms_anual"] = resumen["contaminante"].map(lambda c: ref(c, "oms_anual"))
    resumen["ue_anual"] = resumen["contaminante"].map(lambda c: ref(c, "ue_anual"))
    resumen["veces_guia_oms"] = (resumen["media_periodo"] / resumen["oms_anual"]).round(1)

    # --- Perfiles horario y mensual (para "Mi centro") ---
    perfil_h = (h.groupby(["estacion_codigo", "contaminante", "hora"])["valor"]
                .mean().round(1).reset_index(name="media"))
    perfil_m = (h.groupby(["estacion_codigo", "contaminante", "mes"])["valor"]
                .mean().round(1).reset_index(name="media"))

    return {"anual": anual, "resumen": resumen, "perfil_h": perfil_h, "perfil_m": perfil_m}


# ---------------------------------------------------------------------------
# Exposición estimada de cada centro
# ---------------------------------------------------------------------------

def normalizar_municipio(nombre: str) -> str:
    """'Escorial, El' / 'EL ESCORIAL' / 'Rivas-Vaciamadrid' -> 'EL ESCORIAL' / 'RIVAS VACIAMADRID'."""
    import unicodedata

    s = str(nombre).strip()
    if ", " in s:  # artículo pospuesto: "Rozas de Madrid, Las" -> "Las Rozas de Madrid"
        base, articulo = s.rsplit(", ", 1)
        s = f"{articulo} {base}"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.upper().replace("-", " ").split())


def exposicion_centros(centros: pd.DataFrame, estaciones: pd.DataFrame,
                       resumen: pd.DataFrame, zonas: pd.DataFrame) -> pd.DataFrame:
    """Para cada centro y contaminante, toma la estación más cercana que lo mide."""
    salida = centros[["codigo", "nombre", "tipo", "categoria", "titularidad", "municipio", "distrito",
                      "area_territorial", "lat", "lon"]].copy()

    # Zona de calidad del aire por municipio (los nombres se escriben distinto
    # en cada dataset, así que los normalizamos antes de cruzarlos)
    zonas_map = dict(zip(zonas["municipio"].map(normalizar_municipio), zonas["zona"]))
    sin_articulo = {k.split(" ", 1)[1]: v for k, v in zonas_map.items()
                    if k.split(" ", 1)[0] in ("EL", "LA", "LOS", "LAS") and " " in k}

    def zona_de(muni: str):
        n = normalizar_municipio(muni)
        if n in zonas_map:
            return zonas_map[n]
        partes = n.split(" ", 1)
        if partes[0] in ("EL", "LA", "LOS", "LAS") and len(partes) == 2:
            return zonas_map.get(partes[1])
        return sin_articulo.get(n)

    salida["zona_calidad_aire"] = centros["municipio"].map(zona_de)

    cx, cy = centros["utm_x"].to_numpy(float), centros["utm_y"].to_numpy(float)
    for cont in CONTAMINANTES.values():
        con_dato = resumen[resumen["contaminante"] == cont][["estacion_codigo", "media_periodo",
                                                            "dias_supera_oms_ano"]]
        est = estaciones.merge(con_dato, left_on="codigo", right_on="estacion_codigo")
        if est.empty:
            continue
        ex, ey = est["utm_x"].to_numpy(float), est["utm_y"].to_numpy(float)
        dist = np.hypot(cx[:, None] - ex[None, :], cy[:, None] - ey[None, :]) / 1000
        idx = dist.argmin(axis=1)
        salida[f"{cont}_estacion"] = est["codigo"].to_numpy()[idx]
        salida[f"{cont}_estacion_municipio"] = est["municipio"].to_numpy()[idx]
        salida[f"{cont}_distancia_km"] = dist[np.arange(len(idx)), idx].round(2)
        salida[f"{cont}_media"] = est["media_periodo"].to_numpy()[idx]
        salida[f"{cont}_dias_supera_oms"] = est["dias_supera_oms_ano"].to_numpy()[idx]

    # Fiabilidad según la distancia a la estación de NO2 (la red más densa)
    salida["fiabilidad"] = pd.cut(salida["NO2_distancia_km"], CORTES_FIABILIDAD,
                                  labels=NIVELES_FIABILIDAD).astype(str)

    # Red a la que pertenece la estación de NO2 asignada (28079xxx = Ayuntamiento)
    salida["red"] = np.where(salida["NO2_estacion"].astype(str).str.startswith("28079"),
                             "ayuntamiento", "comunidad")
    return salida


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main() -> int:
    CARPETA_RAW.mkdir(exist_ok=True)
    CARPETA_SALIDA.mkdir(exist_ok=True)

    print("1/4 Descargando datos de la Comunidad de Madrid…")
    f_centros = descargar(url_csv(DS_CENTROS), CARPETA_RAW / "centros_educativos.csv")
    f_est = descargar(url_csv(DS_ESTACIONES), CARPETA_RAW / "calidad_aire_estaciones.csv")
    f_zonas = descargar(url_csv(DS_ZONAS), CARPETA_RAW / "calidad_aire_zonas.csv")
    f_hist = [descargar(url_csv(DS_HISTORICO, str(a)), CARPETA_RAW / f"historico_{a}.csv")
              for a in ANOS]

    print("   … y de la red del Ayuntamiento de Madrid (capital)")
    f_est_ayto, f_hist_ayto = descargar_red_ayuntamiento()

    print("2/4 Preparando centros, estaciones y zonas…")
    centros, cifras = preparar_centros(f_centros)
    est_cm = preparar_estaciones(f_est).assign(red="comunidad")
    est_ay = preparar_estaciones_ayto(f_est_ayto).assign(red="ayuntamiento")
    estaciones = pd.concat([est_cm, est_ay], ignore_index=True)
    zonas = preparar_zonas(f_zonas)

    print(f"3/4 Procesando el histórico horario {ANOS[0]}-{ANOS[-1]} de las dos redes…")
    horario = pd.concat([leer_historico(f) for f in f_hist + f_hist_ayto], ignore_index=True)
    # El fichero acumulado del Ayuntamiento puede solaparse con los anuales
    horario = horario.drop_duplicates(["estacion_codigo", "contaminante", "fecha", "hora"])
    ind = indicadores(horario)

    print("4/4 Estimando la exposición de cada centro…")
    expo = exposicion_centros(centros, estaciones, ind["resumen"], zonas)

    centros.to_csv(CARPETA_SALIDA / "centros.csv", index=False)
    estaciones.to_csv(CARPETA_SALIDA / "estaciones.csv", index=False)
    zonas.to_csv(CARPETA_SALIDA / "zonas.csv", index=False)
    ind["anual"].to_csv(CARPETA_SALIDA / "estaciones_anual.csv", index=False)
    ind["resumen"].to_csv(CARPETA_SALIDA / "estaciones_resumen.csv", index=False)
    ind["perfil_h"].to_csv(CARPETA_SALIDA / "perfil_horario.csv", index=False)
    ind["perfil_m"].to_csv(CARPETA_SALIDA / "perfil_mensual.csv", index=False)
    expo.to_csv(CARPETA_SALIDA / "centros_exposicion.csv", index=False)

    texto = construir_resumen(cifras, centros, estaciones, ind, expo, len(horario))
    (CARPETA_SALIDA / "resumen.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    return 0


def construir_resumen(cifras, centros, estaciones, ind, expo, n_horas) -> str:
    L = ["RESPIRACOLE · RESUMEN DE LA PREPARACIÓN DE DATOS", "=" * 50, "",
         "CENTROS EDUCATIVOS",
         f"  En el directorio:                 {cifras['directorio']:>7,}",
         f"  En alta:                          {cifras['en_alta']:>7,}",
         f"  Excluidos (música, idiomas, adultos, orientación, oficinas…): {cifras['excluidos']:,}",
         f"  Centros escolares (0-18 años y FP): {cifras['escolares']:>5,}",
         f"  … con coordenadas válidas:        {len(centros):>7,}",
         f"  Municipios:                       {centros['municipio'].nunique():>7,}",
         "  Por etapa:"]
    for cat, n in centros["categoria"].value_counts().items():
        L.append(f"    {cat:<32}{n:>7,}")
    L.append("  Por titularidad:")
    for tit, n in centros["titularidad"].value_counts().items():
        L.append(f"    {tit:<32}{n:>7,}")
    n_cm = int((estaciones["red"] == "comunidad").sum())
    n_ay = int((estaciones["red"] == "ayuntamiento").sum())
    L += ["",
          "REDES DE CALIDAD DEL AIRE",
          f"  Estaciones Comunidad / Ayuntamiento: {n_cm} / {n_ay}",
          f"  Horas de datos válidos ({ANOS[0]}-{ANOS[-1]}): {n_horas:>10,}",
          "  Centros según la red de su estación de NO2 más cercana:"]
    for red, n in expo["red"].value_counts().items():
        L.append(f"    {red:<14}{n:>7,}")
    L.append("  Fiabilidad (distancia a la estación de NO2 más cercana):")
    for nivel in NIVELES_FIABILIDAD:
        n = int((expo["fiabilidad"] == nivel).sum())
        L.append(f"    {nivel:<6}{n:>7,}  ({n / len(expo):.0%})")
    L.append("    (alta < 3 km · media 3-8 km · baja > 8 km)")

    r = ind["resumen"].merge(estaciones[["codigo", "red"]], left_on="estacion_codigo",
                             right_on="codigo")
    L += ["", f"MEDIAS DEL PERIODO {ANOS[0]}-{ANOS[-1]} EN LAS ESTACIONES (µg/m³)",
          f"  {'':14}{'mín':>7}{'mediana':>9}{'máx':>7}{'guía OMS':>10}{'límite UE':>11}"]
    for cont in ("NO2", "PM10", "PM25"):
        for red in ("comunidad", "ayuntamiento"):
            s = r[(r["contaminante"] == cont) & (r["red"] == red)]["media_periodo"]
            if s.empty:
                continue
            etiqueta = f"{cont} {red[:5]}."
            L.append(f"  {etiqueta:<14}{s.min():>7.1f}{s.median():>9.1f}{s.max():>7.1f}"
                     f"{GUIA[cont]['oms_anual']:>10}{GUIA[cont]['ue_anual']:>11}")
    s = r[r["contaminante"] == "O3"]["dias_supera_oms_ano"]
    if not s.empty:
        L.append(f"  O3: días/año con máx. 8h > {GUIA['O3']['oms_8h']} µg/m³ (OMS): "
                 f"mediana {s.median():.0f}, máx. {s.max():.0f}")

    L += ["", "CENTROS POR ENCIMA DE LA GUÍA ANUAL DE LA OMS / DEL LÍMITE DE LA UE"]
    for cont in ("NO2", "PM10", "PM25"):
        col = f"{cont}_media"
        if col in expo:
            oms = int((expo[col] > GUIA[cont]["oms_anual"]).sum())
            ue = int((expo[col] > GUIA[cont]["ue_anual"]).sum())
            L.append(f"  {cont:<6} OMS {oms:>6,} ({oms / len(expo):.0%})   UE {ue:>5,}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.RequestException as e:
        print(f"\nNo he podido conectar con el portal de datos: {e}")
        sys.exit(1)
