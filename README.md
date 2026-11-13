# RespiraCole

**El aire que respiran los escolares de la Comunidad de Madrid, centro a centro, con datos abiertos.**

RespiraCole cruza las redes públicas de calidad del aire con el directorio de centros educativos para responder a tres preguntas:

1. **¿Cómo está hoy el aire en mi centro y cómo estará mañana?** Un semáforo por centro, en directo, con una previsión propia para el día siguiente.
2. **¿Qué centros respiran peor y a quién afecta más?** Un mapa de exposición y su relación con la renta de cada zona.
3. **¿Qué puedo hacer?** Recomendaciones para el centro, informe anual, avisos y el camino más saludable para ir andando o en bici.

Proyecto presentado a **DATAMAD 2026**, el datathon de datos abiertos de la Comunidad de Madrid.

**Demo:** https://oscargbp.github.io/respiracole/

## Qué hace

| Pestaña | Para quién | Qué muestra |
|---|---|---|
| Hoy y mañana | Familias y centros | Semáforo en directo de cada centro, previsión para mañana, previsión orientativa a 5 días y recomendaciones |
| Camino al cole | Familias (Madrid capital) | Ruta más saludable a pie o en bici frente a la más corta, paso a paso, con puntos críticos |
| Mi centro | Equipos directivos | Ficha del centro, días con aviso por curso, entorno de tráfico y atropellos, informe en PDF |
| Mi red | Titulares y ayuntamientos | Todos los centros de una red o municipio, ordenables y descargables |
| Equidad | Administración | Exposición por nivel de renta y centros prioritarios |
| Asistente | Todos | Preguntas en lenguaje natural sobre los datos (solo dentro de Claude) |
| API | Desarrolladores | Datos en formatos estándar (Smart Data Models, GeoJSON) y servidor MCP |

## Resultados principales

- **3.335 centros escolares** (0-18 años y FP) de 155 municipios, enlazados a las **52 estaciones** de las dos redes de calidad del aire (28 de la Comunidad y 24 del Ayuntamiento de Madrid), con 6,2 millones de horas de datos de 2021 a 2025.
- **Todos los grupos de renta superan la guía de la OMS**: la media de NO₂ en el entorno de los centros es más del doble de la recomendada (10 µg/m³).
- **La exposición no se reparte por igual.** El 76 % de los centros del quintil de renta más baja supera el doble de la guía de NO₂, frente al 54 % del quintil más alto. En PM2,5 la relación con la renta es clara (correlación de Spearman de −0,53 en la Comunidad).
- **La previsión para mañana funciona.** Evaluada en los días lectivos de 2025 (10.692 casos), detecta el 80,4 % de los días en ámbar o rojo, frente al 68,6 % de suponer que mañana será como hoy; para los días rojos, 65,0 % frente a 52,1 %.

## Datos que usa

Todos son datos abiertos. RespiraCole no usa datos personales.

**Comunidad de Madrid** ([datos.comunidad.madrid](https://datos.comunidad.madrid))

- Red de Calidad del Aire: datos horarios históricos, datos del mes y del día en curso (tiempo real), estaciones y datos meteorológicos
- Directorio de centros educativos
- Indicadores de renta por municipio (IRPF)
- Población, densidad y parque de vehículos por municipio

**Ayuntamiento de Madrid** ([datos.madrid.es](https://datos.madrid.es))

- Calidad del aire: datos horarios históricos y tiempo real
- Panel de indicadores de distritos (renta, zonas verdes, densidad)
- Intensidad de tráfico (aforos), accidentes de tráfico, callejero y direcciones
- Cartografía municipal 1:5000 (sendas, pasarelas y zonas ajardinadas) y red ciclista

**Otras fuentes**

- AEMET: predicción meteorológica por municipio
- Copernicus Atmosphere Monitoring Service (CAMS Europe), vía Open-Meteo: previsión de contaminación a 5 días

## Cómo funciona

```
preparar_datos.py     Descarga las dos redes y el directorio; enlaza cada centro con su estación
analizar_equidad.py   Índice de renta por zona y exposición por quintil
capas_contexto.py     Tráfico cercano, vehículos, densidad y zonas verdes de cada centro
predecir.py           Modelo de previsión para mañana y su evaluación
episodios.py          Días en ámbar y rojo de cada centro por curso
camino_escolar.py     Atropellos de peatones en el entorno de cada centro
rutas_red.py          Red peatonal y ciclista de Madrid capital para el camino al cole
exportar_api.py       Datos en formatos estándar (Smart Data Models y GeoJSON)
tiempo_real.py        Cada hora: semáforo de ahora y previsión real para mañana
pronostico.py         Dos veces al día: previsión a 5 días (AEMET y Copernicus)
publicar_web.py       Prepara la web que se publica en docs/
servidor_mcp.py       Servidor MCP para consultar los datos desde un asistente de IA
```

**El semáforo.** Usa las bandas del Índice de Calidad del Aire. Para NO₂ y ozono, el valor horario más alto entre las 9:00 y las 17:00; para PM10 y PM2,5, la media del día. Verde hasta el nivel «razonablemente bueno», ámbar en «regular» y rojo a partir de «desfavorable».

**La previsión.** Un modelo de *gradient boosting* por contaminante, entrenado con los datos de 2021 a 2025. Predice la métrica del semáforo del día siguiente a partir de los valores de hoy y de ayer, la época del año, el día de la semana y la meteorología regional. Se publica cada tarde a las 17:00, al terminar el horario escolar.

**El camino al cole.** La red se reconstruye a partir del callejero y la cartografía municipales. La ruta saludable penaliza las calles con mucho tráfico y los puntos con atropellos, y favorece parques, sendas y vías ciclistas. Se calcula en el navegador; ninguna dirección sale del dispositivo.

## Actualización automática

Dos tareas de GitHub Actions mantienen la web al día:

- `tiempo-real.yml`: cada hora ejecuta `tiempo_real.py` con los últimos datos de las dos redes.
- `pronostico.yml`: dos veces al día ejecuta `pronostico.py`.

La web publicada (carpeta `docs/`) también consulta directamente a los dos portales al abrirse, para mostrar la última hora disponible.

## Ejecutarlo en tu ordenador

```bash
git clone https://github.com/oscargbp/respiracole.git
cd respiracole
pip install -r requirements-vivo.txt

python tiempo_real.py        # semáforo de ahora (unos segundos)
python pronostico.py         # previsión a 5 días (un par de minutos)
python publicar_web.py       # prepara docs/
python -m http.server --directory docs 8000   # abre http://localhost:8000
```

Para regenerar todo desde cero (descarga unos 750 MB de datos originales en `raw/`):

```bash
pip install -r requirements.txt
python preparar_datos.py && python analizar_equidad.py && python capas_contexto.py
python predecir.py && python episodios.py && python camino_escolar.py
python rutas_red.py && python exportar_api.py
python tiempo_real.py --entrenar
```

## Estructura

```
datos/     Resultados del análisis (CSV y JSON) y el modelo entrenado
demo/      La aplicación (una sola página) y los datos que carga
docs/      La web publicada con GitHub Pages
api/       Ejemplos de datos en formatos estándar
extras/    Funciones desarrolladas y retiradas para mantener el foco del proyecto
```

## Límites que conviene conocer

- **El semáforo es del entorno, no del patio.** Cada centro toma el valor de la estación más cercana que mide cada contaminante; la aplicación indica la distancia y lo fiable que es la asignación.
- **Los datos del día son provisionales.** Las redes los validan después.
- **La previsión a 5 días es orientativa.** Copernicus trabaja con celdas de unos 10 km y suaviza los picos de NO₂ junto a calles con tráfico. Para mañana manda el modelo propio.
- **El camino al cole solo cubre Madrid capital**, donde el Ayuntamiento publica callejero, cartografía y accidentes. El tráfico solo se conoce en las calles con punto de aforo.
- **No es un consejo médico.** Las recomendaciones siguen las pautas generales del Índice de Calidad del Aire.

## Licencia

Código bajo licencia [MIT](LICENSE). Los datos pertenecen a sus fuentes y se reutilizan según las condiciones de cada portal, citando su origen.
