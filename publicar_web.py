"""Prepara la carpeta docs/ que sirve GitHub Pages: la demo como página web completa y sus datos."""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEMO, DOCS = BASE / "demo", BASE / "docs"

CABECERA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="RespiraCole: la calidad del aire en el entorno de cada centro escolar de la Comunidad de Madrid, con datos abiertos.">
</head>
<body>
"""

if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(CABECERA + (DEMO / "index.html").read_text(encoding="utf-8") + "\n</body>\n</html>\n", encoding="utf-8")
    for f in DEMO.glob("demo_*.js"):
        shutil.copy2(f, DOCS / f.name)
    if (DEMO / "api").is_dir():
        shutil.copytree(DEMO / "api", DOCS / "api", dirs_exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print("Web preparada en docs/:", sorted(p.name for p in DOCS.iterdir()))
