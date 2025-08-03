#!/usr/bin/env python3
"""
Genera un archivo Excel con mappings id->name desde todos los JSON de un directorio.
Cada hoja en el Excel corresponderá a un key (por ejemplo: "player", "team", etc.).
"""
from pathlib import Path
import json
import pandas as pd
import logging

# ------ CONFIGURACIÓN ------
# Carpeta con archivos JSON de eventos
EVENTS_FOLDER = Path("open-data-master/data/events")
# Ruta de salida del Excel
OUTPUT_FILE = Path("/utils/id_name_mappings.xlsx")
# Nivel de logging (INFO o DEBUG)
LOG_LEVEL = logging.INFO
# ---------------------------

# Configurar logger
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def collect_id_name_mappings(events_folder: Path) -> dict[str, pd.DataFrame]:
    """
    Recorre todos los JSON en `events_folder` y extrae mappings id->name de columnas terminadas en '_id' y '_name'.
    Retorna un diccionario donde la clave es el nombre base (sin sufijo) y el valor un DataFrame con columnas [<base>_id, <base>_name].
    """
    mappings: dict[str, pd.DataFrame] = {}
    json_files = list(events_folder.rglob("*.json"))
    logger.info("Buscando JSON en %s (%d archivos)" % (events_folder, len(json_files)))

    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            df = pd.json_normalize(data, sep='_')
        except Exception as e:
            logger.warning(f"No se pudo leer {path.name}: {e}")
            continue

        # Encontrar columnas id->name
        for col in [c for c in df.columns if c.endswith('_id')]:
            base = col[:-3]  # quita '_id'
            name_col = f"{base}_name"
            if name_col not in df.columns:
                continue

            # Extraer y limpiar duplicates
            tmp = df[[col, name_col]].dropna().drop_duplicates().sort_values(col)
            if tmp.empty:
                continue

            # Agregar al mapping global
            if base in mappings:
                mappings[base] = pd.concat([mappings[base], tmp], ignore_index=True)
                mappings[base] = mappings[base].drop_duplicates().sort_values(col)
            else:
                mappings[base] = tmp.copy()

    logger.info("Total de keys mapeadas: %d" % len(mappings))
    return mappings


def export_to_excel(mappings: dict[str, pd.DataFrame], output_file: Path) -> None:
    """
    Escribe cada DataFrame de mappings en hojas separadas dentro de un mismo archivo Excel.
    """
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for sheet, df in mappings.items():
            # Limitar nombre de hoja a 31 caracteres
            sheet_name = sheet[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Hoja '{sheet_name}' con {len(df)} filas exportada.")
    logger.info(f"Mappings exportados a {output_file}")


if __name__ == '__main__':
    # 1. Recoger mappings
    mappings = collect_id_name_mappings(EVENTS_FOLDER)
    # 2. Exportar a Excel
    if mappings:
        export_to_excel(mappings, OUTPUT_FILE)
    else:
        logger.warning("No se encontraron mappings para exportar.")
