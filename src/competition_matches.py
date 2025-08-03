from pathlib import Path
import json
import pandas as pd
from functools import lru_cache
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def flatten_keys(d, parent_key = "", sep = "_"):
    items: list[str] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_keys(v, new_key, sep))
        else:
            items.append(new_key)
    return items

def has_required_fields(event_file, required):
    try:
        events = json.loads(event_file.read_text(encoding="utf-8"))
        if not isinstance(events, list):
            logger.warning(f"Formato inesperado en {event_file}, se esperaba lista de eventos.")
            return False
        seen: set[str] = set()
        for e in events:
            if not isinstance(e, dict):
                continue
            for key in flatten_keys(e):
                if key in required:
                    seen.add(key)
                    if seen == required:
                        return True
        return False
    except Exception as ex:
        logger.warning(f"Error leyendo {event_file}: {ex}")
        return False


def get_valid_matches(events_folder, required_fields):
    rows: list[dict] = []
    for file in events_folder.rglob("*.json"):
        if has_required_fields(file, required_fields):
            rows.append({
                "match_id": int(file.stem),
                "event_file": str(file)
            })
    df = pd.DataFrame(rows)
    logger.info(f"Encontrados {len(df)} partidos válidos")
    return df


def add_competition(df_valid, matches_root):
    records: list[dict] = []
    for comp_dir in matches_root.iterdir():
        if not comp_dir.is_dir() or not comp_dir.name.isdigit():
            continue
        comp_id = int(comp_dir.name)
        for season_file in comp_dir.glob("*.json"):
            season_id = int(season_file.stem)
            try:
                matches = json.loads(season_file.read_text(encoding="utf-8"))
            except Exception as ex:
                logger.warning(f"Fallo cargando {season_file}: {ex}")
                continue
            for m in matches:
                records.append({
                    "match_id": int(m.get("match_id")),
                    "competition_id": comp_id,
                    "season_id": season_id
                })
    lookup_df = pd.DataFrame(records)
    df_valid = df_valid.astype({"match_id": int})
    enriched = df_valid.merge(lookup_df, on="match_id", how="left")
    logger.info(f"DataFrame enriquecido con {len(enriched)} filas")
    return enriched


if __name__ == "__main__":
    REQUIRED_FIELDS = {
        'id', 'index', 'period', 'minute', 'second', 'possession',
        'play_pattern_id', 'location', 'under_pressure',
        'shot_key_pass_id', 'shot_body_part_id', 'shot_type_id',
        'shot_outcome_id', 'shot_technique_id', 'shot_first_time',
        'shot_aerial_won', 'pass_length', 'pass_angle',
        'pass_height_id', 'pass_cross', 'pass_cut_back',
        'pass_switch', 'pass_body_part_id'
    }
    EVENTS_FOLDER = Path("open-data-master/data/events")
    MATCHES_ROOT = Path("open-data-master/data/matches")
    OUTPUT_CSV = Path("/output/csv/complete_matches.csv")

    # 1. Filtrar partidos válidos
    valid_df = get_valid_matches(EVENTS_FOLDER, REQUIRED_FIELDS)
    # 2. Enriquecer y guardar
    enriched_df = add_competition(valid_df, MATCHES_ROOT)
    enriched_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Partidos completos guardados en {OUTPUT_CSV}")