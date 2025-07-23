from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging
from functools import lru_cache

# ----------------- CONFIGURACIÓN -----------------
# Ruta al fichero JSON que quieres procesar
JSON_FILE = Path("open-data-master/data/events/9880.json")
# Ruta al CSV de salida (opcional)
OUTPUT_CSV = Path("shots_9880.csv")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# -------------------------------------------------

def load_events(json_path: Path) -> pd.DataFrame:
    """
    Carga eventos desde un JSON y devuelve un DataFrame normalizado.
    """
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
        df = pd.json_normalize(data, sep='_')
        logger.info(f"Cargados {len(df)} eventos desde {json_path.name}")
        return df
    except Exception as e:
        logger.error(f"Error cargando eventos {json_path}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=None)
def compute_metrics(location: tuple[float, float]) -> tuple[float, float]:
    """
    Calcula distancia y ángulo a portería para una ubicación dada.
    """
    goal_x, goal_y = 120, 40
    goal_width = 7.32
    try:
        x, y = location
        distance = np.hypot(goal_x - x, goal_y - y)
        a = abs(goal_y - y)
        b = goal_x - x
        left = np.arctan2((goal_width/2) - a, b)
        right = np.arctan2((goal_width/2) + a, b)
        angle = abs(left - right)
        return distance, angle
    except Exception:
        return np.nan, np.nan

def extract_shots(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra tiros y columnas relevantes, calculando distancias y ángulos.
    """
    shot_df = df_events.query("type_name == 'Shot'").copy()
    cols = [
        'id', 'index', 'period', 'minute', 'second', 'possession',
        'play_pattern_id', 'location', 'under_pressure',
        'shot_key_pass_id', 'shot_body_part_id', 'shot_type_id',
        'shot_outcome_id', 'shot_technique_id', 'shot_first_time', 'shot_aerial_won'
    ]
    shot_df = shot_df.loc[:, shot_df.columns.intersection(cols)]
    metrics = shot_df['location'].apply(
        lambda loc: compute_metrics(tuple(loc) if isinstance(loc, list) else (np.nan, np.nan))
    )
    shot_df[['distance_to_goal', 'angle_to_goal']] = pd.DataFrame(
        metrics.tolist(), index=shot_df.index
    )
    shot_df.dropna(axis=1, how='all', inplace=True)
    return shot_df.reset_index(drop=True)

def extract_passes(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra pases y renombra columnas para merge con tiros.
    """
    pass_df = df_events.query("type_name == 'Pass'").copy()
    cols = [
        'id', 'pass_length', 'pass_angle', 'under_pressure',
        'pass_height_id', 'pass_cross', 'pass_cut_back',
        'pass_switch', 'pass_body_part_id'
    ]
    pass_df = pass_df.loc[:, pass_df.columns.intersection(cols)]
    pass_df.rename(
        columns={
            'id': 'shot_key_pass_id',
            **{c: f"key_{c}" for c in cols if c != 'id'}
        }, inplace=True
    )
    return pass_df

def compute_possession_duration(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la duración de cada posesión en segundos.
    """
    if 'timestamp' not in df_events.columns:
        logger.warning("No se encontró columna 'timestamp', se omite posesión")
        return pd.DataFrame()
    ts = df_events['timestamp'].str.split(':', expand=True)
    df_events['timestamp_sec'] = (
        ts[0].astype(int)*3600 + ts[1].astype(int)*60 + ts[2].astype(float)
    )
    duration = (
        df_events.groupby('possession')['timestamp_sec']
        .agg(possession_duration=lambda x: x.max() - x.min())
        .reset_index()
    )
    return duration

def fill_booleans(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Rellena NaN en columnas booleanas con False.
    """
    for col in cols:
        if col in df:
            df[col] = df[col].fillna(False).astype(bool)
    return df

def process_match(json_path: Path, output_csv: Path = None) -> pd.DataFrame:
    """
    Pipeline completo para procesar un partido y generar un DataFrame de tiros.
    """
    df_events = load_events(json_path)
    if df_events.empty:
        return df_events
    df_shots = extract_shots(df_events)
    df_passes = extract_passes(df_events)
    df_duration = compute_possession_duration(df_events)
    df = (
        df_shots
        .merge(df_passes, on='shot_key_pass_id', how='left')
        .merge(df_duration, on='possession', how='left')
    )
    boolean_cols = [
        'under_pressure', 'shot_first_time', 'shot_aerial_won',
        'key_under_pressure', 'key_pass_cross', 'key_pass_cut_back', 'key_pass_switch'
    ]
    df = fill_booleans(df, boolean_cols)
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"Guardado {len(df)} tiros en {output_csv}")
    return df

# --------- EJECUCIÓN DIRECTA ---------
if __name__ == '__main__':
    result = process_match(JSON_FILE, OUTPUT_CSV)
    print(result.head())
