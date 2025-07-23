import json
import pandas as pd
import numpy as np


def extract_shots_from_match(json_path):
    """Extrae tiros de un partido StatsBomb desde un JSON y devuelve un DataFrame limpio."""
    with open(json_path, 'r', encoding='utf-8') as f:
        events = json.load(f)
    df_events = pd.json_normalize(events, sep='_')

    # Filtrar solo tiros y columnas relevantes
    relevant_columns = [
        'id', 'index', 'period', 'minute', 'second', 'possession',
        'play_pattern_id', 'location', 'under_pressure',
        'shot_key_pass_id', 'shot_body_part_id', 'shot_type_id',
        'shot_outcome_id', 'shot_technique_id', 'shot_first_time', 'shot_aerial_won'
    ]
    df_shots = df_events[df_events['type_name'] == 'Shot'].copy()
    df_shots = df_shots[relevant_columns]
    df_shots.dropna(axis=1, how="all", inplace=True)
    df_shots.reset_index(drop=True, inplace=True)
    return df_shots


def add_distance_and_angle(df):
    """Calcula distancia y ángulo a portería para cada tiro."""
    goal_x, goal_y = 120, 40
    goal_width = 7.32

    def compute_metrics(loc):
        if isinstance(loc, list) and len(loc) == 2:
            x, y = loc
            distance = np.hypot(goal_x - x, goal_y - y)
            a = abs(goal_y - y)
            b = goal_x - x
            left_post = np.arctan2((goal_width / 2) - a, b)
            right_post = np.arctan2((goal_width / 2) + a, b)
            angle = abs(left_post - right_post)
            return pd.Series([distance, angle])
        return pd.Series([np.nan, np.nan])

    df[['distance_to_goal', 'angle_to_goal']] = df['location'].apply(compute_metrics)
    return df


def fill_boolean_na(df):
    """Rellena NaN en columnas booleanas con False."""
    boolean_columns = [
        "under_pressure", "shot_first_time", "shot_aerial_won",
        "key_under_pressure", "key_pass_cross", "key_pass_cut_back", "key_pass_switch"
    ]
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def pass_data(json_path, df):
    """Añade datos del pase previo al DataFrame de tiros."""
    with open(json_path, 'r', encoding='utf-8') as f:
        events = json.load(f)
    df_events = pd.json_normalize(events, sep='_')

    df_pass = df_events[df_events['type_name'] == 'Pass'].copy()
    relevant_columns = [
        'id', 'pass_length', 'pass_angle', 'under_pressure',
        'pass_height_id', 'pass_cross', 'pass_cut_back',
        'pass_switch', 'pass_body_part_id'
    ]
    df_pass = df_pass[relevant_columns]
    df_pass.rename(columns=lambda x: f"key_{x}" if x != "id" else "shot_key_pass_id", inplace=True)

    df = df.merge(df_pass, on='shot_key_pass_id', how='left')
    df['has_key_pass'] = df['shot_key_pass_id'].notnull().astype(bool)
    return df


def add_possession_duration(json_path, df):
    """Añade la duración de la posesión para cada tiro."""
    def timestamp_to_seconds(ts):
        h, m, s = ts.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    with open(json_path, 'r', encoding='utf-8') as f:
        events = json.load(f)
    df_events = pd.json_normalize(events, sep='_')

    df_events['timestamp_sec'] = df_events['timestamp'].apply(timestamp_to_seconds)
    possession_times = df_events.groupby('possession')['timestamp_sec'].agg(['min', 'max'])
    possession_times['possession_duration'] = possession_times['max'] - possession_times['min']

    df = df.merge(
        possession_times[['possession_duration']],
        left_on='possession',
        right_index=True,
        how='left'
    )
    return df


# --------- MAIN EXECUTION ---------
if __name__ == "__main__":
    json_file = "open-data-master/data/events/9880.json"
    shots_df = extract_shots_from_match(json_file)
    shots_df = add_distance_and_angle(shots_df)
    shots_df = pass_data(json_file, shots_df)
    shots_df = add_possession_duration(json_file, shots_df)
    shots_df = fill_boolean_na(shots_df)

    print(shots_df.isna().sum().sort_values(ascending=False))
    shots_df.to_csv("shots_9880.csv", index=False)
