import os
import ast
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import joblib

# --- Constantes ---
GOAL_ID = 97
PENALTY_ID = 88
TRIANGLE_VERTEX_2 = [120.0, 36.0]
TRIANGLE_VERTEX_3 = [120.0, 44.0]
GOAL_CENTER = (120.0, 40.0)
MODEL_PATH = 'models/xg_shot.pkl'
CSV_PATH   = 'output/events.csv'
OUTPUT_CSV = 'output/events_with_xG.csv'

# --- Funciones auxiliares ---
def parse_if_str(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def safe_point_coords(loc):
    try:
        coords = parse_if_str(loc)
        return [float(coords[0]), float(coords[1])]
    except Exception:
        return None

def count_players_in_triangle(freeze_frame, shot_vertex):
    players = parse_if_str(freeze_frame)
    v1 = safe_point_coords(shot_vertex)
    if not isinstance(players, list) or v1 is None:
        return 0
    tri = Polygon([v1, TRIANGLE_VERTEX_2, TRIANGLE_VERTEX_3])
    cnt = 0
    for p in players:
        loc = safe_point_coords(p.get('location'))
        if loc:
            pt = Point(*loc)
            if pt.within(tri) or pt.touches(tri):
                cnt += 1
    return cnt

def keeper_position_score(shot_x, shot_y, keeper_x, keeper_y):
    # Validación básica
    if any(v is None for v in (shot_x, shot_y, keeper_x, keeper_y)):
        return 0.0

    # Construir el triángulo
    v1 = np.array([shot_x, shot_y])
    tri = Polygon([v1, TRIANGLE_VERTEX_2, TRIANGLE_VERTEX_3])
    keeper_pt = Point(keeper_x, keeper_y)
    if not (keeper_pt.within(tri) or keeper_pt.touches(tri)):
        return 0.0

    # Línea central de meta (punto medio entre los postes)
    center_line = np.array([
        (TRIANGLE_VERTEX_2[0] + TRIANGLE_VERTEX_3[0]) / 2.0,
        (TRIANGLE_VERTEX_2[1] + TRIANGLE_VERTEX_3[1]) / 2.0
    ])

    # Vector base y vector portero
    BA = center_line - v1
    PA = np.array([keeper_x, keeper_y]) - v1

    # Distancia perpendicular de portero a la línea de meta
    perp_dist = abs(np.cross(BA, PA)) / np.linalg.norm(BA)
    semi_axis = abs(TRIANGLE_VERTEX_3[1] - TRIANGLE_VERTEX_2[1]) / 2.0

    # Score normalizado e invertido
    score = 1.0 - (perp_dist / semi_axis)
    return float(np.clip(score, 0.0, 1.0))

def euclid(a, b, c, d):
    return np.hypot(c-a, d-b)

def angle_to_goal(x, y):
    a = np.array([TRIANGLE_VERTEX_2[0] - x, TRIANGLE_VERTEX_2[1] - y])
    b = np.array([TRIANGLE_VERTEX_3[0] - x, TRIANGLE_VERTEX_3[1] - y])
    cosang = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    return np.arccos(np.clip(cosang, -1, 1))

# --- Carga y feature engineering ---
df = pd.read_csv(CSV_PATH)

# Target flag (no obligatoria para predicción, pero útil si quieres comparar)
df['is_goal'] = (df.get('shot_outcome_id', -1) == GOAL_ID).astype(int)

# Coords de tiro, pase y portero (si están en tus columnas)
df[['shot_x','shot_y']] = df['shot_location'].apply(lambda s: pd.Series(safe_point_coords(s)))
df[['pass_x','pass_y']] = df['pass_location'].apply(lambda s: pd.Series(safe_point_coords(s)))
df[['gk_x','gk_y']]   = df['gk_location'].apply(lambda s: pd.Series(safe_point_coords(s)))

# Features numéricas
df['distance_to_goal'] = df.apply(lambda r: euclid(r.shot_x, r.shot_y, *GOAL_CENTER), axis=1)
df['angle_to_goal'   ] = df.apply(lambda r: angle_to_goal(r.shot_x, r.shot_y), axis=1)
df['distance_to_goal_squared']   = df['distance_to_goal'] ** 2
df['angle_to_goal_squared']      = df['angle_to_goal'] ** 2
df['distance_angle_interaction'] = df['distance_to_goal'] * df['angle_to_goal']
df.rename(columns={'players_in_range':'players_in_shot_area'}, inplace=True)

# players in area y keeper score
df['keeper_placement_score'] = df.apply(
    lambda r: keeper_position_score(r.shot_x, r.shot_y, r.gk_x, r.gk_y),
    axis=1
)

# key pass length/angle
df['key_pass_length'] = df.apply(
    lambda r: euclid(r.pass_x, r.pass_y, r.shot_x, r.shot_y)
              if pd.notna(r.pass_x) else 0.0,
    axis=1
)
df['key_pass_angle'] = df.apply(
    lambda r: angle_to_goal(r.pass_x, r.pass_y)
              if pd.notna(r.pass_x) else 0.0,
    axis=1
)

# Buckets iguales a los de entrenamiento
df['possession_bucket'] = pd.cut(
    df['possession_duration'], bins=[0,5,15,np.inf],
    labels=['Counterattack','Short Attack','Long Attack']
).astype(str)

df['key_pass_length_bucket'] = pd.cut(
    df['key_pass_length'], bins=[0,10,30,np.inf],
    labels=['Short','Medium','Long']
).astype(str)

df['key_pass_angle_bucket'] = pd.cut(
    df['key_pass_angle'], bins=[-np.pi,-1,1,np.pi],
    labels=['Left','Forward','Right']
).astype(str)

# --- Carga del pipeline y predicción ---
pipeline = joblib.load(MODEL_PATH)

features = [
    'distance_to_goal', 'angle_to_goal', 'possession_duration',
    'key_pass_length', 'key_pass_angle', 'players_in_shot_area',
    'distance_to_goal_squared', 'angle_to_goal_squared',
    'distance_angle_interaction', 'keeper_placement_score',
    'play_pattern_id', 'shot_body_part_id', 'shot_technique_id',
    'shot_type_id', 'key_pass_height_id', 'key_pass_body_part_id',
    'under_pressure', 'shot_first_time', 'shot_aerial_won',
    'key_pass_switch', 'key_under_pressure', 'key_pass_cross',
    'key_pass_cut_back', 'possession_bucket',
    'key_pass_length_bucket', 'key_pass_angle_bucket'
]

X = df[features]
proba = pipeline.predict_proba(X)[:, 1]

# Ajuste de penaltis
mask_pen = X['shot_type_id'] == PENALTY_ID
proba[mask_pen] = 0.75

df['xG_pred'] = proba

# --- Guardar resultado ---
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predicciones guardadas en {OUTPUT_CSV}")
