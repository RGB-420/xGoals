import os
import ast
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import joblib

# --- Constants ---
GOAL_ID = 97
PENALTY_ID = 88
TRIANGLE_VERTEX_2 = [120.0, 36.0]
TRIANGLE_VERTEX_3 = [120.0, 44.0]
GOAL_CENTER = (120.0, 40.0)
MODEL_PATH = 'models/xg_shot.pkl'
CSV_PATH = 'output/events.csv'
OUTPUT_CSV = 'output/events_with_xG.csv'

# --- Helper Functions ---
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
    count = 0
    for p in players:
        loc = safe_point_coords(p.get('location'))
        if loc:
            pt = Point(*loc)
            if pt.within(tri) or pt.touches(tri):
                count += 1
    return count

def keeper_position_score(shot_x, shot_y, keeper_x, keeper_y):
    # Basic validation
    if any(v is None for v in (shot_x, shot_y, keeper_x, keeper_y)):
        return 0.0

    v1 = np.array([shot_x, shot_y])
    tri = Polygon([v1, TRIANGLE_VERTEX_2, TRIANGLE_VERTEX_3])
    keeper_pt = Point(keeper_x, keeper_y)
    if not (keeper_pt.within(tri) or keeper_pt.touches(tri)):
        return 0.0

    # Midpoint of the goal line
    center_line = np.array([
        (TRIANGLE_VERTEX_2[0] + TRIANGLE_VERTEX_3[0]) / 2.0,
        (TRIANGLE_VERTEX_2[1] + TRIANGLE_VERTEX_3[1]) / 2.0
    ])

    BA = center_line - v1
    PA = np.array([keeper_x, keeper_y]) - v1

    perp_dist = abs(np.cross(BA, PA)) / np.linalg.norm(BA)
    semi_axis = abs(TRIANGLE_VERTEX_3[1] - TRIANGLE_VERTEX_2[1]) / 2.0

    # Normalized and inverted score
    score = 1.0 - (perp_dist / semi_axis)
    return float(np.clip(score, 0.0, 1.0))

def euclid(a, b, c, d):
    return np.hypot(c - a, d - b)

def angle_to_goal(x, y):
    a = np.array([TRIANGLE_VERTEX_2[0] - x, TRIANGLE_VERTEX_2[1] - y])
    b = np.array([TRIANGLE_VERTEX_3[0] - x, TRIANGLE_VERTEX_3[1] - y])
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cosang, -1, 1))

# --- Load Data and Feature Engineering ---
df = pd.read_csv(CSV_PATH)

# Optional: add target label for comparison purposes
df['is_goal'] = (df.get('shot_outcome_id', -1) == GOAL_ID).astype(int)

# Extract coordinates
df[['shot_x', 'shot_y']] = df['shot_location'].apply(lambda s: pd.Series(safe_point_coords(s)))
df[['pass_x', 'pass_y']] = df['pass_location'].apply(lambda s: pd.Series(safe_point_coords(s)))
df[['gk_x', 'gk_y']] = df['gk_location'].apply(lambda s: pd.Series(safe_point_coords(s)))

# Create numeric features
df['distance_to_goal'] = df.apply(lambda r: euclid(r.shot_x, r.shot_y, *GOAL_CENTER), axis=1)
df['angle_to_goal'] = df.apply(lambda r: angle_to_goal(r.shot_x, r.shot_y), axis=1)
df['distance_to_goal_squared'] = df['distance_to_goal'] ** 2
df['angle_to_goal_squared'] = df['angle_to_goal'] ** 2
df['distance_angle_interaction'] = df['distance_to_goal'] * df['angle_to_goal']
df.rename(columns={'players_in_range': 'players_in_shot_area'}, inplace=True)

# Add keeper placement score
df['keeper_placement_score'] = df.apply(
    lambda r: keeper_position_score(r.shot_x, r.shot_y, r.gk_x, r.gk_y),
    axis=1
)

# Add key pass length and angle
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

# Bucketing (same as during model training)
df['possession_bucket'] = pd.cut(
    df['possession_duration'], bins=[0, 5, 15, np.inf],
    labels=['Counterattack', 'Short Attack', 'Long Attack']
).astype(str)

df['key_pass_length_bucket'] = pd.cut(
    df['key_pass_length'], bins=[0, 10, 30, np.inf],
    labels=['Short', 'Medium', 'Long']
).astype(str)

df['key_pass_angle_bucket'] = pd.cut(
    df['key_pass_angle'], bins=[-np.pi, -1, 1, np.pi],
    labels=['Left', 'Forward', 'Right']
).astype(str)

# --- Load Model and Make Predictions ---
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

# Override xG for penalties
mask_pen = X['shot_type_id'] == PENALTY_ID
proba[mask_pen] = 0.75

df['xG_pred'] = proba

# --- Save Full Output ---
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")

# --- Save Selected Columns Only ---
SELECTED_COLUMNS = ['match', 'team', 'xG_pred', 'period', 'minute', 'second', 'shot_location', 'is_goal']  # Customize as needed
SUMMARY_CSV = 'output/events_summary.csv'

df[SELECTED_COLUMNS].to_csv(SUMMARY_CSV, index=False)
print(f"Summary saved to {SUMMARY_CSV}")
