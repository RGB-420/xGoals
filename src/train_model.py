import os
import ast
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

# Constants for shot outcome and spatial triangle
GOAL_ID = 97
PENALTY_ID = 88
TRIANGLE_VERTEX_2 = [120.0, 36.0]
TRIANGLE_VERTEX_3 = [120.0, 44.0]
# Directory and filename for saving the trained pipeline
MODEL_DIR = 'models'
MODEL_FILE = 'xg_shot.pkl'


def parse_if_str(x):
    """Convert string representations of Python/JSON objects to native Python types."""
    return ast.literal_eval(x) if isinstance(x, str) else x


def safe_point_coords(loc):
    """
    Ensure the location input is a two-element list of floats.
    Return None if conversion fails.
    """
    try:
        coords = parse_if_str(loc)
        return [float(coords[0]), float(coords[1])]
    except Exception:
        return None


def count_players_in_triangle(freeze_frame, shot_vertex):
    """
    Count players whose 'location' lies inside or on the triangle
    defined by the shot vertex and the two goalposts.
    """
    players = parse_if_str(freeze_frame)
    v1 = safe_point_coords(shot_vertex)
    if not isinstance(players, list) or v1 is None:
        return 0

    triangle = Polygon([v1, TRIANGLE_VERTEX_2, TRIANGLE_VERTEX_3])
    count = 0
    for player in players:
        loc = safe_point_coords(player.get('location'))
        if loc:
            pt = Point(*loc)
            if pt.within(triangle) or pt.touches(triangle):
                count += 1
    return count


def keeper_position_score(freeze_frame, shot_vertex):
    """
    Calculate a normalized score [0,1] indicating how centrally
    the away goalkeeper stands in the shot triangle.
    """
    players = parse_if_str(freeze_frame)
    v1 = safe_point_coords(shot_vertex)
    if not isinstance(players, list) or v1 is None:
        return 0.0

    # Identify the away goalkeeper
    keeper = next(
        (p for p in players
         if p.get('position', {}).get('name') == 'Goalkeeper' and not p.get('teammate', True)),
        None
    )
    if not keeper:
        return 0.0

    # Check if keeper is inside or on the triangle
    triangle = Polygon([v1, TRIANGLE_VERTEX_2, TRIANGLE_VERTEX_3])
    keeper_coords = safe_point_coords(keeper.get('location'))
    if keeper_coords is None or not (Point(*keeper_coords).within(triangle) or Point(*keeper_coords).touches(triangle)):
        return 0.0

    # Compute perpendicular distance from keeper to goal line center
    center_line = np.array([(TRIANGLE_VERTEX_2[0] + TRIANGLE_VERTEX_3[0]) / 2.0,
                             (TRIANGLE_VERTEX_2[1] + TRIANGLE_VERTEX_3[1]) / 2.0])
    BA = center_line - np.array(v1)
    PA = np.array(keeper_coords) - np.array(v1)
    cross_prod = abs(np.cross(BA, PA))
    perp_distance = cross_prod / np.linalg.norm(BA)
    semi_axis = (TRIANGLE_VERTEX_3[1] - TRIANGLE_VERTEX_2[1]) / 2.0

    score = 1.0 - (perp_distance / semi_axis)
    return float(np.clip(score, 0.0, 1.0))


def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load CSV, engineer features, and compute the target variable."""
    df = pd.read_csv(path)
    df['is_goal'] = (df['shot_outcome_id'] == GOAL_ID).astype(int)

    # Create squared and interaction features
    df['distance_to_goal_squared'] = df['distance_to_goal'] ** 2
    df['angle_to_goal_squared'] = df['angle_to_goal'] ** 2
    df['distance_angle_interaction'] = df['distance_to_goal'] * df['angle_to_goal']

    # Create categorical buckets
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

    # Compute spatial features
    df['players_in_shot_area'] = df.apply(
        lambda row: count_players_in_triangle(row['shot_freeze_frame'], row['location']), axis=1
    )
    df['keeper_placement_score'] = df.apply(
        lambda row: keeper_position_score(row['shot_freeze_frame'], row['location']), axis=1
    )

    # Indicator if there is a key pass
    df['has_key_pass'] = df['key_pass_length'].notna().astype(int)

    return df.dropna(subset=['is_goal']).copy()


def build_pipeline(numerical_features, categorical_features) -> Pipeline:
    """Build a preprocessing and XGBoost pipeline."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=0,
        n_estimators=1456,
        learning_rate=0.0179,
        max_depth=6,
        subsample=0.9358,
        colsample_bytree=0.8721,
        gamma=3.0459,
        min_child_weight=4,
        reg_alpha=0.0865,
        reg_lambda=0.4743
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb_model)
    ])


def main():
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and prepare data
    df = load_and_prepare_data('output/csv/all_shots.csv')
    numeric_cols = [
        'distance_to_goal', 'angle_to_goal', 'possession_duration',
        'key_pass_length', 'key_pass_angle', 'players_in_shot_area',
        'distance_to_goal_squared', 'angle_to_goal_squared', 'distance_angle_interaction',
        'keeper_placement_score'
    ]

    categorical_cols = [
        'play_pattern_id', 'shot_body_part_id', 'shot_technique_id',
        'shot_type_id', 'key_pass_height_id', 'key_pass_body_part_id',
        'under_pressure', 'shot_first_time', 'shot_aerial_won',
        'key_pass_switch', 'key_under_pressure', 'key_pass_cross', 'key_pass_cut_back',
        'possession_bucket', 'key_pass_length_bucket', 'key_pass_angle_bucket'
    ]

    X = df[numeric_cols + categorical_cols]
    y = df['is_goal']

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train on training set
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = pipeline.predict_proba(X_test)[:, 1]
    penalty_mask = X_test['shot_type_id'] == PENALTY_ID
    y_pred[penalty_mask] = 0.75
    auc = roc_auc_score(y_test, y_pred)
    print(f"Test ROC AUC: {auc:.4f}")

    # Retrain on the entire dataset for the final model
    pipeline.fit(X, y)

    # Save the final trained pipeline
    joblib.dump(pipeline, os.path.join(MODEL_DIR, MODEL_FILE))
    print(f"Final model saved at {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # Add predicted xG values to the dataframe
    df['xG'] = pipeline.predict_proba(X)[:, 1]
    df.loc[X['shot_type_id'] == PENALTY_ID, 'xG'] = 0.75
    print(df[['xG', 'is_goal']].head())


if __name__ == '__main__':
    main()
