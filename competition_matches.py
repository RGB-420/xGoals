import os
import json
import pandas as pd

def flatten_keys(d, parent_key='', sep='_'):
    """
    Aplana las claves de un diccionario anidado.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_keys(v, new_key, sep=sep))
        else:
            items.append(new_key)
    
    return items

def has_required_fields(event_file, required_fields):
    """
    Comprueba si un archivo de eventos contiene al menos una vez cada campo necesario.
    """
    try:
        with open(event_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        if not events:
            return False

        # Aplanar todas las claves de todos los eventos del partido
        all_flat_keys = set()
        for e in events:
            flat_keys = flatten_keys(e)
            all_flat_keys.update(flat_keys)

        # Comprobar si todos los campos requeridos aparecen al menos una vez
        return all(field in all_flat_keys for field in required_fields)

    except Exception as ex:
        print(f"Error leyendo {event_file}: {ex}")
        return False

def get_valid_matches(events_folder, required_fields, output_file="valid_matches.csv"):
    """
    Busca partidos con todos los campos necesarios y los guarda en un CSV.
    """
    valid_matches = []
    for root, dirs, files in os.walk(events_folder):
        for file in files:
            print(file)
            if file.endswith('.json'):
                event_file = os.path.join(root, file)
                if has_required_fields(event_file, required_fields):
                    # Extraer competición y partido del path
                    match_id = os.path.splitext(file)[0]
                    valid_matches.append({
                        "match_id": match_id,
                        "event_file": event_file
                    })

    # Crear DataFrame y guardar en CSV
    df_valid = pd.DataFrame(valid_matches)
    df_valid.to_csv(output_file, index=False)
    print(f"Partidos válidos guardados en {output_file} ({len(df_valid)} partidos encontrados).")

    return df_valid

required_fields = [
    'id', 'index', 'period', 'minute', 'second', 'possession',
    'play_pattern_id', 'location', 'under_pressure',
    'shot_key_pass_id', 'shot_body_part_id', 'shot_type_id',
    'shot_outcome_id', 'shot_technique_id', 'shot_first_time', 'shot_aerial_won',
    'id', 'pass_length', 'pass_angle', 'pass_height_id', 'pass_cross',
    'pass_cut_back', 'pass_switch', 'pass_body_part_id'
]

events_folder = "open-data-master/data/events"

# Ejecutar y guardar resultados
valid_df = get_valid_matches(events_folder, required_fields, output_file="valid_matches.csv")
