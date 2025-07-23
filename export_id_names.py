import os
import json
import pandas as pd

def process_all_json_and_map_ids(folder_path, output_file="id_name_mappings.xlsx"):
    all_mappings = {}

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"Encontrados {len(json_files)} archivos JSON en {folder_path}")

    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
        df_events = pd.json_normalize(events, sep='_')

        for col in df_events.columns:
            if col.endswith('_id'):
                name_col = col.replace('_id', '_name')
                if name_col in df_events.columns:
                    mapping_df = df_events[[col, name_col]].drop_duplicates().sort_values(col)
                    key = name_col.replace('_name', '')
                    
                    if key in all_mappings:
                        all_mappings[key] = pd.concat([all_mappings[key], mapping_df]).drop_duplicates().sort_values(col)
                    else:
                        all_mappings[key] = mapping_df

    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, mapping_df in all_mappings.items():
            mapping_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Todos los mappings exportados a {output_file}")

process_all_json_and_map_ids("open-data-master/data/events", "id_name_mappings.xlsx")