import json
import pandas as pd
import csv
import os
from flatten_json import flatten  # Puedes instalar esto con pip install flatten_json

def flatten_json(json_data):
    flat_data = flatten(json_data, ".", root_keys_to_ignore=set(["eval_per_class"]))
    return flat_data

def save_to_csv(data, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def process_data(data):
    processed_data = []
    for item in data:
        dataset = item["dataset"]
        language = item["language"]
        flat_item = flatten_json(item)
        processed_data.append({"dataset": dataset, "language": language, **flat_item})
    return processed_data

def generate_csv_from_report():
    data = json.load(open('report.json'))
    processed_data = process_data(data)

    # Creamos un diccionario para almacenar los datos agrupados por "dataset" e "idioma"
    grouped_data = {}
    for item in processed_data:
        dataset_language = f"{item['dataset']}_{item['language']}"
        if dataset_language not in grouped_data:
            grouped_data[dataset_language] = []
        grouped_data[dataset_language].append(item)

    # Guardamos los datos en archivos CSV separados
    for key, value in grouped_data.items():
        file_name = f"csvs/{key}.csv"
        save_to_csv(value, file_name)
        df = pd.read_csv(file_name)
        df = df.drop_duplicates(subset=['model_config.output_dir'])
        df.to_csv(file_name)


    print("CSVs generados con éxito.")
