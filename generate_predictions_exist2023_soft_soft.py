import os
import random
import copy

from odesia_classification import OdesiaTextClassification, OdesiaTextClassificationWithDisagreements
from odesia_configs import DATASETS, GENERIC_MODEL_CONFIG
from odesia_utils import compose_dataset_path, save_json

def get_dataset_config(dataset_name):
    dataset_config = next((task['dataset_config'] for task in DATASETS if task['name'] == dataset_name), None)
    if not dataset_config:
        raise ValueError(f"Dataset name {dataset_name} not found.")
    return dataset_config


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"The path '{path}' was created.")
    else:
        print(f"The path '{path}' already exists.")


def generate_predictions(model_path, model_name,  dataset_name, dataset_lang, output_path,):

    # cargamos los diccionarios con la config del modelo y creamos las carpetas donde lo almacenaremos
    model_config = copy.copy(GENERIC_MODEL_CONFIG)
    model_config['output_dir'] = f"./{output_path}/"   

    dataset_config = get_dataset_config(dataset_name)
    dataset_path = compose_dataset_path(dataset_name, dataset_lang)

    print(f">>>> Predicting {dataset_name} for {model_name}")
    # iniciamos el modelo
    odesia_model = OdesiaTextClassificationWithDisagreements(model_path=model_path,
                                        dataset_path=dataset_path,
                                        model_config=model_config,
                                        dataset_config=dataset_config)
    odesia_model.setup()
    
    # generamos y guardamos las predicciones
    predictions = odesia_model.predict()
    filename = f"preds_{model_name}_{dataset_lang}_{dataset_name}.json"
    path = f"{output_path}/{dataset_name}"

    predictions_modified = [] 
    # For each entry in predictions, change the key 'value' to 'soft_label'
    for entry in predictions['test']:
        new_entry = {}
        new_entry = entry.copy()
        new_entry['value'] =  {k.upper(): v for k, v in new_entry['value'].items()}
        # Now convert the keys inside soft_label from 'NON-SEXIST' to 'NO' and 'SEXIST' to 'YES'
        new_entry['value'] = {k.replace('NON-SEXIST', 'NO').replace('SEXIST', 'YES'): v for k, v in new_entry['value'].items()}

        predictions_modified.append(new_entry)

    ensure_path_exists(path)
    save_json(f"{path}/{filename}", {"test": predictions_modified})


models_base_folder = "trained_models"

for dataset_name in ['exist_2023_t1_soft_soft', 'exist_2023_t2_soft_soft', 'exist_2023_t3_soft_soft']:
    for model_folder in os.listdir(models_base_folder):
        for lan in ['en', 'es']:
            model_base_path = f"{models_base_folder}/{model_folder}/{dataset_name}_{lan}"
            if os.path.isdir(model_base_path):
                for config_folder in os.listdir(model_base_path):
                    model_config_path = f"{model_base_path}/{config_folder}/model"
                    if os.path.isdir(model_config_path):
                        model_path = model_config_path
                        model_name = model_folder
                        output_path = f"./preds/preds_{dataset_name}_{lan}"
                        generate_predictions(model_path, model_name,  dataset_name, lan, output_path,)


# # Search which folder starting with _per_device_train_batch contains the model
# for folder in os.listdir(model_base_path):
#     if folder.startswith('_per_device_train_batch'):
#         # Check if it contains the model folder
#         model_path = f"{model_base_path}/{model_folder}"
#         if os.path.isdir(model_path):




