


import copy
from odesia_classification import OdesiaTokenClassification
from odesia_configs import DATASETS, GENERIC_MODEL_CONFIG
from odesia_utils import compose_dataset_path, save_json


# cargamos los diccionarios con la config del modelo y creamos las carpetas donde lo almacenaremos
model_config = copy.copy(GENERIC_MODEL_CONFIG)
model_config['output_dir'] = "."   

for task in DATASETS:
        dataset_name = task['name']        
        dataset_config = task['dataset_config']
        dataset_path = compose_dataset_path(dataset_name, 'es')
        if(dataset_name == 'multiconer_2022'):
            odesia_model = OdesiaTokenClassification(model_path='/data/gmarco/odesia_benchmark/trained_models/roberta-large/multiconer_2022_en/_per_device_train_batch_size_8_gradient_accumulation_steps_2_learning_rate_5e-05_weight_decay_0.1/model',
                                                dataset_path=dataset_path,
                                                model_config=model_config,
                                                dataset_config=dataset_config)
            predictions = odesia_model.predict()
            print(predictions)
            save_json(f"./predictions.json", predictions)