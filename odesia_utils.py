import os
from itertools import product
import jsbeautifier
import json

import numpy as np
def create_directories(path):
    try:
        # Create all directories in the path
        os.makedirs(path)
        print(f"Directories created at: {path}")
    except FileExistsError:
        # The path already exists, not an error
        print(f"The path already exists: {path}")
    except OSError as e:
        # A different error occurred
        print(f"Error creating directories at {path}: {e}")


def create_grid(hparams_to_search: dict):
    grid = []
    
    keys = hparams_to_search.keys()
    values = hparams_to_search.values()
    
    # Calcular el producto cartesiano usando itertools.product
    for combination in product(*values):
        cartesian_product = dict(zip(keys, combination))
        grid.append(cartesian_product)

    return grid

def remove_special_chars(text):
    return text.replace("/","-").replace(".","").replace(",","")

def save_log(log_history, path):
    with open(f'{path}/trainer_log.txt', 'w') as f:
        f.write(f"{log_history}")

def compose_output_dir(dataset_name, model, hparams, language):
    model = remove_special_chars(model)
    output_dir = f"trained_models/{model}/{dataset_name}_{language}/"
    for param in hparams:
        value = hparams[param]
        output_dir += f"_{param}_{value}"
    return output_dir
    
def compose_dataset_path(dataset, language):
    dataset_path = {}
    for split in ['train', 'test', 'val']:
        dataset_path[split] = f'datasets/{dataset}/{split}_{language}.json'
    return dataset_path

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def save_json(path, data):
    with open(path, 'w', encoding='utf8') as fp:
        fp.write(jsbeautifier.beautify(json.dumps(data, cls=NumpyFloatValuesEncoder, ensure_ascii=False)))