import os
from itertools import product
import jsbeautifier
import json
import glob

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

def keep_keys(dictionary, keys_to_keep):
    # Create a copy of the original dictionary to avoid modifying it directly
    dictionary_copy = dictionary.copy()
    
    # Get the list of keys from the original dictionary
    original_keys = list(dictionary_copy.keys())
    
    # Remove keys that are not in the list of keys to keep
    for key in original_keys:
        if key not in keys_to_keep:
            del dictionary_copy[key]
    
    return dictionary_copy

def get_documents_in_folder(folder_path):
    # Use os.path.join to get the complete path and '*' pattern to get all files in the folder
    complete_path = os.path.join(folder_path, '*.json')    
    # Use glob to get the list of files matching the pattern
    documents = glob.glob(complete_path)    
    return documents

def add_ids_to_dataset(dataset_path):
    documents_in_folder = get_documents_in_folder(dataset_path)
    i = 0
    for path in documents_in_folder:
        data = json.load(open(path))
        for row in data:
            row['id'] = i
            i += 1
        save_json(data=data, path=path)
        
def rename_item_dataset(dataset_path, last_name, new_name):
    documents_in_folder = get_documents_in_folder(dataset_path)
    for path in documents_in_folder:
        data = json.load(open(path))
        for row in data:
            row[new_name] = row[last_name]
            del row[last_name]
        save_json(data=data, path=path)

def rename_item_dataset_dipromats(dataset_path, last_name, new_name):
    documents_in_folder = get_documents_in_folder(dataset_path)
    for path in documents_in_folder:
        data = json.load(open(path))
        for row in data:
            if row[last_name]:
                row[last_name] = "propaganda"
            else:
                row[last_name] = "non-propaganda"
            row[new_name] = row[last_name]
            del row[last_name]
        save_json(data=data, path=path) 