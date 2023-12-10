import copy
from datetime import timedelta
import itertools
import json
from generate_csv import generate_csv_from_report
from odesia_classification import OdesiaTextClassification, OdesiaTokenClassification
from odesia_qa import OdesiaQuestionAnswering
from odesia_sentence_similarity import OdesiaSentenceSimilarity
from odesia_configs import DATASETS, GENERIC_MODEL_CONFIG
from odesia_utils import compose_dataset_path, compose_output_dir, create_directories, create_grid, get_documents_in_folder, save_json
import time
import datetime
import shutil
import os
from transformers import logging
import warnings
import pandas as pd

# Suprimir todas las advertencias de tipo UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

#logging.set_verbosity_warning()
#logging.set_verbosity_error()


def odesia_benchmark(model : str, language="es", grid_search : dict = None, datasets_to_eval : list = []):
    
    grid = create_grid(grid_search)
    datasets_len = len(datasets_to_eval) if datasets_to_eval else len(DATASETS)
    total_trainings = datasets_len * len(grid) 
    current_iterations = 0
    

    # recorremos todos los datasets de ODESIA
    for task in DATASETS:
        dataset_name = task['name']        
        dataset_config = task['dataset_config']
        dataset_path = compose_dataset_path(dataset_name, language)

        # si el dataset está en los elegidos por el usuario
        if not datasets_to_eval or dataset_name in datasets_to_eval:

            # para cada dataset buscamos el mejor modelo con nuestro grid
            for hparams in grid:

                start_time = time.time()

                # cargamos los diccionarios con la config del modelo y creamos las carpetas donde lo almacenaremos
                model_config = copy.copy(GENERIC_MODEL_CONFIG)
                model_config['output_dir'] = compose_output_dir(dataset_name, model, hparams, language)                
                create_directories(model_config['output_dir'])

                # si ya tenemos este modelo entrenado, pasamos
                df_past_trainings = pd.read_csv(f'csvs/{dataset_name}_{language}.csv')
                '''
                Este fragmento habrá que borrarlo en la versión final, o hacerlo de otra manera.
                Si el modelo es grande y hay que hacer una acumulación de gradiente, 
                hay que hallar si se ha entrenado un modelo equivalente de tal manera que no se entrene dos veces.
                '''
                list_grid_models = df_past_trainings['model_config.output_dir'].unique()
                dict_equivalences = {32: ['per_device_train_batch_size_8_gradient_accumulation_steps_4', 'per_device_train_batch_size_4_gradient_accumulation_steps_8', 'per_device_train_batch_size_32'],
                                     16: ['per_device_train_batch_size_8_gradient_accumulation_steps_2', 'per_device_train_batch_size_4_gradient_accumulation_steps_4', 'per_device_train_batch_size_16']}
                already_trained = False
                if hparams.get('gradient_accumulation_steps') != None:
                    total_batch_size = hparams['gradient_accumulation_steps'] * hparams['per_device_train_batch_size']
                    for elemento in dict_equivalences[total_batch_size]:
                        elemento = '/'.join(model_config['output_dir'].split("/")[:-1]) + f"/_{elemento}_learning_rate_{hparams['learning_rate']}_weight_decay_{hparams['weight_decay']}"
                        if elemento in list_grid_models:
                            print("<<<>>>> It was already trained in other configuration.", elemento)
                            already_trained = True
                            
                if already_trained:
                    continue

                if model_config['output_dir'] in list_grid_models:
                    print(f">>>>>>>>> Already trained. Skipping model {model_config['output_dir']}.")
                    continue

                # añadimos los parametros del grid que vamos a estudiar
                model_config['hf_parameters'].update(hparams)                 
                
                # inicializamos los modelos en función del tipo de problema del dataset
                problem_type = task['dataset_config']['problem_type']            
                if problem_type in ['single_label_classification', '', 'multi_class_classification', 'multi_label_classification']:
                    odesia_model = OdesiaTextClassification(model_path=model,
                                                            dataset_path=dataset_path,
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)

                elif problem_type == "token_classification":
                    odesia_model = OdesiaTokenClassification(model_path=model,
                                                            dataset_path=dataset_path,
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)
                elif problem_type == "question_answering":
                    odesia_model = OdesiaQuestionAnswering(model_path=model,
                                                            dataset_path=dataset_path,
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)
                elif problem_type == "sentence_similarity":
                    odesia_model = OdesiaSentenceSimilarity(model_path=model,
                                                            dataset_path=dataset_path,
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)
                
                print(f"[{datetime.datetime.now()}] >>>> Training...")                
                odesia_model.train()

                print(f"[{datetime.datetime.now()}] >>>> Evaluation...", datetime.datetime.now())
                evaluation_report = save_evaluation_report(odesia_model)

                print(f"[{datetime.datetime.now()}] >>>> Prediction...", datetime.datetime.now())                
                save_predictions(odesia_model)

                # quitamos de la memoria de la gpu el modelo
                odesia_model.purge_model()
                
                # calculamos el tiempo de esta ejecucion
                iteration_time = time.time() - start_time

                remaining_time_estimate = iteration_time * (total_trainings - current_iterations - 1)

                days = int(remaining_time_estimate // (24 * 3600))
                hours = int((remaining_time_estimate % (24 * 3600)) // 3600)
                minutes = int((remaining_time_estimate % 3600) // 60)
                seconds = int(remaining_time_estimate % 60)
                current_iterations += 1
                print("*****************************")
                print(f"Iteration {current_iterations}/{total_trainings} - Estimated remaining for model {model} in {dataset_name}: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
                print("*****************************")
                
                # guardamos los datos de la ejecución por si necesitamos reanudarla en algún momento
                append_model_to_history(model, model_config, dataset_name, language, iteration_time, evaluation_report)
                generate_csv_from_report()
                
                # limpiamos el disco duro
                purge_disk(path = '/'.join(model_config['output_dir'].split('/')[0:-1]), 
                           main_metric=odesia_model.dataset_config['main_metric'], 
                           num_model_preserve = 1)
    return     

def append_model_to_history(model, model_config, dataset, language, time, evaluation_report):
    report = json.load(open('./report.json'))

    row = {
        'date': str(datetime.datetime.now()),
        'dataset':dataset,
        'model':model,
        'model_config':model_config,
        'language':language,
        'training_time':time,
        'evaluation':evaluation_report,
    }
    
    report.append(row)
    save_json(path='./report.json', data=report)

def save_predictions(model):
    predictions = model.predict()
    save_json(f"{model.model_config['output_dir']}/predictions.json", predictions)
    
def save_evaluation_report(model):
    evaluation_report = {}
    for split in ['val', 'test']:
         evaluation_output = model.evaluate(split=split)
         evaluation_report[split] = evaluation_output
    
    save_json(f"{model.model_config['output_dir']}/evaluation.json", evaluation_report) 
    return evaluation_report

def purge_disk(path, main_metric, num_model_preserve):
        
    all_models = [os.path.join(path, nombre) for nombre in os.listdir(path) if os.path.isdir(os.path.join(path, nombre))]
    models_metric = {}
    
    for model_path in all_models:
        eval_path = f'{model_path}/evaluation.json'
        if os.path.exists(eval_path):
            evaluation_data = json.load(open(eval_path))['val']
            models_metric[model_path] = evaluation_data[main_metric]
        
    if len(models_metric) == 0:
        return
    
    # ordenamos mejores modelos por valor
    models_metric = {k: v for k, v in sorted(models_metric.items(), key=lambda x: x[1])}
        
    print(f'Best model: {list(models_metric.keys())[-1]}')
    print(f'    {main_metric}: {list(models_metric.values())[-1]}')
    
    if len(models_metric) > 1:
        dict_slice = list(itertools.islice(models_metric.items(), len(models_metric)-num_model_preserve))
        for path, metric in dict_slice:
            if os.path.isdir(f'{path}/model'):
                print(f'    >>>> Deleting model...', path, metric)
                shutil.rmtree(f'{path}/model')
    else:
        print('Nothing to purge in the disk.')
            
    