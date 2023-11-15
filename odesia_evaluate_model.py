import copy
from odesia_classification import OdesiaTextClassification, OdesiaTokenClassification
from odesia_qa import OdesiaQuestionAnswering
from odesia_sentece_similarity import OdesiaSentenceSimilarity
from odesia_configs import DATASETS, GENERIC_MODEL_CONFIG
from odesia_utils import compose_dataset_path, compose_output_dir, create_directories, create_grid, save_json

def calculate_gradient_accumulation():
        
    return


def odesia_benchmark(model : str, language="es", grid_search : dict = None, datasets_to_eval : list = []):
    
    grid = create_grid(grid_search)
       
    # recorremos todos los datasets de ODESIA
    for task in DATASETS:
        dataset_name = task['name']        
        dataset_config = task['dataset_config']
        dataset_path = compose_dataset_path(dataset_name, language)

        # si el dataset está en los elegidos por el usuario
        if dataset_name in datasets_to_eval:
            # para cada dataset buscamos el mejor modelo con nuestro grid
            for hparams in grid:
                # cargamos los diccionarios con la config del modelo y creamos las carpetas donde lo almacenaremos
                model_config = copy.copy(GENERIC_MODEL_CONFIG)
                model_config['output_dir'] = compose_output_dir(dataset_name, model, hparams, language)
                # creamos las carpetas para guardarlo
                create_directories(model_config['output_dir'])
                # añadimos los parametros del grid que vamos a estudiar
                model_config['hf_parameters'].update(hparams)                 
                
                # inicializamos los modelos en función del tipo de problema del dataset
                problem_type = task['problem_type']            
                if problem_type == "text_classification":
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
                
                ''''''
                odesia_model.train()
                save_predictions(odesia_model)
                save_evaluation_report(odesia_model)
    return     

def save_predictions(model):
    predictions = model.predict()
    save_json(model.model_config['output_dir']+"/predictions.json", predictions)

def save_evaluation_report(model):
    evaluation_report = {}
    for split in ['train', 'val', 'test']:
         evaluation_output = model.evaluate(split=split)
         evaluation_report[split] = evaluation_output
    
    save_json(model.model_config['output_dir']+"/evaluation.json",
                          evaluation_report) 

def odesia_generate_inform():
    return