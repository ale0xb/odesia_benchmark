import copy
import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from odesia_evaluate_model import odesia_benchmark


def main():

    # Ensure csv folder exists
    if not os.path.exists('csvs'):
        os.makedirs('csvs')

    # Ensure report.json exists. If not, create it with an empty list
    if not os.path.exists('./report.json'):
        with open('report.json', 'w') as f:
            f.write('[]')
    
    
    LARGE = ['PlanTL-GOB-ES/roberta-large-bne', 'xlm-roberta-large', 'xlm-roberta-base', 'roberta-large', 'bert-base-multilingual-cased','bert-base-cased',]
    
    language_models = {'en':[ 
                            'distilbert-base-uncased', 
                            # 'roberta-base', 
                            # 'roberta-large', 
                            # 'distilbert-base-multilingual-cased', 
                            # 'bert-base-cased', 'bert-base-multilingual-cased',
                            # 'ixa-ehu/ixambert-base-cased',  'xlm-roberta-large', 
                            # 'xlm-roberta-base',                                                
                            ],
                             'es':[
                                 #'PlanTL-GOB-ES/roberta-base-bne','PlanTL-GOB-ES/roberta-large-bne','bertin-project/bertin-roberta-base-spanish',  
                             'distilbert-base-multilingual-cased',
                            #  'CenIA/distillbert-base-spanish-uncased',  
                              #'dccuchile/bert-base-spanish-wwm-cased', 'bert-base-multilingual-cased', 
                              # 'ixa-ehu/ixambert-base-cased',  'xlm-roberta-large',
                              #'xlm-roberta-base',                      
                        ],
                        
                    
            }

    hparams_to_search_small = {
            'per_device_train_batch_size' : [32, 16],
            'learning_rate': [0.00001, 0.00003, 0.00005],
            'weight_decay': [0.1, 0.01]
        }
    
    hparams_to_search_large = {
            'per_device_train_batch_size' : [8],
            'gradient_accumulation_steps' : [4, 2],
            'learning_rate': [0.00001, 0.00003, 0.00005],
            'weight_decay': [0.1, 0.01]
        }
    
    total_iterations = len(language_models['es']) + len(language_models['en'])
    current_iterations = 0
    for language in language_models:
        
        for model in language_models[language]:                        

            start_time = time.time()

            if model in LARGE:
                hparams_to_search = copy.deepcopy(hparams_to_search_large)
            else:
                hparams_to_search = copy.deepcopy(hparams_to_search_small)
            
            odesia_benchmark(model=model, 
                             language=language, 
                             grid_search=hparams_to_search, 
                             datasets_to_eval=[
                                #  'dipromats_2023_t2',
                                #  'dipromats_2023_t3',
                                #  'exist_2023_t1_hard_hard',
                                #  'exist_2023_t1_hard_soft',
                                #  'exist_2023_t2_hard_hard',
                                #  'exist_2023_t2_hard_soft',
                                'exist_2023_t3_hard_hard',
                            ]
            )
                      

            # calculamos el tiempo de esta ejecucion
            iteration_time = time.time() - start_time

            remaining_time_estimate = iteration_time * (total_iterations - current_iterations - 1)

            days = int(remaining_time_estimate // (24 * 3600))
            hours = int((remaining_time_estimate % (24 * 3600)) // 3600)
            minutes = int((remaining_time_estimate % 3600) // 60)
            seconds = int(remaining_time_estimate % 60)
            current_iterations += 1
            print("##############################")
            print("##############################")
            print(f"Model {current_iterations}/{total_iterations} - Estimated remaining based in the last model {model}: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
            print("##############################")
            print("##############################")
            
                
if __name__ == "__main__":
    main()