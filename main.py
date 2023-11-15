import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from odesia_evaluate_model import odesia_benchmark


def main():
    language_models = {'en':['distilbert-base-uncased','xlm-roberta-base','xlm-roberta-large', 'bert-base-multilingual-cased',
                'distilbert-base-multilingual-cased', 'roberta-base', 'roberta-large',
                 'bert-base-cased', 'bert-base-uncased','ixa-ehu/ixambert-base-cased'],
                    'es':['CenIA/distillbert-base-spanish-uncased','xlm-roberta-base','xlm-roberta-large', 'bert-base-multilingual-cased',
                        'distilbert-base-multilingual-cased', 'PlanTL-GOB-ES/roberta-base-bne',
                        'PlanTL-GOB-ES/roberta-large-bne', 'bertin-project/bertin-roberta-base-spanish',
                        'dccuchile/bert-base-spanish-wwm-cased', 
                        'ixa-ehu/ixambert-base-cased'],
            } 
    
    hparams_to_search = {
            'per_device_train_batch_size' : [16, 32],
            'learning_rate': [0.00001, 0.00003, 0.00005],
            'weight_decay': [0.1, 0.01]
        }
    
    for language in language_models:
        for model in language_models[language]:
            odesia_benchmark(model=model, 
                             language=language, 
                             grid_search=hparams_to_search, 
                             datasets_to_eval=['dipromats_2023_t1'])
    
    
if __name__ == "__main__":
    main()