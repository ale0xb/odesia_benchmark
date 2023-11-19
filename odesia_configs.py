
DATASETS = [{'name' : 'dipromats_2023_t1',             
             'dataset_config':{"evall_test_case":"DIPROMATS2023",
                               'main_metric': 'eval_f1_macro',
                               'problem_type':'text_classification_multiclass',
                                "label2id":{"false": 0, 
                                           "true": 1},
                               'label_column':'label_task1_hf',}
             },            
            
            {'name':'dipromats_2023_t2',              
             'dataset_config':{
                                "evall_test_case":"DIPROMATS2023",
                                'main_metric': 'eval_f1_macro',
                                'problem_type':'text_classification_multilabel',
                                "label2id":{"1 appeal to commonality": 0,
                                            "2 discrediting the opponent": 1,
                                            "3 loaded language": 2,
                                            "4 appeal to authority": 3,
                                            "false": 4},
                                    "label_column":"label_task2_hf"
                            }
                }
            
            ,{'name':'dipromats_2023_t3',              
             'dataset_config':{
                                "evall_test_case":"DIPROMATS2023",
                                'main_metric': 'eval_f1_macro',
                                'problem_type':'text_classification_multilabel',
                                "label2id":{"1 appeal to commonality - ad populum":0,
                                        "1 appeal to commonality - flag waving":1,
                                        "2 discrediting the opponent - absurdity appeal":2,
                                        "2 discrediting the opponent - demonization":3,
                                        "2 discrediting the opponent - doubt":4,
                                        "2 discrediting the opponent - fear appeals (destructive)":5,
                                        "2 discrediting the opponent - name calling":6,
                                        "2 discrediting the opponent - propaganda slinging":7,
                                        "2 discrediting the opponent - scapegoating":8,
                                        "2 discrediting the opponent - undiplomatic assertiveness/whataboutism":9,
                                        "3 loaded language":10,
                                        "4 appeal to authority - appeal to false authority":11,
                                        "4 appeal to authority - bandwagoning":12,
                                        "false":13},
                                    "label_column":"label_task3_hf",
                            }
             }
            
            ,{'name':'exist_2022_t1',              
             'dataset_config':{"evall_test_case":"EXIST2022",
                               'main_metric': 'eval_f1_macro',
                               'problem_type':'text_classification_multiclass',
                                "label2id":{'non-sexist':0,
                                            'sexist':1},
                                "label_column":"label_text",                                
                                }
             }
            
            ,{'name':'exist_2022_t2',              
             'dataset_config':{"evall_test_case":"EXIST2022",
                               'main_metric': 'eval_f1_macro',
                               'problem_type':'text_classification_multiclass',
                                "label2id":{'sexual-violence':0,
                                            'stereotyping-dominance':1, 
                                            'non-sexist':2,
                                            'misogyny-non-sexual-violence':3,      
                                            'ideological-inequality':4,             
                                            'objectification': 5},
                                "label_column":"label_text",
                                }
             },
            
            {'name':'mldoc_2018',              
             'dataset_config':{"evall_test_case":"MLDOC",
                               'main_metric': 'eval_f1_macro',
                               'problem_type':'text_classification_multiclass',
                                "label2id":{'MCAT':0, 'GCAT':1, 'ECAT':2,'CCAT':3},
                                "label_column":"label_text",
                                }
            },
            
            {'name' :'diann_2023',
            'dataset_config'    : {"evall_test_case":"DIAN2023",
                                   'main_metric': 'eval_f1',
                                  'problem_type':'token_classification',
                                   "label2id":{'O':0,
                                              'B-DIS': 1, 
                                              'I-DIS': 2}
                                    }
            },
            
            {'name':'multiconer_2022',              
             'dataset_config':{"evall_test_case":'CONER2022',
                               'main_metric': 'eval_f1',
                               'problem_type':'token_classification',
                               "label2id":{'B-GRP': 0, 
                                               'B-CW': 1, 
                                               'I-PER': 2, 
                                               'I-CW': 3, 
                                               'B-CORP': 4, 
                                               'I-CORP': 5, 
                                                'I-LOC': 6,
                                                'I-PROD': 7, 
                                                'B-LOC': 8, 
                                                'I-GRP': 9, 
                                                'B-PROD': 10, 
                                                'O': 11, 
                                                'B-PER': 12},
                                "label_column":"label",
                            }
            }
            
            ,{'name':'sqad_2022_squad_2016',              
             'dataset_config':{'evall_test_case':"SAQ2022",
                               'main_metric': 'f1',
                               'problem_type':'question_answering'}}
            
            ,{'name':'sts_2017',              
             'dataset_config':{'evall_test_case':"STS2017",
                               'main_metric': 'cosine_pearson',
                               'problem_type':'sentence_similarity',
                               'max_score':5.0}},
            
        ]


GENERIC_MODEL_CONFIG = {
        "output_dir" : "",
        "hf_parameters": {
                'per_device_train_batch_size':8,
                'num_train_epochs': 5,
                'evaluation_strategy':"no",
                'save_strategy':"no",
                'load_best_model_at_end':True}
}

