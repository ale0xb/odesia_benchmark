import os
import pandas as pd

NORMALIZATION_INTERVALS = { # -gold/+gold 
    "t1": {
        "en": {
            "hard_hard": [-0.9798, 0.9798],    
            "hard_soft": [-3.1141, 3.1141],
            "soft_soft": [-3.1141, 3.1141],
        },
        "es": {
            "hard_hard": [-0.9999, 0.9999],
            "hard_soft": [-3.1177, 3.1177],
            "soft_soft": [-3.1177, 3.1177],
        }
    },
    "t2": {
        "en": {
            "hard_hard": [-1.4449, 1.4449],
            "hard_soft": [-6.1178, 6.1178],
            "soft_soft": [-6.1178, 6.1178],
        },
        "es": {
            "hard_hard": [-1.6007, 1.6007],
            "hard_soft": [-6.2431, 6.2431],
            "soft_soft": [-6.2431, 6.2431],
        }       
    },
    "t3": {
        "en": {
            "hard_hard": [-2.0402, 2.0402],
            "hard_soft": [-9.1255, 9.1255],
            "soft_soft": [-9.1255, 9.1255],
        },
        "es": {
            "hard_hard": [-2.2393, 2.2393],
            "hard_soft": [-9.6071, 9.6071],
            "soft_soft": [-9.6071, 9.6071]
        }
    }
}

NORMALIZATION_INTERVALS_COMPETITION =  { # -gold/+gold 
    "t1": {
        "en": {
            "hard_hard": [-0.3965, 0.9798],    
            "hard_soft": [-3.8158, 3.1141],
            "soft_soft": [-3.8158, 3.1141],
        },
        "es": {
            "hard_hard": [-0.4897, 0.9999],
            "hard_soft": [-2.5742, 3.1177],
            "soft_soft": [-2.5742, 3.1177],
        }
    },
    "t2": {
        "en": {
            "hard_hard": [-3.4728, 1.4449],
            "hard_soft": [-39.4948, 6.1178],
            "soft_soft": [-39.4948, 6.1178],
        },
        "es": {
            "hard_hard": [-2.939, 1.6007],
            "hard_soft": [-28.7093, 6.2431],
            "soft_soft": [-28.7093, 6.2431],
        }       
    },
    "t3": {
        "en": {
            "hard_hard": [-2.9279, 2.0402],
            "hard_soft": [-46.9473, 9.1255],
            "soft_soft": [-46.9473, 9.1255],
        },
        "es": {
            "hard_hard": [-3.3196, 2.2393],
            "hard_soft": [-45.426, 9.6071],
            "soft_soft": [-45.426, 9.6071]
        }
    }
}


MAIN_METRICS = {
    "hard_hard": "eval_icm_hard",
    "hard_soft": "eval_icm_soft",
    "soft_soft": "eval_icm_soft",
}

OUTPUT_DIR = "csvs/exist_2023_best/"

model_to_class = {
    "distilbert-base-uncased": "DistilBERT",
    "roberta-base": "RoBERTa",
    "roberta-large": "RoBERTa",
    "distilbert-base-multilingual-cased": "DistilBERT",
    "bert-base-cased": "BERT",
    "bert-base-multilingual-cased": "BERT",
    "ixa-ehu/ixambert-base-cased": "BERT",
    "xlm-roberta-large": "RoBERTa",
    "xlm-roberta-base": "RoBERTa",
    "PlanTL-GOB-ES/roberta-base-bne": "RoBERTa",
    "PlanTL-GOB-ES/roberta-large-bne": "RoBERTa",
    "bertin-project/bertin-roberta-base-spanish": "RoBERTa",
    "CenIA/distillbert-base-spanish-uncased": "DistilBERT",
    "dccuchile/bert-base-spanish-wwm-cased": "BERT"
}


model_to_size = {
    "distilbert-base-uncased": "base",
    "roberta-base": "base",
    "roberta-large": "large",
    "distilbert-base-multilingual-cased": "base",
    "bert-base-cased": "base",
    "bert-base-multilingual-cased": "base",
    "ixa-ehu/ixambert-base-cased": "base",
    "xlm-roberta-large": "large",
    "xlm-roberta-base": "base",
    "PlanTL-GOB-ES/roberta-base-bne": "base",
    "PlanTL-GOB-ES/roberta-large-bne": "large",
    "bertin-project/bertin-roberta-base-spanish": "base",
    "CenIA/distillbert-base-spanish-uncased": "base",
    "dccuchile/bert-base-spanish-wwm-cased": "base"
}

model_to_language = {
    "distilbert-base-uncased": "en",
    "roberta-base": "en",
    "roberta-large": "en",
    "distilbert-base-multilingual-cased": "multilingual",
    "bert-base-cased": "en",
    "bert-base-multilingual-cased": "multilingual",
    "ixa-ehu/ixambert-base-cased": "multilingual",
    "xlm-roberta-large": "multilingual",
    "xlm-roberta-base": "multilingual",
    "PlanTL-GOB-ES/roberta-base-bne": "es",
    "PlanTL-GOB-ES/roberta-large-bne": "es",
    "bertin-project/bertin-roberta-base-spanish": "es",
    "CenIA/distillbert-base-spanish-uncased": "es",
    "dccuchile/bert-base-spanish-wwm-cased": "es"
}

monolingual_class = {
    'distilbert-base-uncased': 'distilbert-base-uncased',
    'bert-base-cased': 'bert-base-cased',
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',
    'CenIA/distillbert-base-spanish-uncased': 'distilbert-base-uncased',
    'dccuchile/bert-base-spanish-wwm-cased': 'bert-base-cased',
    'PlanTL-GOB-ES/roberta-base-bne': 'roberta-base',
    'PlanTL-GOB-ES/roberta-large-bne': 'roberta-large'
}

def main():
    # Make sure the output dir is created
    os.makedirs(OUTPUT_DIR, exist_ok=True)   
    # Read baselines
    baselines_df = pd.read_csv("csvs/exist_2023_baseline_results.csv")
    baselines_df.set_index("run")

    all_df = []

    avg_baselines = baselines_df.mean()

    for task in ["t1", "t2", "t3"]:
        for mode in ['hard_hard', 'hard_soft', 'soft_soft']:
            task_id = f"{task}_{mode}"
            # read the csv for this task 
            task_df_en = pd.read_csv(f"csvs/exist_2023_{task_id}_en.csv")
            task_df_es = pd.read_csv(f"csvs/exist_2023_{task_id}_es.csv")
    
            best_models_en = task_df_en.loc[task_df_en.groupby(['model', 'language'])[f'evaluation.val.{MAIN_METRICS[mode]}'].idxmax()]
            best_models_es = task_df_es.loc[task_df_es.groupby(['model', 'language'])[f'evaluation.val.{MAIN_METRICS[mode]}'].idxmax()]

            NORM_INTERVAL_EN = NORMALIZATION_INTERVALS[task]["en"][mode]
            NORM_INTERVAL_ES = NORMALIZATION_INTERVALS[task]["es"][mode]

            # Set the lower end of the norm9
            if best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}'].min() < NORM_INTERVAL_EN[0]:
                print(f"Baseline for {task_id} English is outside the normalization interval")
                print("Will use the baseline as lower end of the interval")
                NORM_INTERVAL_EN = [best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}'].min(), NORM_INTERVAL_EN[1]]

            if best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}'].min() < NORM_INTERVAL_ES[0]:
                print(f"Baseline for {task_id} Spanish is outside the normalization interval")
                print("Will use the baseline as lower end of the interval")
                NORM_INTERVAL_ES = [best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}'].min(), NORM_INTERVAL_ES[1]]

            # Save task and mode
            best_models_en["task"] = task
            best_models_en["mode"] = mode
            best_models_es["task"] = task
            best_models_es["mode"] = mode

            # Set mono class
            best_models_en["mono_class"] = best_models_en["model"].map(monolingual_class)
            best_models_es["mono_class"] = best_models_es["model"].map(monolingual_class)
        


            # Save baseline info
            best_models_en["baseline"] = avg_baselines[f"{task_id}_en"]
            best_models_es["baseline"] = avg_baselines[f"{task_id}_es"]
            
            # Now normalize the values of 'evaluation.test.eval_icm_hard' for each model between 0 and 1
            best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] = (best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}'] - NORM_INTERVAL_EN[0]) / (NORM_INTERVAL_EN[1] - NORM_INTERVAL_EN[0])
            best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] = (best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}'] - NORM_INTERVAL_ES[0]) / (NORM_INTERVAL_ES[1] - NORM_INTERVAL_ES[0])

            # Compute the absolute increment
            best_models_en["increment_abs"] = best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}'] - best_models_en["baseline"]
            best_models_es["increment_abs"] = best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}'] - best_models_es["baseline"]


            # Normalize the baseline too
            best_models_en["baseline_norm"] = (best_models_en["baseline"] - NORM_INTERVAL_EN[0]) / (NORM_INTERVAL_EN[1] - NORM_INTERVAL_EN[0])
            best_models_es["baseline_norm"] = (best_models_es["baseline"] - NORM_INTERVAL_ES[0]) / (NORM_INTERVAL_ES[1] - NORM_INTERVAL_ES[0])


            # Calculate the difference between the model and the baseline
            best_models_en["increment_norm"] = best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] - best_models_en["baseline_norm"]
            best_models_es["increment_norm"] = best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] - best_models_es["baseline_norm"]

            # Keep only the relevant columns: model, language, and the metrics
            best_models_en = best_models_en[['model',
                                             'task',
                                             'mode',
                                             'language',
                                             'baseline_norm',
                                             'mono_class',
                                             f'evaluation.test.{MAIN_METRICS[mode]}_norm', 
                                             f'evaluation.test.{MAIN_METRICS[mode]}', 
                                             'baseline',
                                             'increment_abs',
                                             'increment_norm']]
            
            best_models_es = best_models_es[['model',
                                             'task',
                                             'mode', 
                                             'language', 
                                             'baseline_norm',
                                             'mono_class',
                                             f'evaluation.test.{MAIN_METRICS[mode]}_norm', 
                                             f'evaluation.test.{MAIN_METRICS[mode]}', 
                                             'baseline',
                                             'increment_abs',
                                             'increment_norm']]

            # Sort models from best to worst
            best_models_en = best_models_en.sort_values(by=f'evaluation.test.{MAIN_METRICS[mode]}_norm', ascending=False)
            best_models_es = best_models_es.sort_values(by=f'evaluation.test.{MAIN_METRICS[mode]}_norm', ascending=False)

            # Round all numbers to 4 decimal places
            best_models_en = best_models_en.round(4)
            best_models_es = best_models_es.round(4)

            # # Move the last column two positions to the left
            # cols = best_models_en.columns.tolist()
            # cols = cols[:4] + cols[-1:] + cols[4:-1]
            # best_models_en = best_models_en[cols]
            # best_models_es = best_models_es[cols]

            # Concat the two dataframes
            merged_df = pd.concat([best_models_en, best_models_es], ignore_index=True)
        
            # Here we should add the baseline info. 
            merged_df["class"] = merged_df["model"].map(model_to_class)
            merged_df["training_language"] = merged_df["model"].map(model_to_language)
            merged_df['size'] = merged_df['model'].map(model_to_size)
            # Save the merged dataframe
            merged_df.to_csv(f"{OUTPUT_DIR}/exist_2023_{task_id}_best_en_es.csv", index=False)
            
            all_df.append(merged_df)
    # Merge and save all dataframes
    all_df = pd.concat(all_df, ignore_index=True)
    all_df.to_csv(f"{OUTPUT_DIR}/exist_2023_best.csv", index=False)


if __name__ == "__main__":
    main()