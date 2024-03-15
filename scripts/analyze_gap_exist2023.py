import os
import pandas as pd

NORMALIZATION_INTERVALS = { # -gold/+gold 
    "t1": {
        "en": {
            "hard_hard": 0.9798,    
            "hard_soft": 3.1141,
            "soft_soft": 3.1141,
        },
        "es": {
            "hard_hard": 0.9999,
            "hard_soft": 3.1177,
            "soft_soft": 3.1177,
        }
    },
    "t2": {
        "en": {
            "hard_hard": 1.4449,
            "hard_soft": 6.1178,
            "soft_soft": 6.1178,
        },
        "es": {
            "hard_hard": 1.6007,
            "hard_soft": 6.2431,
            "soft_soft": 6.2431,
        }       
    },
    "t3": {
        "en": {
            "hard_hard": 2.0402,
            "hard_soft": 9.1255,
            "soft_soft": 9.1255,
        },
        "es": {
            "hard_hard": 2.2393,
            "hard_soft": 9.6071,
            "soft_soft": 9.6071
        }
    }
}

MAIN_METRICS = {
    "hard_hard": "eval_icm_hard",
    "hard_soft": "eval_icm_soft",
    "soft_soft": "eval_icm_soft",
}

OUTPUT_DIR = "csvs/exist_2023_best/"

def main():
    # Make sure the output dir is created
    os.makedirs(OUTPUT_DIR, exist_ok=True)   
    # Read baselines
    baselines_df = pd.read_csv("csvs/exist_2023_baseline_results.csv")
    baselines_df.set_index("run")

    avg_baselines = baselines_df.mean()

    for task in ["t1", "t2", "t3"]:
        for mode in ['hard_hard', 'hard_soft', 'soft_soft']:
            task_id = f"{task}_{mode}"
            # read the csv for this task 
            task_df_en = pd.read_csv(f"csvs/exist_2023_{task_id}_en.csv")
            task_df_es = pd.read_csv(f"csvs/exist_2023_{task_id}_es.csv")

    
            best_models_en = task_df_en.loc[task_df_en.groupby(['model', 'language'])[f'evaluation.val.{MAIN_METRICS[mode]}'].idxmax()]
            best_models_es = task_df_es.loc[task_df_es.groupby(['model', 'language'])[f'evaluation.val.{MAIN_METRICS[mode]}'].idxmax()]

            NORM_INTERVAL_EN = [-NORMALIZATION_INTERVALS[task]["en"][mode], NORMALIZATION_INTERVALS[task]["en"][mode]]
            NORM_INTERVAL_ES = [-NORMALIZATION_INTERVALS[task]["es"][mode], NORMALIZATION_INTERVALS[task]["es"][mode]]

            # Save baseline info
            best_models_en["baseline"] = avg_baselines[f"{task_id}_en"]
            best_models_es["baseline"] = avg_baselines[f"{task_id}_es"]
            
            # Now normalize the values of 'evaluation.test.eval_icm_hard' for each model between 0 and 1
            best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] = (best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}'] - NORM_INTERVAL_EN[0]) / (NORM_INTERVAL_EN[1] - NORM_INTERVAL_EN[0])
            best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] = (best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}'] - NORM_INTERVAL_ES[0]) / (NORM_INTERVAL_ES[1] - NORM_INTERVAL_ES[0])

            # Normalize the baseline too
            best_models_en["baseline_norm"] = (best_models_en["baseline"] - NORM_INTERVAL_EN[0]) / (NORM_INTERVAL_EN[1] - NORM_INTERVAL_EN[0])
            best_models_es["baseline_norm"] = (best_models_es["baseline"] - NORM_INTERVAL_ES[0]) / (NORM_INTERVAL_ES[1] - NORM_INTERVAL_ES[0])

            # Keep only the relevant columns: model, language, and the metrics
            best_models_en = best_models_en[['model', 
                                             'language',
                                             'baseline_norm',
                                             f'evaluation.test.{MAIN_METRICS[mode]}_norm', 
                                             f'evaluation.test.{MAIN_METRICS[mode]}', 
                                             'baseline']]
            
            best_models_es = best_models_es[['model', 
                                             'language', 
                                             'baseline_norm',
                                             f'evaluation.test.{MAIN_METRICS[mode]}_norm', 
                                             f'evaluation.test.{MAIN_METRICS[mode]}', 
                                             'baseline']]

            # Calculate the difference between the model and the baseline
            best_models_en["increment"] = best_models_en[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] - best_models_en["baseline_norm"]
            best_models_es["increment"] = best_models_es[f'evaluation.test.{MAIN_METRICS[mode]}_norm'] - best_models_es["baseline_norm"]

            # Sort models from best to worst
            best_models_en = best_models_en.sort_values(by=f'evaluation.test.{MAIN_METRICS[mode]}_norm', ascending=False)
            best_models_es = best_models_es.sort_values(by=f'evaluation.test.{MAIN_METRICS[mode]}_norm', ascending=False)

            # Round all numbers to 4 decimal places
            best_models_en = best_models_en.round(4)
            best_models_es = best_models_es.round(4)

            # Move the last column two positions to the left
            cols = best_models_en.columns.tolist()
            cols = cols[:4] + cols[-1:] + cols[4:-1]
            best_models_en = best_models_en[cols]
            best_models_es = best_models_es[cols]


            # Concat the two dataframes
            merged_df = pd.concat([best_models_en, best_models_es], ignore_index=True)
            
            # Here we should add the baseline info. 
            
            # Save the merged dataframe
            merged_df.to_csv(f"{OUTPUT_DIR}/exist_2023_{task_id}_best_en_es.csv", index=False)



if __name__ == "__main__":
    main()