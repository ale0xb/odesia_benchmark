import pathlib as pl
import pandas as pd
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file_path', type=str, help='Path to the training file')
    parser.add_argument('-d', '--dev_file_path', type=str, help='Path to the dev file')
    parser.add_argument('-s', '--test_file_path', type=str, help='Path to the test file')
    parser.add_argument('-g', '--gold_folder', type=str, help='Path to the gold folder')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder')

    return parser.parse_args()

def load_split_data(train_file_path, dev_file_path, test_file_path):
    return {'train': pd.read_json(train_file_path, orient='index'), 
            'dev': pd.read_json(dev_file_path, orient='index'), 
            'test': pd.read_json(test_file_path, orient='index')}

def load_gold_data(gold_folder):
    """
    Load gold data from the specified folder.

    Parameters:
    gold_folder (str): The path to the folder containing the gold data files.

    Returns:
    dict: A dictionary containing the loaded gold data.
          The keys of the dictionary are the task numbers (1, 2, 3).
          The values are nested dictionaries, where the keys are the gold types ('hard', 'soft')
          and the values are pandas DataFrames containing the gold data.
    """
    gold_data = {}
    for task in range(1, 4):
        gold_data[task] = {}
        for gold_type in ['hard', 'soft']:
            train_gold_file_path = pl.Path(gold_folder) / f'EXIST2023_training_task{task}_gold_{gold_type}.json'
            dev_gold_file_path = pl.Path(gold_folder) / f'EXIST2023_dev_task{task}_gold_{gold_type}.json'
            test_gold_file_path = pl.Path(gold_folder) / f'EXIST2023_test_task{task}_gold_{gold_type}.json'
            gold_df = pd.concat([pd.read_json(train_gold_file_path, orient='index'), 
                                 pd.read_json(dev_gold_file_path, orient='index'), 
                                 pd.read_json(test_gold_file_path, orient='index')])
            gold_data[task][gold_type] = gold_df
    return gold_data

def join_gold_data_to_split_data(split_data, gold_data):
    """
    Joins gold data to split data.

    Args:
        split_data (dict): A dictionary containing split data.
        gold_data (dict): A dictionary containing gold data.

    Returns:
        dict: A dictionary containing the joined split data.

    """
    for task in range(1, 4):
        for gold_type in ['hard', 'soft']:
            for split, data in split_data.items():
                prefix = f'task{task}_'
                new_df = gold_data[task][gold_type].add_prefix(prefix)
                overlap = set(data.columns) & set(new_df.columns)
                if overlap:
                    data = data.join(new_df, lsuffix='_left', rsuffix='_right')
                else:
                    data = data.join(new_df)
                split_data[split] = data
    return split_data

def save_data_to_json(split_data, output_folder):
    """
    Save split data to JSON files.

    Args:
        split_data (dict): A dictionary containing the split data.
        output_folder (str): The path to the output folder.

    Returns:
        None
    """
    for task in range(1, 4):
        task_output_folder = pl.Path(output_folder, f'exist_2023_t{task}')
        task_output_folder.mkdir(parents=True, exist_ok=True)
        for split, data in split_data.items():
            for lang in ['en', 'es']:
                lang_data = data[data['lang'] == lang]
                task_columns = [col for col in lang_data.columns if f'task{task}_' in col] + ['id_EXIST', 'tweet']
                task_data = lang_data[task_columns]
                task_data = task_data.rename(columns={col: col.replace(f'task{task}_', '') for col in task_columns})
                task_data = task_data.rename(columns={'id_EXIST': 'id', 'tweet': 'text'})
                output_file_path = task_output_folder / f'{split}_{lang}.json' 
                task_data.to_json(output_file_path, orient='records', lines=True, force_ascii=False, indent=4)

def main():
    args = parse_arguments()

    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path
    test_file_path = args.test_file_path
    gold_folder = args.gold_folder
    output_folder = args.output_folder

    # Print the arguments
    print(f'train_file_path: {train_file_path}')
    print(f'dev_file_path: {dev_file_path}')
    print(f'test_file_path: {test_file_path}')
    print(f'gold_folder: {gold_folder}')
    print(f'output_folder: {output_folder}')
    

    pl.Path(output_folder).mkdir(parents=True, exist_ok=True)

    split_data = load_split_data(train_file_path, dev_file_path, test_file_path)
    gold_data = load_gold_data(gold_folder)
    split_data = join_gold_data_to_split_data(split_data, gold_data)
    save_data_to_json(split_data, output_folder)

if __name__ == "__main__":
    main()