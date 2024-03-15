import os
import sys
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
# Additional imports as necessary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Get the directory containing this script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_directory = os.path.dirname(script_directory)
# Add the parent directory to the system path
sys.path.append(parent_directory)

from vendor.exist2023evaluation import ICM_Hard, ICM_Soft
from tqdm import tqdm


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)  # num_classes output units
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Logits for each class
        return x

# Constants
TASK_HIERARCHIES = {
    "t1": None,
    "t2": {"YES":["DIRECT","REPORTED","JUDGEMENTAL"], "NO":[]},
    "t3": {"YES":["IDEOLOGICAL-INEQUALITY","STEREOTYPING-DOMINANCE","OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"], "NO":[]}
}

TASK_TYPES = {
    "t1": "mono_label",
    "t2": "mono_label",
    "t3": "multi_label"
}

N_EPOCHS = 20

def clean_tweet(tweet, language="en"):
    """Clean a single tweet by removing URLs, user mentions, and punctuation."""
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    if language == "es":
        filtered_words = [word for word in tweet_tokens if word not in stopwords.words('spanish')]
    elif language == "en":
        filtered_words = [word for word in tweet_tokens if word not in stopwords.words('english')]
    else:
        raise ValueError("Invalid language. Supported languages are 'es' for Spanish and 'en' for English.")
    return ' '.join(filtered_words)

def preprocess_data(df, language="en"):
    """Apply cleaning to all tweets in a DataFrame and return the processed DataFrame."""
    df['text_clean'] = df['text'].apply(clean_tweet)
    return df



def train_model_soft(X_train, y_train):
    loss_function = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(TensorDataset(
                                    torch.tensor(X_train, dtype=torch.float32), 
                                    torch.tensor(y_train, dtype=torch.float32)), 
                                batch_size=32, shuffle=True)
    # Instantiate the model    
    model = SimpleNN(input_size=len(train_loader.dataset[0][0]), 
                     num_classes=len(train_loader.dataset[0][1])) # This changes wrt the hard model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    print(f"Training SimpleNN in soft mode for task {task}-{language} for {N_EPOCHS} epochs...")
    for epoch in tqdm(range(N_EPOCHS), desc="Epochs"):
        model.train()
        with tqdm(train_loader, unit="batch") as t:
            for inputs, labels in t:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)  # `soft_labels` is your tensor of soft labels
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item())
    return model 
   

def compute_hard_baseline(task, language):
    print(f"Calculating baseline for task {task}-{language} in mode Hard-Hard...")

    train_hard_path = f'datasets/exist_2023_{task}_hard_hard/train_{language}.json'
    test_hard_hard_path = f'datasets/exist_2023_{task}_hard_hard/test_{language}.json'
    test_hard_soft_path = f'datasets/exist_2023_{task}_hard_soft/test_{language}.json'

    df_train_hard = preprocess_data(pd.read_json(train_hard_path), language=language)
    df_test_hard_hard = preprocess_data(pd.read_json(test_hard_hard_path), language=language)
    df_test_hard_soft = preprocess_data(pd.read_json(test_hard_soft_path), language=language)

    # Vectorizing text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_hard = vectorizer.fit_transform(df_train_hard['text_clean']).toarray()
    y_train_hard = df_train_hard['label']

    # Train the model
    trained_model_hard = train_model_hard(X_train_hard, y_train_hard)

    # Evaluate the model
    X_test_hard = vectorizer.transform(df_test_hard_hard['text_clean']).toarray()
    y_test_hard = df_test_hard_hard['label']

    X_test_hard_t = torch.tensor(X_test_hard, dtype=torch.float32)

    trained_model_hard.eval()
    with torch.no_grad():
        logits = trained_model_hard(X_test_hard_t)
        y_pred = torch.argmax(logits, dim=1)

    df_test_hard_hard['predicted_label']  = y_pred.numpy()
    df_test_hard_hard['value'] = df_test_hard_hard['predicted_label'].map({0: 'NO', 1: 'YES'})
    df_labels = df_test_hard_hard[['id', 'label']].rename(columns={'label': 'value'})
    df_labels['value'] = df_labels['value'].map({0: 'NO', 1: 'YES'})

    # First, ICM Hard
    icm_hard = ICM_Hard(df_test_hard_hard[['id', 'value']], df_labels, TASK_TYPES["t1"], TASK_HIERARCHIES["t1"])
    icm_hard_result = icm_hard.evaluate()
    print(f"ICM Hard for task {task}-{language} in mode Hard-Hard: {icm_hard_result}")

    # Second, ICM Soft
    X_test_soft = vectorizer.transform(df_test_hard_soft['text_clean']).toarray()
    X_test_soft_t = torch.tensor(X_test_soft, dtype=torch.float32)
    with torch.no_grad():
        logits = trained_model_hard(X_test_soft_t)
        y_pred = torch.softmax(logits, dim=1)

    df_test_hard_soft['pred_probs'] = [list(p) for p in y_pred.numpy()]
    df_test_hard_soft['pred_probs'] = df_test_hard_soft['pred_probs'].apply(lambda x: {'NO': x[0], 'YES': x[1]})
    # Convert soft label column dictionaries from 'sexist'/'non-sexist' to 'YES'/'NO'
    df_test_hard_soft['soft_label'] = df_test_hard_soft['soft_label'].apply(lambda x: {'NO': x['non-sexist'], 'YES': x['sexist']})

    icm_soft = ICM_Soft(df_test_hard_soft[['id', 'pred_probs']].rename(columns={'pred_probs':'value'}), 
    df_test_hard_soft[['id', 'soft_label']].rename(columns={'soft_label': 'value'}), TASK_TYPES["t1"], TASK_HIERARCHIES["t1"])
    icm_soft_result = icm_soft.evaluate()
    print(f"ICM Soft for task {task}-{language} in mode Hard-Soft: {icm_soft_result}")
    

    # evaluate_model_trained_hard(trained_model, df_test_hard_hard, df_test_hard_soft)

    return {"icm_hard": icm_hard_result, "icm_soft": icm_soft_result}


def train_model_hard(X_train, y_train):
    loss_function = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                            torch.tensor(y_train.values, dtype=torch.int32).view(-1, 1)), 
                              batch_size=32, 
                              shuffle=True)

    # Instantiate the model
    model = SimpleNN(input_size=X_train.shape[1], num_classes=len(y_train.unique()))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        

    print(f"Training SimpleNN in Hard mode for task {task}-{language} for {N_EPOCHS} epochs...")
    for epoch in tqdm(range(N_EPOCHS), desc="Epochs"):
        model.train()
        with tqdm(train_loader, unit="batch") as t:
            for inputs, labels in t:
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.view(-1)  # Reshape labels to be 1D
                loss = loss_function(outputs, labels.long())  # Ensure labels are long type
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item())
    return model 
    

def compute_soft_baseline(task, language):
    print(f"Calculating baseline for task {task}-{language} in mode Soft-Soft...")

    train_soft_path = f'datasets/exist_2023_{task}_soft_soft/train_{language}.json'
    test_soft_soft_path = f'datasets/exist_2023_{task}_soft_soft/test_{language}.json'

    df_train_soft = preprocess_data(pd.read_json(train_soft_path), language=language)
    df_test_soft_soft = preprocess_data(pd.read_json(test_soft_soft_path), language=language)
    
    # Vectorizing text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_soft = vectorizer.fit_transform(df_train_soft['text_clean']).toarray()
    X_test_soft = vectorizer.transform(df_test_soft_soft['text_clean']).toarray()

    df_train_soft['soft_label'] = df_train_soft['label'].apply(lambda x: [x['non-sexist'], x['sexist']])  
    y_train_soft = df_train_soft['soft_label']

    # Train the model
    trained_model_soft = train_model_soft(X_train_soft, y_train_soft)

    # Evaluate the model
    X_test_soft_t = torch.tensor(X_test_soft, dtype=torch.float32)

    trained_model_soft.eval()
    with torch.no_grad():
        y_pred = trained_model_soft(X_test_soft_t)
        y_pred = torch.softmax(y_pred, dim=1)
    
    df_test_soft_soft['pred_probs'] = [list(p) for p in y_pred.numpy()]
    df_test_soft_soft['pred_probs'] = df_test_soft_soft['pred_probs'].apply(lambda x: {'NO': x[0], 'YES': x[1]})
    # Convert soft label column dictionaries

    df_test_soft_soft['soft_label'] = df_test_soft_soft['label'].apply(lambda x: {'NO': x['non-sexist'], 'YES': x['sexist']})
    
    icm_soft = ICM_Soft(df_test_soft_soft[['id', 'pred_probs']].rename(columns={'pred_probs':'value'}),
    df_test_soft_soft[['id', 'soft_label']].rename(columns={'soft_label': 'value'}), TASK_TYPES["t1"], TASK_HIERARCHIES["t1"])
    icm_soft_result = icm_soft.evaluate()
    print(f"ICM Soft for task {task}-{language} in mode Soft-Soft: {icm_soft_result}")
    return {"icm_soft": icm_soft_result}
    

results = {}
if __name__ == "__main__":
    for task in ["t1", "t2", "t3"][0:1]:
        results[task] = {}
        for language in ["en", "es"]:
            icm_results_hard = compute_hard_baseline(task, language)
            icm_results_soft = compute_soft_baseline(task, language)
            # Save results
            results[task][language] = {'hard-hard': icm_results_hard['icm_hard'], 'hard-soft': icm_results_hard['icm_hard'], 'soft-soft': icm_results_soft['icm_soft']}
    
    # Pretty print the results
    print(results)
            

            

            

            




