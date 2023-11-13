import os
import numpy as np

from datasets import load_dataset, load_from_disk, ClassLabel
import torch

from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import pipeline

import evaluate
from sklearn.metrics import f1_score

from odesia_core import OdesiaHFModel


class OdesiaUniversalClassification(OdesiaHFModel):
    def __init__(self, model_path, dataset_path, model_config, dataset_config):
        super().__init__(model_path, dataset_path, model_config, dataset_config)
        
        # processing labels
        self.label2id = dataset_config['label2id']
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.label_list = list(self.label2id.keys())
        self.num_labels = len(dataset_config['label2id'])

class OdesiaTokenClassification(OdesiaUniversalClassification):

    def __init__(self, model_path, dataset_path, model_config, dataset_config):
        super().__init__(model_path, dataset_path, model_config, dataset_config)
               
        # Step 1. Load DataCollator            
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Step 2. Tokenized the dataset
        if not self.tokenized_dataset:           
            self.tokenized_dataset = self.dataset.map(self.tokenize_and_align_labels, batched=True)
            self.tokenized_dataset.save_to_disk(self.dataset_path_tokenized)
           
        # Step 3. Loading model, trainer and metrics
        self.seqeval = evaluate.load("seqeval")

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path, num_labels=self.num_labels, id2label=self.id2label, label2id=self.label2id
        )        

        self.trainer = self.load_trainer(model=self.model, 
                                         data_collator=self.data_collator, 
                                         tokenized_dataset=self.tokenized_dataset, 
                                         compute_metrics_function=self.compute_metrics)
        

        
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], 
                                          is_split_into_words=True, 
                                          padding='max_length', 
                                          truncation=True, 
                                          max_length=self.tokenizer.model_max_length)

        labels = []
        for i, label in enumerate(examples[f"ner_tags_index"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            label_ids = label_ids
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs
    
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def map_predict_output_to_evall(text, result):
        '''
        Hay varios problemas que solventar.
        1) Pasar de texto a token
        2) La salida anota subtokens
        '''
        return
    def predict(self, split="test"):
        inputs = self.dataset[split]
        classifier = pipeline("ner", model=self.model.to(torch.device("cpu")), tokenizer=self.tokenizer)
        results = []
        for input in inputs:
            text = ' '.join(input['tokens'])
            result = classifier(text)
            if result:
                print("eeeeeeeeeeeee",result)
                result = result[0] 
                start = result["start"]
                end = result["start"]+len(result["word"].replace("##", ""))
                tag = result["entity"]

                print (f"Entity: {tag}, Start:{start}, End:{end}, Token:{text[start:end]}")           
                results.append(result)                       
        return results
    
    
class OdesiaTextClassification(OdesiaUniversalClassification):
    
    def __init__(self, model_path, dataset_path, model_config, dataset_config):                
        super().__init__(model_path, dataset_path, model_config, dataset_config)
        
        # Step 1. Load DataCollator            
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)        

        # Step 2. Tokenized the dataset
        self.dataset = self.dataset.rename_column(dataset_config["label_column"], "label")
        if not self.tokenized_dataset:            
            # para clasificación binaria
            if self.dataset_config["hf_parameters"] and not self.dataset_config["hf_parameters"]["problem_type"]:
                self.dataset = self.dataset.cast_column('label', ClassLabel(names=self.label_list))
            self.tokenized_dataset = self.dataset.map(lambda ex: self.tokenizer(ex["text"], truncation=True, padding=True), batched=True)
            self.tokenized_dataset.save_to_disk(self.dataset_path_tokenized)

        # Step 3. Loading model, trainer and metrics
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=self.num_labels, 
            problem_type=self.dataset_config["hf_parameters"]["problem_type"]
        )
        self.trainer = self.load_trainer(model=self.model, 
                                         data_collator=self.data_collator, 
                                         tokenized_dataset=self.tokenized_dataset, 
                                         compute_metrics_function=self.compute_metrics)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        if self.num_labels <= 2:
            predictions = pred.predictions.argmax(axis=1)
        else:
            predictions = (pred.predictions > 0.5).astype(int)
        f1_scores = f1_score(labels, predictions, average=None)
        return {"f1_per_class": f1_scores.tolist()}
