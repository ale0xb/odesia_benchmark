import os
import numpy as np

from datasets import load_dataset, load_from_disk, ClassLabel
import torch

from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import pipeline

import evaluate
from sklearn.metrics import f1_score, accuracy_score

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
                                          max_length=self.tokenizer.model_max_length,
                                          return_tensors="pt")
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
    
    # no funciona para evaluar el dataset en modo evall
    def predict(self, split="test"):
        inputs = self.dataset[split]
        classifier = pipeline("ner", model=self.model.to(torch.device("cpu")), tokenizer=self.tokenizer)
        results = []
        for input in inputs:
            input_tokens = input['tokens']

            ner_tags = [None for _ in input_tokens]
            ner_tags_index = [None for _ in input_tokens]

            text = ' '.join(input_tokens)
            classifier_outputs = classifier(text)
            # si clasifica los tokens
            if classifier_outputs:
                # para ese ejemplo vamos token por token
                for classifier_output in classifier_outputs:
                    # sacamos la palabra clasificada
                    start = classifier_output["start"]
                    end = classifier_output["start"]+len(classifier_output["word"].replace("##", ""))
                    tag = classifier_output["entity"]
                    token_result = text[start:end]
                    
                    for i, input_token in enumerate(input_tokens):
                        if token_result == input_token:
                            ner_tags[i] = tag
                            ner_tags_index[i] = self.label2id[tag]

                ner_tags = [ner_tag if ner_tag else 'O' for ner_tag in ner_tags]
                ner_tags_index = [ner_tag_index if ner_tag_index else self.label2id['O'] for ner_tag_index in ner_tags_index]



                result = {"test_case": self.test_case,
                        "id_sentence":input["id_sentence"],
                        "ner_tags":ner_tags, 
                        "ner_tags_index":ner_tags_index}
                                               
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
        f1_scores = f1_score(labels, predictions, average=None).tolist()
        f1_macro =  f1_score(labels, predictions, average="macro").tolist()
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy":accuracy, "f1_macro" : f1_macro,"f1_per_class": f1_scores}

    def predict(self, split="test"):
        results = []
        dataset = self.tokenized_dataset[split]
        pred = self.trainer.predict(dataset)

        
        if self.num_labels <= 2:
            predictions = pred.predictions.argmax(axis=1)
            print(predictions)
        else:
            predictions = (pred.predictions > 0.5).astype(int)

            for i,prediction in enumerate(predictions):
                if self.num_labels <= 2:
                   print(prediction)
                   predicted_labels = self.id2label[int(prediction)]
                else:
                    predicted_labels = []
                    for j,label_id in enumerate(prediction):
                        if label_id == 1:
                            predicted_labels.append(self.id2label[j])
                result = {'test_case':self.test_case,
                            'id':dataset[i]['id'],
                            'label':predicted_labels}
                results.append(result)
        return results
            