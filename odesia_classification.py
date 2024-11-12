import os
import numpy as np

from datasets import ClassLabel

from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import pipeline
from transformers import Trainer, TrainingArguments

from torch.nn import BCEWithLogitsLoss

import evaluate
from sklearn.metrics import f1_score, accuracy_score

from odesia_core import OdesiaHFModel

import pandas as pd

from vendor.exist2023evaluation import ICM_Hard, ICM_Soft




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
        
        # Load DataCollator
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Tokenize the dataset
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.dataset.map(self.tokenize_and_align_labels, batched=False)
            self.tokenized_dataset.save_to_disk(self.dataset_path_tokenized)
        
        # Load model, trainer, and metrics
        self.seqeval = evaluate.load("seqeval")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path, num_labels=self.num_labels, id2label=self.id2label, label2id=self.label2id
        )
        
        self.trainer = self.load_trainer(model=self.model, 
                                         data_collator=self.data_collator, 
                                         tokenized_dataset=self.tokenized_dataset, 
                                         compute_metrics_function=self.compute_metrics)
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

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

    def tokenize_and_align_labels(self, example, label_all_tokens=True):
        texts = example["tokens"]
        tags = example.get("ner_tags", None)
        
        tokenized_inputs = self.tokenizer(
            texts,
            is_split_into_words=True,
            padding='max_length',  # Añade padding hasta la longitud máxima
            truncation=True,
            return_offsets_mapping=True
        )

        word_ids = tokenized_inputs.word_ids()

        if tags is not None:
            raw_labels = [self.label2id.get(tag, -100) for tag in tags]
            labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(-100)
                elif word_idx != previous_word_idx:
                    labels.append(raw_labels[word_idx])
                else:
                    labels.append(raw_labels[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            tokenized_inputs["labels"] = labels

        tokenized_inputs.pop("offset_mapping")
        tokenized_inputs["original_tokens"] = texts
        tokenized_inputs["word_ids"] = word_ids

        return tokenized_inputs

    def predict(self, split="test", examples=30):
        dataset_split = self.tokenized_dataset[split]
        dataset_subset = dataset_split#.select(range(min(len(dataset_split), examples)))
        
        predictions, labels, _ = self.trainer.predict(dataset_subset)
        
        predictions = np.argmax(predictions, axis=2)
        aligned_predictions = []

        for i, example in enumerate(dataset_subset):
            tokens = example['original_tokens']
            word_ids = example['word_ids']
            predicted_labels = [self.label_list[p] for p in predictions[i]]
            aligned_tokens = []
            aligned_labels = []
            previous_word_idx = None

            for idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == previous_word_idx:
                    continue
                aligned_tokens.append(tokens[word_idx])
                aligned_labels.append(predicted_labels[idx])
                previous_word_idx = word_idx
            
            aligned_tokens, aligned_labels = self.adjust_alignment(tokens, aligned_tokens, aligned_labels)

            def list_difference(list1, list2):
                set1 = set(list1)
                set2 = set(list2)
                
                difference1 = set1 - set2  # Elementos en list1 pero no en list2
                difference2 = set2 - set1  # Elementos en list2 pero no en list1
                
                return list(difference1), list(difference2)

            # Verificación de longitud
            if len(aligned_tokens) != len(tokens):
                print(f"Error en el alineamiento: {tokens}")
                print(f"Tokens alineados: {aligned_tokens}")
                print(f"Etiquetas alineadas: {aligned_labels}")
                raise ValueError(f"Mismatch in lengths: {len(tokens)} tokens vs {len(aligned_tokens)} tokens")

            
            # Verificar que las longitudes de tokens y etiquetas sean iguales
            if len(aligned_tokens) != len(aligned_labels):
                raise ValueError(f"Mismatch in lengths: {len(aligned_tokens)} tokens vs {len(aligned_labels)} labels")

            aligned_predictions.append({
                'id': example['id'],
                'ner_tags': aligned_labels,
                'tokens': aligned_tokens
            })

        # Verificar que el número de predicciones y referencias sea igual
        if len(aligned_predictions) != len(dataset_subset):
            raise ValueError(f"Mismatch in number of predictions: {len(aligned_predictions)} vs references: {len(dataset_subset)}")

        return aligned_predictions
    
    def adjust_alignment(self, tokens, aligned_tokens, aligned_labels):
        idx_orig = 0
        idx_aligned = 0
        
        while idx_orig < len(tokens) and idx_aligned < len(aligned_tokens):
            if tokens[idx_orig] != aligned_tokens[idx_aligned]:
                # Identificar si se está entre B-algo e I-algo
                if idx_aligned > 0 and idx_aligned < len(aligned_labels) - 1:
                    prev_label = aligned_labels[idx_aligned - 1]
                    next_label = aligned_labels[idx_aligned]
                    
                    if prev_label.startswith('B-') and next_label.startswith('I-') and prev_label[2:] == next_label[2:]:
                        aligned_tokens.insert(idx_aligned, tokens[idx_orig])
                        aligned_labels.insert(idx_aligned, f'I-{prev_label[2:]}')
                    else:
                        aligned_tokens.insert(idx_aligned, tokens[idx_orig])
                        aligned_labels.insert(idx_aligned, 'O')
                else:
                    aligned_tokens.insert(idx_aligned, tokens[idx_orig])
                    aligned_labels.insert(idx_aligned, 'O')
            
            idx_orig += 1
            idx_aligned += 1
        
        # Añadir los tokens y etiquetas restantes
        while idx_orig < len(tokens):
            aligned_tokens.append(tokens[idx_orig])
            aligned_labels.append('O')
            idx_orig += 1

        return aligned_tokens, aligned_labels

    
    
    
class OdesiaTextClassification(OdesiaUniversalClassification):
    
    def __init__(self, model_path, dataset_path, model_config, dataset_config):                
        super().__init__(model_path, dataset_path, model_config, dataset_config)
         # Load DataCollator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        

    def setup(self):
    
        self.tokenize_dataset() 
        self.initialize_model()
        self.setup_trainer()

    
    def tokenize_dataset(self):
         # This method is used to setup the model and the trainer. Needs to be called after instantiating the class. 
        column_names = self.dataset['train'].column_names
        if 'label' in column_names and self.dataset_config["label_column"].strip() != 'label':
            self.dataset = self.dataset.rename_column('label', "label_old")        
        self.dataset = self.dataset.rename_column(self.dataset_config["label_column"], "label")

        if not self.tokenized_dataset:
            if 'multi_label_classification' not in self.problem_type:
                self.dataset = self.dataset.cast_column('label', ClassLabel(names=self.label_list))
            self.tokenized_dataset = self.dataset.map(lambda ex: self.tokenizer(ex["text"], truncation=True, padding=True), batched=True)
            self.tokenized_dataset.save_to_disk(self.dataset_path_tokenized)
    
    def initialize_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            num_labels=self.num_labels, 
            problem_type='multi_label_classification' if 'multi_label_classification' in self.problem_type else None
        )

    def setup_trainer(self):
        self.trainer = self.load_trainer(
            model=self.model, 
            data_collator=self.data_collator, 
            tokenized_dataset=self.tokenized_dataset, 
            compute_metrics_function=self.compute_metrics
        )

    def convert_predictions(self, pred):
        if 'multi_class_classification' in self.problem_type:
            predictions = np.argmax(pred.predictions, axis=-1)
        elif 'multi_label_classification' in self.problem_type:
            probs = 1 / (1 + np.exp(-pred.predictions))
            predictions = (probs > 0.5).astype(int)
        else:
            raise ValueError(f"Problem type {self.problem_type} not supported")
        return predictions
                

    def compute_metrics(self, pred):
        labels = pred.label_ids
        predictions = self.convert_predictions(pred)

        f1_scores = f1_score(labels, predictions, average=None).tolist()
        f1_macro = f1_score(labels, predictions, average="macro").tolist()
        accuracy = accuracy_score(labels, predictions)

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_per_class": {label_f1: f1_value for label_f1, f1_value in zip(self.label_list, f1_scores)}
        }

    def predict(self, split="test"):
        results = []
        dataset = self.tokenized_dataset[split]
        pred = self.trainer.predict(dataset)
        predictions = self.convert_predictions(pred)

        for i, prediction in enumerate(predictions):
            if 'multi_class_classification' in self.problem_type:
                predicted_labels = self.id2label[int(prediction)]
            elif 'multi_label_classification' in self.problem_type:
                predicted_labels = [self.id2label[j] for j, label_id in enumerate(prediction) if label_id == 1]
            else:
                raise ValueError(f"Problem type {self.problem_type} not supported")
        
            result = {
                'test_case': self.test_case,
                'id': dataset[i]['id'],
                'label': predicted_labels
            }
            results.append(result)
        return {split: results}

    
class OdesiaTextClassificationWithDisagreements(OdesiaTextClassification):
    
    def __init__(self, model_path, dataset_path, model_config, dataset_config):                
        super().__init__(model_path, dataset_path, model_config, dataset_config)
        # Save hierarchy and task
        self.hierarchy = dataset_config.get("hierarchy", None)
        self.exist_task = dataset_config.get("exist_task", None)   
        self.training_mode = dataset_config.get("training_mode", None)
        self.eval_mode = dataset_config.get("eval_mode", None)
    
    def tokenize_dataset(self):
        __this__ = self 
        if self.training_mode == 'soft':
            # def preprocess_labels(example):
            #     # Convert label dict to list of probabilities in the correct order
            #     example["labels"] = [example["label"][label] for label in __this__.label_list]
            #     return example
            
            if not self.tokenized_dataset:
                # self.dataset = self.dataset.map(preprocess_labels, batched=False)  # Ensure labels are processed
                self.tokenized_dataset = self.dataset.map(
                    lambda ex: self.tokenizer(ex["text"], truncation=True, padding=True), 
                    batched=True
                )
                self.tokenized_dataset.save_to_disk(self.dataset_path_tokenized)
        else: # Resort to the parent class method
            super().tokenize_dataset()
    
    def initialize_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            num_labels=self.num_labels,
            # Set problem type to multi_label_classification if we are using soft training mode
            problem_type='multi_label_classification' if ('multi_label_classification' in self.problem_type or self.training_mode=='soft') else None
        )

    def setup_trainer(self):
        self.trainer = self.load_trainer(
            model=self.model, 
            data_collator=self.data_collator, 
            tokenized_dataset=self.tokenized_dataset, 
            compute_metrics_function=self.compute_metrics
        )

    def convert_predictions(self, pred):
        if 'multi_class_classification' in self.problem_type:
            predictions = np.argmax(pred.predictions, axis=-1)
        elif 'multi_label_classification' in self.problem_type:
            probs = 1 / (1 + np.exp(-pred.predictions))
            predictions = (probs > 0.5).astype(int)
        else:
            raise ValueError(f"Problem type {self.problem_type} not supported")
        return predictions
    
    def predict(self, split="test"):
        results = []
        dataset = self.tokenized_dataset[split]
        pred = self.trainer.predict(dataset) 
        if self.exist_task == 'multi_label':
            probs = 1 / (1 + np.exp(-pred.predictions))
        else:
            # Apply softmax to the predictions only if we are in monolabel 
            probs = np.exp(pred.predictions) / np.exp(pred.predictions).sum(-1, keepdims=True)
        predictions = probs
        for i, prediction in enumerate(predictions):
            predicted_labels = {self.id2label[j]: label_id for j, label_id in enumerate(prediction)}
            result = {
                'test_case': self.test_case,
                'id': dataset[i]['id'],
                'value': predicted_labels
            }
            results.append(result)
        return {split: results}

    def compute_metrics(self, pred):
        if self.training_mode == 'soft': # Only compute the icm_soft
            labels = pred.label_ids
            # Get the soft labels for the predictions
            if self.exist_task == 'multi_label':
                probs = 1 / (1 + np.exp(-pred.predictions))
            else:
                # Apply softmax to the predictions only if we are in monolabel 
                probs = np.exp(pred.predictions) / np.exp(pred.predictions).sum(-1, keepdims=True)
            converted_rows = []
            for row in probs:
                    converted_row = {}
                    for i, value in enumerate(row):
                        converted_row[self.label_list[i]] = value
                    converted_rows.append(converted_row)
            predictions_df = pd.DataFrame({'value': converted_rows})

            converted_labels = []
            for row in labels:
                converted_row = {}
                for i, value in enumerate(row):
                    converted_row[self.label_list[i]] = value
                converted_labels.append(converted_row)
            labels_df = pd.DataFrame({'value': converted_labels})

            # Create column 'id' for predictions_df and labels_df
            predictions_df['id'] = predictions_df.index
            labels_df['id'] = labels_df.index

            # Compute ICM_Soft (needs to be converted into pandas dataframe)
            icm_soft = ICM_Soft(predictions_df, labels_df, self.exist_task, self.hierarchy)
            icm_soft_result = icm_soft.evaluate()
            ## Add results to base_metrics
            return {
                'icm_soft': icm_soft_result
            }

        else: 
            base_metrics = super().compute_metrics(pred)
        
            if self.eval_mode == 'hard':
                labels = pred.label_ids
                predictions = self.convert_predictions(pred)

                # Convert to pandas dataframe
                if 'multi_class_classification' in self.problem_type:
                    predictions_df = pd.DataFrame([self.id2label[pred] for pred in predictions], columns=['value'])
                    labels_df = pd.DataFrame([self.id2label[label] for label in labels], columns=['value'])
                else:
                    transformed_predictions = []
                    for pred in predictions:
                        transformed_predictions.append([self.id2label[i] for i, value in enumerate(pred) if value > 0])
                    predictions_df = pd.DataFrame({'value': transformed_predictions})
                    transformed_labels = []
                    for label in labels:
                        transformed_labels.append([self.id2label[i] for i, value in enumerate(label) if value > 0])
                    labels_df = pd.DataFrame({'value': transformed_labels})
                
                # Create column 'id' for predictions_df and labels_df
                predictions_df['id'] = predictions_df.index
                labels_df['id'] = labels_df.index
                # Compute ICM_Hard (needs to be converted into pandas dataframe)
                icm_hard =  ICM_Hard(predictions_df, labels_df, self.exist_task, self.hierarchy)
                icm_hard_result = icm_hard.evaluate()
                ## Add results to base_metrics
                base_metrics['icm_hard'] = icm_hard_result
            else:
                # WARN: A bit hacky but...
                # Get the dataset we are using to evaluate by comparing the size of pred.predictions to the size of all datasets in self.dataset
                dataset_keys = list(self.dataset.keys())
                dataset_sizes = [len(self.dataset[split]) for split in dataset_keys]
                dataset_index = dataset_sizes.index(pred.predictions.shape[0])
                gold_soft_labels = self.dataset[dataset_keys[dataset_index]]['soft_label']
                # Get the soft labels for the predictions
                if self.exist_task == 'multi_label':
                    probs = 1 / (1 + np.exp(-pred.predictions))
                else:
                    # Apply softmax to the predictions only if we are in monolabel 
                    probs = np.exp(pred.predictions) / np.exp(pred.predictions).sum(-1, keepdims=True)

                # Now map the probabilities to the labels
                converted_rows = []
                for row in probs:
                    converted_row = {}
                    for i, value in enumerate(row):
                        converted_row[self.label_list[i]] = value
                    converted_rows.append(converted_row)
                
                predictions_df = pd.DataFrame({'value': converted_rows})
                labels_df = pd.DataFrame({'value': gold_soft_labels})

                predictions_df['id'] = predictions_df.index 
                labels_df['id'] = labels_df.index
                # Convert 
                icm_soft = ICM_Soft(predictions_df, labels_df, self.exist_task, self.hierarchy)
                icm_soft_result = icm_soft.evaluate()
                ## Add results to base_metrics
                base_metrics['icm_soft'] = icm_soft_result
        
            return base_metrics
    
    def load_trainer(self, model, tokenized_dataset, data_collator, compute_metrics_function):
        if self.training_mode == 'hard':      
            return super().load_trainer(model, tokenized_dataset, data_collator, compute_metrics_function)
        else:
            class DisagreementSoftTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = BCEWithLogitsLoss()
                    # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))
                    loss = loss_fct(logits, labels.float())  # Ensure labels are float for BCEWithLogitsLoss
                    return (loss, outputs) if return_outputs else loss
                
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                run_name=self.output_dir,
                overwrite_output_dir=True,
                **self.model_config['hf_parameters']
            )

            trainer = DisagreementSoftTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["val"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_function,
            )
            return trainer
