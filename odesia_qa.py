import os
import torch 
from odesia_core import OdesiaHFModel
from transformers import (AutoModel,
                          AutoModelForQuestionAnswering, 
                          pipeline, 
                          DefaultDataCollator)
from peft import LoraConfig, get_peft_model
from evaluate import load
from odesia_configs import PEFT_TASK_MAPPING


class OdesiaQuestionAnswering(OdesiaHFModel):
    
    def __init__(self, model_path, dataset_path, model_config, dataset_config):                
        super().__init__(model_path, dataset_path, model_config, dataset_config)
        
        # Step 1. Load DataCollator            
        self.data_collator = DefaultDataCollator()

        if self.peft_parameters:
            self.tokenizer.model_max_length = 1024

        # Step 2. Tokenized the dataset 
        if not self.tokenized_dataset:            
            self.tokenized_dataset = self.dataset.map(self.preprocess_function, 
                                                      batched=True, 
                                                      remove_columns=self.dataset["train"].column_names)
            self.tokenized_dataset.save_to_disk(self.dataset_path_tokenized)

        # Step 3. Loading model, trainer and metrics     
        self.model = AutoModel.from_pretrained(model_path, torch_dtype="auto")

        if 'Llama' in model_path:
            # Hack for bug https://github.com/huggingface/transformers/issues/30381
            # Create folder for pretrained models if it does not exist
            print("Saving and reloading Llama model to avoid bug")
            if not os.path.exists("pretrained_models"):
                os.makedirs("pretrained_models")
            save_route = f"pretrained_models/{model_path.split('/')[-1]}-base_model"
            self.model.save_pretrained(save_route)
            self.model = AutoModelForQuestionAnswering.from_pretrained(save_route, torch_dtype="auto")
        if self.peft_parameters:
        ## This is a PEFT model 
            self.model = self.convert_model_to_PEFT(self.model)  
        self.trainer = self.load_trainer(model=self.model, 
                                         data_collator=self.data_collator, 
                                         tokenized_dataset=self.tokenized_dataset, 
                                         compute_metrics_function=None)
        self.metric = load("squad")
        self.predictions = {}
    
    def convert_model_to_PEFT(self, model):
        print("Convert model to PEFT")
        lora_config = LoraConfig(**self.peft_parameters)
        lora_config.task_type = PEFT_TASK_MAPPING[self.problem_type]
        model = get_peft_model(model, lora_config)
        model.config.pretraining_tp = 1
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model
    

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.tokenizer.model_max_length - 128,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def evaluate(self, split="val"):
        if split in self.predictions:
            results_prediction = self.predictions[split]
        else:
            results_prediction = self.predict(split=split, num_examples = 500, return_references=True)
        predictions = []
        # eliminamos todo de las predicciones, menos las claves que nos hacen falta
        for prediction in results_prediction['predictions']:
            # renombrar prediction a label
            prediction = {key: value for key, value in prediction.items() if key in ['id', 'prediction_text']}
            predictions.append(prediction)
        results = self.metric.compute(predictions=predictions, references=results_prediction['references'])
        return results
    
    def predict(self, split="test", num_examples="max", return_references=False):
        len_num_example = len(self.dataset[split])
        num_examples = len_num_example if num_examples == "max" or num_examples > len_num_example else num_examples        
        predictions_dataset = self.dataset[split].select(range(num_examples))
        
        num_predictions = len(predictions_dataset)
        predictions = []
        references = []
        
        for i, item in enumerate(predictions_dataset):
            inputs = self.tokenizer(
                item["question"],
                item["context"],
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax() + 1
            
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start_index:answer_end_index])
            )
            
            result = {
                'prediction_text': answer,
                'id': item["id"]
            }
            
            if i % 200 == 0:
                print(f"Generation prediction ({i} of {num_predictions}) {result}")
            
            predictions.append(result)
            references.append({'answers': item['answers'], "id": item["id"]})
        
        if return_references:
            return {"predictions": predictions, "references": references}
        return {"predictions": predictions}
                
    def compute_metrics(self):
        return None
