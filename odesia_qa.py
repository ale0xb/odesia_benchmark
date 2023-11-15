import torch 
from odesia_core import OdesiaHFModel
from transformers import (AutoModelForQuestionAnswering, 
                          pipeline, 
                          DefaultDataCollator)
from evaluate import load


class OdesiaQuestionAnswering(OdesiaHFModel):
    
    def __init__(self, model_path, dataset_path, model_config, dataset_config):                
        super().__init__(model_path, dataset_path, model_config, dataset_config)
        
        # Step 1. Load DataCollator            
        self.data_collator = DefaultDataCollator()      

        # Step 2. Tokenized the dataset 
        if not self.tokenized_dataset:            
            self.tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset["train"].column_names)
            self.tokenized_dataset.save_to_disk(self.dataset_path_tokenized)

        # Step 3. Loading model, trainer and metrics     
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)   
        self.trainer = self.load_trainer(model=self.model, 
                                         data_collator=self.data_collator, 
                                         tokenized_dataset=self.tokenized_dataset, 
                                         compute_metrics_function=None)
        self.metric = load("squad")
        

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
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
            while sequence_ids[idx] == 1:
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
        results_prediction = self.predict(split, return_references=True)
        predictions = []
        # eliminamos todo de las predicciones, menos las claves que nos hacen falta
        for prediction in results_prediction['predictions']:
            prediction = {key: value for key, value in prediction.items() if key in ['id', 'prediction_text']}
            predictions.append(prediction)
        results = self.metric.compute(predictions=predictions, references=results_prediction['references'])
        return results
    
    def predict(self, split="test", num_examples = "max", return_references=False):
        
        num_examples = len(self.dataset[split]) if num_examples == "max" else num_examples
        
        predictions_dataset = self.dataset[split].select(range(num_examples))
        question_answerer = pipeline("question-answering", model=self.model.to(torch.device("cpu")), tokenizer=self.tokenizer)
        
        
        num_predictions = len(predictions_dataset)
        predictions = []
        references = []
        for i,item in enumerate(predictions_dataset):            
            
            result = question_answerer(question=item["question"], context=item['context'])                        
            result['prediction_text'] = result['answer']
            del result['answer']
            result['id'] = item["id"]
            
            #if i%100 == 0:
            #    print(f"Generation prediction ({i} of {num_predictions}){result}")
            
            predictions.append(result)    
            references.append({'answers' : item['answers'], "id":item["id"]})
        if return_references:
            return {'predictions': predictions,
                    'references' : references} 
        else:
            return predictions
                
    def compute_metrics(self):
        return None
