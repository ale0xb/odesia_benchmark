from abc import ABC, abstractmethod
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
import os 
from datasets import load_from_disk
from datasets import load_dataset
import copy
 
class OdesiaAbstractModel(ABC):
    @abstractmethod
    def __init__(self, model_path, dataset_path, model_config, dataset_config):        
        pass 

    @abstractmethod
    def load_trainer(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def compute_metrics(self):
        pass

class OdesiaHFModel(OdesiaAbstractModel):

    def __init__(self, model_path, dataset_path, model_config, dataset_config):
        super().__init__(model_path, dataset_path, model_config, dataset_config)
        
        # Basic configs
        self.model_config = copy.copy(model_config)
        self.dataset_config = copy.copy(dataset_config)
        self.model_path = model_path
        self.dataset_path = dataset_path        
        self.output_dir = model_config['output_dir']
        self.test_case = dataset_config['evall_test_case']
        self.problem_type = dataset_config['problem_type']
        
        # Tokenizer
        
        
        # Load dataset if it was tokenized before
        self.dataset = load_dataset('json', data_files=dataset_path)           
        self.dataset_path_tokenized = "/".join(dataset_path['train'].split('/')[:-1])+"/tokenized_"+model_path.replace("/","-")
        if os.path.isdir(self.dataset_path_tokenized):
            print("Loading pretokenized dataset...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
            self.tokenized_dataset = None
            self.tokenized_dataset = load_from_disk(self.dataset_path_tokenized)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True) 
            self.tokenized_dataset = None

    def load_trainer(self, model, tokenized_dataset, data_collator, compute_metrics_function):        
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            run_name=self.output_dir,
            overwrite_output_dir=True,
            **self.model_config['hf_parameters']
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_function,
        )
        return trainer

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.output_dir+'/model')

    def predict(self, split="test"):
        return self.trainer.predict(self.tokenized_dataset[split])
    
    def evaluate(self, split="val"):
        return self.trainer.evaluate(eval_dataset=self.tokenized_dataset[split])
    
    def purge_model(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def compute_metrics(self):
        return 
