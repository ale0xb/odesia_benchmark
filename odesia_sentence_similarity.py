from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import math

from odesia_core import OdesiaHFModel


class OdesiaSentenceSimilarity(OdesiaHFModel):
    
    def __init__(self, model_path, dataset_path, model_config, dataset_config):                
        super().__init__(model_path, dataset_path, model_config, dataset_config)
                
        # Step 2. Tokenized the dataset            
        self.max_score = self.dataset_config["max_score"]
        self.tokenized_dataset = self.preprocess_function(self.dataset, 
                                                         self.max_score)#, batched=True, remove_columns=self.dataset["train"].column_names)
        # Step 3. Loading model, trainer and metrics     
        self.model = SentenceTransformer(model_path)
        self.train_dataloader = DataLoader(self.tokenized_dataset['train'], shuffle=True, batch_size=model_config['hf_parameters']['per_device_train_batch_size'])
        del model_config['hf_parameters']['per_device_train_batch_size']
        self.train_loss = losses.CosineSimilarityLoss(model=self.model)
        self.evaluator = EmbeddingSimilarityEvaluator.from_input_examples(self.tokenized_dataset['val'], name='sts-dev')
        
        # Mapping from hf_params to SenteceTransformers Params   
        if self.model_config['hf_parameters']['learning_rate']:
            self.model_config['hf_parameters']['optimizer_params'] = {'lr':self.model_config['hf_parameters']['learning_rate']}
            del self.model_config['hf_parameters']['learning_rate']
        if self.model_config['hf_parameters']['num_train_epochs']:
            self.model_config['hf_parameters']['epochs'] = self.model_config['hf_parameters']['num_train_epochs']
            del self.model_config['hf_parameters']['num_train_epochs']
            
        self.model_config['hf_parameters']['warmup_steps'] = math.ceil(len(self.train_dataloader) * self.model_config['hf_parameters']['epochs'] * 0.1) #10% of train data for warm-up
        
    
    def preprocess_function(self, dataset, max_score=5.0):         
        tokenized_dataset = {}
        for split in dataset:
            tokenized_dataset[split] = []
            for row in dataset[split]:
                score = float(row['similarity_score']) / max_score  # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
                tokenized_dataset[split].append(inp_example)
        return tokenized_dataset
    
    def train(self):
        self.model.fit(train_objectives=[(self.train_dataloader, self.train_loss)],
              evaluator=self.evaluator,
              output_path=self.output_dir,
              **self.model_config['hf_parameters'])
    
    def evaluate(self, split="val"):
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(self.tokenized_dataset[split], name='sts-test')
        return test_evaluator(self.model, output_path=self.output_dir)
    
    def predict(self, split="test"):
        examples = self.tokenized_dataset[split]
        predictions = []
        for example in examples:
            text1, text2 = example.texts
            
            # Obtener las incrustaciones (embeddings) para los textos
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)

            # Calcular la puntuación de similitud utilizando el coseno de los embeddings
            similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

            # Imprimir o almacenar la similitud según sea necesario
            predictions.append({
                            "sentence1": text1,
                            "sentence2": text2,
                            "similarity_score": similarity_score*self.max_score,
                            "test_case":self.test_case
                        })
            # Agregar la similitud como una nueva propiedad a los ejemplos si lo deseas
            # example.similarity = similarity_score
        return predictions 