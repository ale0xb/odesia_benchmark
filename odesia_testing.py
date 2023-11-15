import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from odesia_classification import OdesiaTextClassification, OdesiaTokenClassification
from odesia_qa import OdesiaQuestionAnswering
from odesia_sentece_similarity import OdesiaSentenceSimilarity


def test_token_classification():
    dataset_config = {
        "label2id":{'B-GRP': 0, 'B-CW': 1, 'I-PER': 2, 'I-CW': 3, 'B-CORP': 4, 'I-CORP': 5, 'I-LOC': 6,
            'I-PROD': 7, 'B-LOC': 8, 'I-GRP': 9, 'B-PROD': 10, 'O': 11, 'B-PER': 12}
    }
    model_config = {
        "output_dir" : "/data/gmarco/predicciones-leaderboard/my_awesome_wnut_model",
        "device": 3,
        "hf_parameters": {
                'learning_rate':2e-5,
                'per_device_train_batch_size':512,
                'per_device_eval_batch_size':256,
                'num_train_epochs':1,
                'weight_decay':0.01,
                'evaluation_strategy':"epoch",
                'save_strategy':"epoch",
                'load_best_model_at_end':True}
    }

    clssifierToken = OdesiaTokenClassification(model_path="distilbert-base-uncased",
                                                dataset_path={'train': 'datasets/multicornell_2022/en_train.json', 
                                                            'test':'datasets/multicornell_2022/en_test.json'},
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)
    #clssifierToken.train()
    print(clssifierToken.predict(split="test"))

def test_text_classification():
    dataset_config = {
        "label2id":{"1 appeal to commonality - ad populum":0,
                "1 appeal to commonality - flag waving":1,
                "2 discrediting the opponent - absurdity appeal":2,
                "2 discrediting the opponent - demonization":3,
                "2 discrediting the opponent - doubt":4,
                "2 discrediting the opponent - fear appeals (destructive)":5,
                "2 discrediting the opponent - name calling":6,
                "2 discrediting the opponent - propaganda slinging":7,
                "2 discrediting the opponent - scapegoating":8,
                "2 discrediting the opponent - undiplomatic assertiveness/whataboutism":9,
                "3 loaded language":10,
                "4 appeal to authority - appeal to false authority":11,
                "4 appeal to authority - bandwagoning":12,
                "false":13},
            "label_column":"label_task3_hf",
            "hf_parameters":{"problem_type":"multi_label_classification"}
    }
    model_config = {
        "output_dir" : "/data/gmarco/predicciones-leaderboard/trained_models/mldoc",
        "hf_parameters": {
                'learning_rate':2e-5,
                'per_device_train_batch_size':8,
                'per_device_eval_batch_size':1,
                'gradient_accumulation_steps':4,
                'num_train_epochs':2,
                'weight_decay':0.01,
                'evaluation_strategy':"epoch",
                'save_strategy':"epoch",
                'load_best_model_at_end':True,}
    }

    classifier = OdesiaTextClassification(model_path="distilbert-base-uncased",
                                            dataset_path={'train':'./datasets/dipromats/baseline/dipromats_en_train.json',
                                                            'test':'./datasets/dipromats/baseline/dipromats_en_test.json'},
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)

    classifier.train()
    print(classifier.predict(split="test")) 

def test_question_answering():
    dataset_config = {"tokenizer":"distilbert-base-uncased"}
    model_config = {
        "output_dir" : "/data/gmarco/predicciones-leaderboard/qa-destilbert",
        "device": 3,
        "hf_parameters": {
                'learning_rate':2e-5,
                'per_device_train_batch_size':64,
                'per_device_eval_batch_size':64,
                'num_train_epochs':1,
                'weight_decay':0.01,}
                #'evaluation_strategy':"steps",
                #'eval_steps':50,
                #'save_strategy':"steps",
                #'load_best_model_at_end':True}
    }

    qa_trainer = OdesiaQuestionAnswering(model_path="distilbert-base-uncased",
                                            dataset_path={'train': '/data/gmarco/superimedio/datasets/SQAC_SQUAD/en_train.json', 
                                                            'test':'/data/gmarco/superimedio/datasets/SQAC_SQUAD/en_validation.json'},
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)
    #print(qa_trainer.predict())
    qa_trainer.train()
    print(qa_trainer.evaluate())
    #print(clssifierToken.predict(split="test"))
    
def test_sentece_similarity():
    
    dataset_config = {"tokenizer":"distilbert-base-uncased", 
                      }
    model_config = {
        "output_dir" : "results/sts/destilbert",
        "device": 3,
        
        "hf_parameters": {
                'learning_rate':2e-5,
                'per_device_train_batch_size':64,
                'num_train_epochs':1,
                'weight_decay':0.01,}
    }

    sts_trainer = OdesiaSentenceSimilarity(model_path="distilbert-base-uncased",
                                            dataset_path={'train': 'datasets/datasets_finished/sts_2017/train_t1_en.json', 
                                                            'test':'datasets/datasets_finished/sts_2017/test_t1_en.json',
                                                            'validation':'datasets/datasets_finished/sts_2017/val_t1_en.json'},
                                                            model_config=model_config,
                                                            dataset_config=dataset_config)
    #print(qa_trainer.predict())
    print(sts_trainer.predict())
    #print(sts_trainer.evaluate())

def main():
    #test_text_classification()
    #test_token_classification()
    #test_question_answering()
    test_sentece_similarity()

if __name__ == "__main__":
    main()