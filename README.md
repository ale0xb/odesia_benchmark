# odesia_benchmark
Code for evaluate language models with the ODESIA benchmark (http://leaderboard.odesia.uned.es/)

# How to use
If you want to evaluate your model in all ODESIA datasets:

```
  from odesia_evaluate_model import odesia_benchmark
  
  odesia_benchmark(model="PlanTL-GOB-ES/roberta-large-bne", language="es")
```

The code also allows to perform a grid search for finding the best hyperparameters:


```
  from odesia_evaluate_model import odesia_benchmark

   hparams_to_search = {
            'per_device_train_batch_size' : [16, 32],
            'learning_rate': [0.00001, 0.00003, 0.00005],
            'weight_decay': [0.1, 0.01]
        }
  
  odesia_benchmark(model="PlanTL-GOB-ES/roberta-large-bne", language="es", grid_search=hparams_to_search)
```
