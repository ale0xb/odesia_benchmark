from evaluate import load



def get_overall_f1(predictions, gold_standard):
        metrica = load('seqeval', trust_remote_code=True)
        print(len(predictions))
        print(len(gold_standard))
        result = metrica.compute(predictions=predictions, references=gold_standard, scheme='IOB2')
        return result, result['overall_f1']


def evaluate(predictions_file_path, goldstandard_file_path):
    print("FMeasure.evaluate")
    print(f"Evaluating {predictions_file_path} with gold file: {goldstandard_file_path}:")
    
    with open(goldstandard_file_path) as gold_file:
        file_gold = gold_file.read()
    with open(predictions_file_path) as pred_file:
        file_pred = pred_file.read()

    gold = eval(file_gold)
    pred = eval(file_pred)

    aux_pred = []
    aux_gold = []
    ids_faults = []
    for p, g in zip(pred, gold):
        for i_p in p['ner_tags']:
            if len(i_p) == 0:
                ids_faults.append(p['id'])
                print(g['id'])
                print(g)
                continue
            aux_pred.append(i_p)
        for i_g in g['value']:
            if g['id'] in ids_faults:
                print(g['id'])
                print(g)
                continue
            aux_gold.append(i_g)
    print("·························")
    print(goldstandard_file_path)
    print(len(aux_gold))
    print(len(aux_pred))
    print("·························")

      # Count the number of tokens for each element in aux_gold and aux_pred
    gold_token_counts = [len(tokens) for tokens in aux_gold]
    pred_token_counts = [len(tokens) for tokens in aux_pred]
    
    # Compute the mean for each one
    mean_gold_tokens = sum(gold_token_counts) / len(gold_token_counts) if gold_token_counts else 0
    mean_pred_tokens = sum(pred_token_counts) / len(pred_token_counts) if pred_token_counts else 0
    
    print("Mean number of tokens in aux_gold:", mean_gold_tokens)
    print("Mean number of tokens in aux_pred:", mean_pred_tokens)

    res, f1 = get_overall_f1([aux_pred], [aux_gold])
    return f1

preds = 'trained_models/meta-llama-Meta-Llama-31-8B/diann_2023_es/_per_device_train_batch_size_1_gradient_accumulation_steps_4_learning_rate_0.0001_weight_decay_0.0/predictions.json'
gold = 'scripts/DIANN_2023_T1_es.json'
print(evaluate(preds, gold))