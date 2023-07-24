import json

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from dataloader import load_data_crowspairs,load_data_local


def train_SD_SL(new_data, model_path, bias_type, batch_size,epoch, learning_rate,output_dir,seed):

    id2label = {
        0: "unrelated",
        1: "stereotype",
        2: "anti-stereotype",
    }
    label2id = {
        "unrelated": 0,
        "stereotype": 1,
        "anti-stereotype": 2,
    }

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, id2label=id2label,
                                                               label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Convert the list of dictionaries into a datasets object
    temp_data = Dataset.from_pandas(pd.DataFrame(new_data))

    # Tokenize the datasets
    tokenized_datasets = temp_data.map(tokenize_function, batched=True)

    # Convert the 'label' column to a list
    tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': [examples['label']]})
    final_dataset = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True,seed=seed)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

        balanced_acc = balanced_accuracy_score(labels, predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced accuracy": balanced_acc,
        }

    final_dir = None
    if output_dir is None:
        final_dir = "SD_SL_best_model/" + bias_type
    else:
        final_dir = output_dir

    training_args = TrainingArguments(
        output_dir=final_dir,
        num_train_epochs=epoch,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1)
    model = model.to("mps")
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    result = trainer.evaluate(final_dataset["test"])
    print(result)

    with open(final_dir + '/result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return result

# The main function that will be called when the script is executed
def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--model_path', type=str, default='distilbert-base-uncased', help='Path to the model')
    parser.add_argument('--bias_type', type=str, default='religion', help='Type of bias')
    parser.add_argument('--dataset_select', nargs='+', default=['intersentence'], help='Dataset selection')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=6, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None, help='Save Directory')
    parser.add_argument('--seed', type=int, default=66, help='Seed')

    args = parser.parse_args()

    new_data = None
    if "intersentence" in args.dataset_select:
        intersentence_dataset = load_data_local("intersentence")
        if new_data is None:
            new_data = intersentence_dataset[args.bias_type].copy()
        else:
            new_data.extend(intersentence_dataset[args.bias_type].copy())

    if "intrasentence" in args.dataset_select:
        intrasentence_dataset = load_data_local("intrasentence", marked=False)
        if new_data is None:
            new_data = intrasentence_dataset[args.bias_type].copy()
        else:
            new_data.extend(intrasentence_dataset[args.bias_type].copy())

    if "crowspairs" in args.dataset_select:
        crowspairs_dataset = load_data_crowspairs(marked=False)
        mask_bias_type = args.bias_type
        if args.bias_type == "race":
            mask_bias_type = "race-color"
        if new_data is None:
            new_data = crowspairs_dataset[mask_bias_type].copy()
        else:
            new_data.extend(crowspairs_dataset[mask_bias_type].copy())

    result = train_SD_SL(new_data, args.model_path, args.bias_type, args.batch_size, args.epoch, args.learning_rate,args.output_dir,args.seed)
    print(result)

if __name__ == '__main__':
    main()