import json

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification, \
    DataCollatorForTokenClassification
import argparse
from preprocessing import prepare_text_single, prepare_dataset
from dataloader import load_data_local, load_data_crowspairs

def train_SD_TL(new_data, model_path, bias_type, batch_size,epoch, learning_rate,output_dir,seed):
    id2label = {
        0: "stereotype",
        1: "anti-stereotype",
        2: "unrelated"
    }
    label2id = {
        "stereotype": 0,
        "anti-stereotype": 1,
        "unrelated": 2
    }

    model = AutoModelForTokenClassification.from_pretrained(
        model_path, num_labels=3, id2label=id2label, label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = prepare_text_single(new_data)
    tokenized_data = prepare_dataset(tokenizer, data)
    final_dataset = tokenized_data.train_test_split(0.2, shuffle=True,seed = seed)

    # Define data collator to handle padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    label_list = ["stereotype", "anti-stereotype", "unrelated"]
    labels = [label_list[i] for i in data[0]["labels"]]

    def compute_metrics_new(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Flatten the lists
        true_predictions = [pred for sublist in true_predictions for pred in sublist]
        true_labels = [label for sublist in true_labels for label in sublist]

        # Calculate precision, recall, f1_score, and support with "macro" average
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='macro')

        balanced_acc = balanced_accuracy_score(true_labels, true_predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced accuracy": balanced_acc,
        }

    final_dir = None
    if output_dir is None:
        final_dir = "SD_TL_best_model/" + bias_type
    else:
        final_dir = output_dir

    training_args = TrainingArguments(
        output_dir=final_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1
    )
    model = model.to("mps")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_new,
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
    parser.add_argument('--dataset_select', nargs='+', default=['intrasentence'], help='Dataset selection')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=6, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None, help='Save Directory')
    parser.add_argument('--seed', type=int, default=66, help='Seed')


    args = parser.parse_args()

    new_data = None

    if "intrasentence" in args.dataset_select:
        intrasentence_dataset = load_data_local("intrasentence", marked=True)
        if new_data is None:
            new_data = intrasentence_dataset[args.bias_type].copy()
        else:
            new_data.extend(intrasentence_dataset[args.bias_type].copy())

    if "crowspairs" in args.dataset_select:
        crowspairs_dataset = load_data_crowspairs(marked=True)
        mask_bias_type = args.bias_type
        if args.bias_type == "race":
            mask_bias_type = "race-color"
        if new_data is None:
            new_data = crowspairs_dataset[mask_bias_type].copy()
        else:
            new_data.extend(crowspairs_dataset[mask_bias_type].copy())

    result = train_SD_TL(new_data, args.model_path, args.bias_type, args.batch_size, args.epoch, args.learning_rate,args.output_dir,args.seed)
    print(result)

if __name__ == '__main__':
    main()