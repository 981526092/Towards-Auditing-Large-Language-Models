import argparse

import numpy as np
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer

from training import prepare_text_multiple, prepare_dataset, load_data_local, load_data_crowspairs
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score


def train_MD_TL(new_data, model_path, batch_size, epoch, learning_rate, output_dir):
    data = prepare_text_multiple(new_data)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_data = prepare_dataset(tokenizer, data)
    final_dataset = tokenized_data.train_test_split(0.2, shuffle=True)

    # Define data collator to handle padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    label_list = ["unrelated", "stereotype_gender", "anti-stereotype_gender", "stereotype_race", "anti-stereotype_race",
                  "stereotype_profession", "anti-stereotype_profession", "stereotype_religion",
                  "anti-stereotype_religion"]
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

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, true_predictions)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced accuracy": balanced_acc,
        }

    id2label = {
        0: "unrelated",
        1: "stereotype_gender",
        2: "anti-stereotype_gender",
        3: "stereotype_race",
        4: "anti-stereotype_race",
        5: "stereotype_profession",
        6: "anti-stereotype_profession",
        7: "stereotype_religion",
        8: "anti-stereotype_religion",

    }
    label2id = {
        "unrelated": 0,
        "stereotype_gender": 1,
        "anti-stereotype_gender": 2,
        "stereotype_race": 3,
        "anti-stereotype_race": 4,
        "stereotype_profession": 5,
        "anti-stereotype_profession": 6,
        "stereotype_religion": 7,
        "anti-stereotype_religion": 8,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        model_path, num_labels=9, id2label=id2label, label2id=label2id
    )

    final_dir = None
    if output_dir is None:
        final_dir = "MD_TL_best_model/"
    else:
        final_dir = output_dir

    training_args = TrainingArguments(
        #use_mps_device=True,
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

    return result


# The main function that will be called when the script is executed
def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--model_path', type=str, default='distilbert-base-uncased', help='Path to the model')
    parser.add_argument('--dataset_select', nargs='+', default=['intrasentence'], help='Dataset selection')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=6, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None, help='Save Directory')

    args = parser.parse_args()

    new_data = None

    if "intrasentence" in args.dataset_select:
        intrasentence_dataset = load_data_local("intrasentence", marked=True)
        if new_data is None:
            new_data = intrasentence_dataset.copy()

    if "crowspairs" in args.dataset_select:
        crowspairs_dataset = load_data_crowspairs(marked=True)
        new_data['race'].extend(crowspairs_dataset['race-color'])
        new_data['gender'].extend(crowspairs_dataset['gender'])
        new_data['religion'].extend(crowspairs_dataset['religion'])


    result = train_MD_TL(new_data, args.model_path, args.batch_size, args.epoch, args.learning_rate,args.output_dir)
    print(result)

if __name__ == '__main__':
    main()