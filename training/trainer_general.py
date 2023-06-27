import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification, \
    DataCollatorForTokenClassification
from datasets import Dataset
from transformers import AutoModelForSequenceClassification

from training import prepare_text_single, prepare_dataset


def train_SD_SL(new_data, model_path, bias_type, batch_size=16):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Convert the list of dictionaries into a datasets object
    temp_data = Dataset.from_pandas(pd.DataFrame(new_data))

    temp_data = temp_data.shuffle(seed=42)
    # Tokenize the datasets
    tokenized_datasets = temp_data.map(tokenize_function, batched=True)

    # Convert the 'label' column to a list
    tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': [examples['label']]})
    final_dataset = tokenized_datasets.train_test_split(test_size=0.2)

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

    training_args = TrainingArguments(
        use_mps_device=True,
        output_dir="specific_best_model/" + bias_type,
        num_train_epochs=6,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    result = trainer.evaluate(final_dataset["test"])

    return result


def train_SD_TL(new_data, model_path, bias_type, batch_size=16):
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
        "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = prepare_text_single(new_data)
    tokenized_data = prepare_dataset(tokenizer, data)
    final_dataset = tokenized_data.train_test_split(0.2)

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

    training_args = TrainingArguments(
        use_mps_device=True,
        output_dir="specific_best_model/" + bias_type,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=6,
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
    return result