

import json
import torch
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from typing import List, Dict

class Detector:
    """
    A class for detecting various forms of bias in text using pre-trained models.
    """

    def __init__(self, classifier, model_type, device=None):
        """
        Initializes the detector with a specific model and device.

        Args:
            classifier (str): The type of classifier to use.
            model_type (str): The type of the model to use.
            device (str): The device to use for computations. If None, the device is set to 'cuda' if available, else 'cpu'.
        """
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)

        # Maps classifiers to their available models and class_names
        self.classifier_model_mapping = {
            "Token": {
                "All": "wu981526092/token-level-bias-detector",
                "Specific": "unitary/unbiased-toxic-roberta",
            },
            "Sentence": {
                "single": "cardiffnlp/roberta-large-tweet-topic-single-all",
                "multi": "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all",
            },
            "Dialogue": {
                "deberta": "JasperLS/deberta-v3-base-injection",
                "gelectra": "JasperLS/gelectra-base-injection",
            },
        }

        self.classifier = classifier

        if classifier not in self.classifier_model_mapping:
            raise ValueError(f"Invalid classifier. Expected one of: {list(self.classifier_model_mapping.keys())}")

        if model_type not in self.classifier_model_mapping[classifier]:
            raise ValueError(
                f"Invalid model_type for {classifier}. Expected one of: {list(self.classifier_model_mapping[classifier].keys())}")

        self.model_path = self.classifier_model_mapping[classifier][model_type]

        try:
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.classifier == 'Token':
                self.pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device)
            else:
                self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=self.device)
        except Exception as e:
            print("Failed to initialize the pipeline")
            raise e

        # Fetch the class labels by making a dummy prediction
        dummy_output = self.pipe("dummy text")
        self.class_labels = list(set([item['entity'] for item in dummy_output])) if self.classifier == 'Token' else list(set([item['label'].split('__')[-1] for item in dummy_output]))

    @torch.no_grad()
    def predict(self, texts: List[str]):
        """
        Predicts the bias of the given text or list of texts.

        Args:
            texts (List[str]): A list of strings to analyze.

        Returns:
            A list of dictionaries. Each dictionary contains the 'label' and 'score' for each text.
        """
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in 'texts' should be of str type")

        results = []

        try:
            predictions = self.pipe(texts)
        except RuntimeError as e:
            print("Prediction failed due to a RuntimeError")
            raise e
        except Exception as e:
            print("Prediction failed due to an unknown error")
            raise e

        for text, prediction in zip(texts, predictions):
            result = {}
            if self.classifier == 'Token':
                for item in prediction:
                    # Include the token in the result
                    result[item['word']] = {item['entity']: item['score']}
            else:
                result = {item['label'].split('__')[-1]: item['score'] for item in prediction}
            results.append({text: result})

        return results


if __name__ == '__main__':
    detector_test = Detector("Token","All")
    print(detector_test.predict(["this is a test message","tasdasda"]))