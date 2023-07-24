import time
import torch
import requests
from typing import List

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

        # Maps classifiers to their available models
        self.classifier_model_mapping = {
                "Token": {
                    "All": "wu981526092/Token-Level-Multidimensional-Bias-Detector",
                    "Race": "wu981526092/Token-Level-Race-Bias-Detector",
                    "Gender": "wu981526092/Token-Level-Gender-Bias-Detector",
                    "Profession": "wu981526092/Token-Level-Profession-Bias-Detector",
                    "Religion": "wu981526092/Token-Level-Religion-Bias-Detector",
                },
                "Sentence": {
                    "All":"wu981526092/Sentence-Level-Multidimensional-Bias-Detector",
                    "Religion": "wu981526092/Sentence-Level-Religion-Bias-Detector",
                    "Profession": "wu981526092/Sentence-Level-Profession-Bias-Detector",
                    "Race": "wu981526092/Sentence-Level-Race-Bias-Detector",
                    "Gender": "wu981526092/Sentence-Level-Gender-Bias-Detector",
                }
        }

        self.classifier = classifier
        self.model_type = model_type

        if classifier not in self.classifier_model_mapping:
            raise ValueError(f"Invalid classifier. Expected one of: {list(self.classifier_model_mapping.keys())}")

        if model_type not in self.classifier_model_mapping[classifier]:
            raise ValueError(
                f"Invalid model_type for {classifier}. Expected one of: {list(self.classifier_model_mapping[classifier].keys())}")

        self.model_path = self.classifier_model_mapping[classifier][model_type]

        # Create the API endpoint from the model path
        self.API_URL = f"https://api-inference.huggingface.co/models/{self.model_path}"
        API_token = "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"
        # Add authorization token (if required)
        self.headers = {"Authorization": f"Bearer {API_token}"} # Replace `your_api_token` with your token

    import time

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload).json()

        # If the model is loading, wait for the estimated time and retry
        while 'error' in response and 'estimated_time' in response:
            print(f"Model is currently loading. Waiting for {response['estimated_time']} seconds.")
            time.sleep(response['estimated_time'])

            # Retry the request
            print("Retrying the request...")
            response = requests.post(self.API_URL, headers=self.headers, json=payload).json()

        if 'error' in response:
            raise RuntimeError(f"Error: {response['error']}")

        return response

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

        # Prepare the payload
        payload = {"inputs": texts}

        # Query the API
        try:
            predictions = self.query(payload)
        except RuntimeError as e:
            print("Prediction failed due to a RuntimeError")
            raise e
        except Exception as e:
            print("Prediction failed due to an unknown error")
            raise e

        # Process the predictions
        for text, prediction in zip(texts, predictions):
            # Check if an error occurred while loading the model
            if 'error' in prediction:
                print(f"Error for text '{text}': {prediction['error']}")
                results.append({text: {'error': prediction['error']}})
            else:
                result = {}
                if self.classifier == 'Token':
                    for item in prediction:
                        result[item['word']] = {item['entity_group']: item['score']}
                elif self.classifier == 'Sentence' and self.model_type == 'All':
                    result = {item['label'].split('__')[-1]: item['score'] for item in
                              prediction}
                else:
                    result = {item['label'].split('__')[-1]: item['score'] for item in
                              prediction}
                results.append({text: result})

        return results


if __name__ == '__main__':
    detector_test = Detector("Sentence","All")
    test_data = ["The girl performed poorly at reading in school.","Sir is an address showing respect for a person. It usually refers to a male."]
    result = detector_test.predict(test_data)
    print(result[1][test_data[1]])