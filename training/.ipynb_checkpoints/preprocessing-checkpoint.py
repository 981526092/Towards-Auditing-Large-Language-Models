import re
from datasets import Dataset


def prepare_text_single(data):
    new_data = []
    for item in data:
        text = item['text']
        label = item['label']

        # Split sentence also by punctuation
        tokens = re.findall(r"===\w+===|[\w']+|[.,!?;]", text)

        labels = []
        tokenlist = []
        for token in tokens:
            in_bias = False
            if token.startswith('===') and token.endswith('==='):

                in_bias = True
                new_token = token[3:]  # Remove the marker
                new_token = new_token[:-3]  # Remove the marker
                tokenlist.append(new_token)
            else:
                tokenlist.append(token)

            if in_bias:
                labels.append(label)  # bias token with given label
            else:
                labels.append(2)  # non-bias token with label 'unrelated'

        new_item = {
            'tokens': tokenlist,
            'labels': labels
        }
        new_data.append(new_item)
    return new_data

def prepare_text_multiple(data, bias_type=None):
    if bias_type is None:
        bias_type = ["gender", "race", "profession", "religion"]
    new_data = []
    for type_bias in bias_type:
        for item in data[type_bias]:
            text = item['text']
            label = item['label']

            # Split sentence also by punctuation
            tokens = re.findall(r"===\w+===|[\w']+|[.,!?;]", text)

            labels = []
            tokenlist = []
            for token in tokens:
                in_bias = False
                if token.startswith('===') and token.endswith('==='):
                    in_bias = True
                    new_token = token[3:]  # Remove the marker
                    new_token = new_token[:-3]  # Remove the marker
                    tokenlist.append(new_token)
                else:
                    tokenlist.append(token)

                if in_bias:
                    if (type_bias == "gender"):
                        if(label == 0):
                            labels.append(1)
                        else:
                            labels.append(2)
                    if (type_bias == "race"):
                        if(label == 0):
                            labels.append(3)
                        else:
                            labels.append(4)
                    if (type_bias == "profession"):
                        if(label == 0):
                            labels.append(5)
                        else:
                            labels.append(6)
                    if (type_bias == "religion"):
                        if(label == 0):
                            labels.append(7)
                        else:
                            labels.append(8)
                else:
                    labels.append(0)  # non-bias token with label 'unrelated'

            new_item = {
                'tokens': tokenlist,
                'labels': labels
            }
            new_data.append(new_item)
    return new_data
# 0: "unrelated"
# 1: "stereotype_gender",
# 2: "anti-stereotype_gender",
# 3: "stereotype_race",
# 4: "anti-stereotype_race",
# 5: "stereotype_profession",
# 6: "anti-stereotype_profession",
# 7: "stereotype_religion",
# 8: "anti-stereotype_religion",

def prepare_dataset(tokenizer, data):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    hf_dataset = Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})
    tokenized_data = hf_dataset.map(tokenize_and_align_labels, batched=True)

    return tokenized_data

def prepare_MD_SL_data(new_data):
    bias_type = ["gender","profession","race","religion"]
    return_data = []
    for type_bias in bias_type:
        set = new_data[type_bias]
        for entry in set:
            label = entry["label"]
            text = entry["text"]
            new_label = 0
            if label == 0 or label == 1:
                if (type_bias == "gender"):
                    if(label == 0):
                        new_label = 1
                    else:
                        new_label = 2
                if (type_bias == "race"):
                    if(label == 0):
                        new_label = 3
                    else:
                        new_label = 4
                if (type_bias == "profession"):
                    if(label == 0):
                        new_label = 5
                    else:
                        new_label = 6
                if (type_bias == "religion"):
                    if(label == 0):
                        new_label = 7
                    else:
                        new_label = 8
            else:
                new_label = 0  # non-bias token with label 'unrelated'

            new_item = {
            'text': text,
            'label': new_label
            }
            return_data.append(new_item)
    return return_data
