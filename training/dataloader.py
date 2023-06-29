import json
import re

import pandas as pd
from datasets import load_dataset
from preprocessing import add_markers,add_marker_crowspairs



def load_data(type,marked = True):

    intrasentence_dataset = load_dataset('stereoset' ,type)["validation"]

    profession_dataset = []
    race_dataset = []
    gender_dataset = []
    religion_dataset = []

    for x in range(len(intrasentence_dataset)):
        entry = intrasentence_dataset[x]
        bias_type = entry['bias_type']
        sentence_group = entry['sentences']
        if type == "intrasentence":
            if marked == True:
                sentence_marked = add_markers(entry['context'] ,sentence_group['sentence'])
            else:
                sentence_marked = []
                for x in sentence_group['sentence']:
                    sentence_marked.append(x)
        elif type == "intersentence":
            sentence_marked = []
            for x in sentence_group['sentence']:
                sentence_marked.append(entry['context'] + " " + x)
        label = sentence_group['gold_label']  # 0 stereotype 1 anti-stereotype 2 unrelated

        for x in range(len(sentence_marked)):
            temp_data = {}
            temp_data["text"] = sentence_marked[x]
            temp_data["label"] = label[x]

            if bias_type == "profession":
                profession_dataset.append(temp_data)
            if bias_type == "race":
                race_dataset.append(temp_data)
            if bias_type == "gender":
                gender_dataset.append(temp_data)
            if bias_type == "religion":
                religion_dataset.append(temp_data)

    result_dataset = {"profession": profession_dataset, "race": race_dataset, "gender": gender_dataset,
                      "religion": religion_dataset}

    return result_dataset

def load_data_local(type,data_path='/Users/zekunwu/Desktop/bias_detector/data/',marked = True):

    f = open(data_path+'test.json')
    test_data = json.load(f)
    f.close()

    f = open(data_path+'dev.json')
    dev_data = json.load(f)
    f.close()

    dataset = test_data['data'][type].copy()
    dataset.extend(dev_data['data'][type])

    profession_dataset = []
    race_dataset = []
    gender_dataset = []
    religion_dataset = []

    for x in range(len(dataset)):
        entry = dataset[x]
        bias_type = entry['bias_type']
        sentence_group = entry['sentences']
        if type == "intrasentence":
            if marked == True:
                group_sentence = []
                label = []
                for x in sentence_group:
                    group_sentence.append(x['sentence'])
                    label.append(x['gold_label'])
                sentence_marked = add_markers(entry['context'] ,group_sentence)
            else:
                sentence_marked = []
                label = []
                for x in sentence_group:
                    sentence_marked.append(x['sentence'])
                    label.append(x['gold_label'])
        elif type == "intersentence":
            sentence_marked = []
            label = []
            for x in sentence_group:
                sentence_marked.append(entry['context'] + " " + x['sentence'])
                label.append(x['gold_label'])  # 0 stereotype 1 anti-stereotype 2 unrelated

        new_label = []
        for pre in label:
            if pre == "stereotype":
                new_label.append(0)
            elif pre == "anti-stereotype":
                new_label.append(1)
            else:
                new_label.append(2)

        for x in range(len(sentence_marked)):
            temp_data = {}
            temp_data["text"] = sentence_marked[x]
            temp_data["label"] = new_label[x]

            if bias_type == "profession":
                profession_dataset.append(temp_data)
            if bias_type == "race":
                race_dataset.append(temp_data)
            if bias_type == "gender":
                gender_dataset.append(temp_data)
            if bias_type == "religion":
                religion_dataset.append(temp_data)

    result_dataset = {"profession": profession_dataset, "race": race_dataset, "gender": gender_dataset,
                      "religion": religion_dataset}

    return result_dataset


def load_data_crowspairs(data_path = "/Users/zekunwu/Desktop/bias_detector/data/crows_pairs_anonymized.csv",marked = True):

    crows_pair = pd.read_csv(data_path)
    crows_pair = crows_pair.drop(columns=["annotations","anon_writer","anon_annotators"])
    bias_types = list(crows_pair["bias_type"].unique())

    # Create a dictionary mapping each bias type to an empty list
    datasets = {bias_type: [] for bias_type in bias_types}

    for idx in range(len(crows_pair)):
        entry1 = {}
        entry2 = {}

        s1 = crows_pair["sent_more"][idx]
        s2 = crows_pair["sent_less"][idx]
        if marked == True:
            sent1, sent2 = add_marker_crowspairs(s1, s2)
        else:
            sent1 = s1
            sent2 = s2
        label = crows_pair["stereo_antistereo"][idx]

        if label == "stereo":
            entry1["text"] = sent1
            entry1["label"] = 0
            entry2["text"] = sent2
            entry2["label"] = 1
        elif label == "antistereo":
            entry1["text"] = sent1
            entry1["label"] = 1
            entry2["text"] = sent2
            entry2["label"] = 0
        else:
            print(f"Unexpected label value: {label}")
            continue

        # Get the bias type of the current pair
        bias_type = crows_pair["bias_type"][idx]

        # Append to the appropriate dataset
        if bias_type in datasets:
            datasets[bias_type].append(entry1)
            datasets[bias_type].append(entry2)

    return datasets


