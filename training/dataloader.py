from datasets import load_dataset
from training import add_markers


def load_data(type):

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
            sentence_marked = add_markers(entry['context'] ,sentence_group['sentence'])
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


