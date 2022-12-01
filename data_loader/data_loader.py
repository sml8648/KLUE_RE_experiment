import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm

class RE_Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, labels):

        self.example = []
        for each in dataset:
            self.example.append({k: torch.tensor(v) for k, v in each.items()})
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):

        self.example[idx]['label'] = self.labels[idx]
        return self.example[idx]

    def __len__(self):
        return len(self.example)


def tokenized_dataset(dataset, tokenizer):

    data = []
    for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):
        subj = eval(item["subject_entity"])["word"]
        obj = eval(item["object_entity"])["word"]
        concat_entity = tokenizer.sep_token.join([subj, obj])
        output = tokenizer(concat_entity, item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True)
        data.append(output)

    return data


def load_dataset(tokenizer, data_path):
    dataset = pd.read_csv(data_path, index_col=0)
    tokenized_test = tokenized_dataset(dataset, tokenizer)
    RE_dataset = RE_Dataset(tokenized_test, dataset.label.values)
    return RE_dataset

def load_test_dataset(tokenizer, predict_path):
    predict_dataset = pd.read_csv(predict_path, index_col=0)
    predict_label = None
    tokenized_predict = tokenized_dataset(predict_dataset, tokenizer)
    RE_predict_dataset = RE_Dataset(tokenized_predict, predict_label)
    return RE_predict_dataset
