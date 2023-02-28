import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast


class ValueEvalTAPTDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int,
                 dataset_file: str):
        """
        Initialize the class by loading the dataset and its labels
        :param tokenizer: the huggingface tokenizer to encode the input text
        :param max_len: the max length of the input text
        :param dataset_file: the file where the dataset is stored
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = pd.read_csv(filepath_or_buffer=dataset_file, sep='\t')
        self.stance_to_id = {'against': 0, 'in favor of': 1}

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        conclusion = self.dataset.iloc[item]['Conclusion']
        premise = self.dataset.iloc[item]['Premise']
        stance = self.dataset.iloc[item]['Stance']  # labels
        encoded_features = self.tokenizer(conclusion, premise,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          return_token_type_ids=True,
                                          padding="max_length",
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_tensors="pt")

        label = self.stance_to_id[stance]
        return dict(
            input_ids=encoded_features['input_ids'].flatten(),
            token_type_ids=encoded_features['token_type_ids'].flatten(),
            attention_mask=encoded_features['attention_mask'].flatten(),
            label=torch.tensor(label)
        )

