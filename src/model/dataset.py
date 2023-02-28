from typing import List
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast


class ValueEvalDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int,
                 dataset_file: str,
                 labels_file: str,
                 mode: str = 'train'):
        """
        Initialize the class by loading the dataset and its labels
        :param tokenizer: the huggingface tokenizer to encode the input text
        :param max_len: the max length of the input text
        :param dataset_file: the file where the dataset is stored
        :param labels_file: the file where the labels are stored
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        dataset = pd.read_csv(filepath_or_buffer=dataset_file, sep='\t')
        self.mode = mode
        if mode == 'train' or mode == 'val':
            self._merge_labels_with_dataset(dataset=dataset, labels_file=labels_file)
        else:
            self.dataset = dataset


    def _merge_labels_with_dataset(self,
                                   dataset: pd.DataFrame,
                                   labels_file:str):
        """
        Merge the arguments pandas dataframe with their labels' pandas dataframe
        :param dataset: the arguments stored in a pandas dataframe
        :param labels_file: the file that are stored the labels of the arguments
        """
        labels = pd.read_csv(filepath_or_buffer=labels_file, sep='\t')
        self.dataset = pd.merge(left=dataset,
                                right=labels,
                                left_on='Argument ID',
                                right_on='Argument ID')

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        conclusion = self.dataset.iloc[item]['Conclusion']
        stance = self.dataset.iloc[item]['Stance']
        premise = self.dataset.iloc[item]['Premise']
        encoded_conclusion = self.tokenizer.encode_plus(conclusion,
                                                        add_special_tokens=True,
                                                        max_length=self.max_len,
                                                        return_token_type_ids=False,
                                                        padding="max_length",
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors="pt")
        encoded_premise = self.tokenizer.encode_plus(premise,
                                                     add_special_tokens=True,
                                                     max_length=self.max_len,
                                                     return_token_type_ids=False,
                                                     padding="max_length",
                                                     truncation=True,
                                                     return_attention_mask=True,
                                                     return_tensors="pt"
                                                     )
        if self.mode == 'train':
            labels = np.asarray(self.dataset.iloc[item][4:], dtype=np.int32)
            return dict(conclusion_input_ids=encoded_conclusion['input_ids'].flatten(),
                        conclusion_attention_mask=encoded_conclusion['attention_mask'].flatten(),
                        premise_input_ids=encoded_premise['input_ids'].flatten(),
                        premise_attention_mask=encoded_premise['attention_mask'].flatten(),
                        labels=torch.FloatTensor(labels),
                        stance=stance)
        else:
            return dict(conclusion_input_ids=encoded_conclusion['input_ids'].flatten(),
                        conclusion_attention_mask=encoded_conclusion['attention_mask'].flatten(),
                        premise_input_ids=encoded_premise['input_ids'].flatten(),
                        premise_attention_mask=encoded_premise['attention_mask'].flatten(),
                        stance=stance)


class ValueEvalDataLoader(object):
    def __init__(self,
                 train_dataset_file: str,
                 train_labels_file: str,
                 val_dataset_file: str,
                 val_labels_file: str,
                 batch_size: int,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int,
                 mode: str = 'train'):
        """
        Initialize the dataloader for the train and val sets.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_dataset_file = train_dataset_file
        self.train_labels_file = train_labels_file
        self.val_dataset_file = val_dataset_file
        self.val_labels_file = val_labels_file
        self.mode = mode
        self._setup_dataloaders()

    def _setup_dataloaders(self) -> None:
        """
        Set up Train and Validation Datasets.
        """

        self.train_dataset = ValueEvalDataset(tokenizer=self.tokenizer,
                                              max_len=self.max_len,
                                              dataset_file=self.train_dataset_file,
                                              labels_file=self.train_labels_file,
                                              mode=self.mode)

        self.val_dataset = ValueEvalDataset(tokenizer=self.tokenizer,
                                            max_len=self.max_len,
                                            dataset_file=self.val_dataset_file,
                                            labels_file=self.val_labels_file,
                                            mode=self.mode)

    def train_dataloader(self) -> DataLoader:
        """
        Create, set up and return the Train DataLoader.
        :return: the train DataLoader
        """
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=2,
                          shuffle=True,
                          drop_last=False)

    def val_dataloader(self) -> DataLoader:
        """
        Create, set up and return the Validation DataLoader.
        :return: the validation DataLoader
        """
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=2,
                          drop_last=False,
                          shuffle=False)


class TaptValueEvalDataset(ValueEvalDataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int,
                 dataset_file: str,
                 labels_file: str,
                 mode: str = 'train'):
        """
        Initialize the class by loading the dataset and its labels
        :param tokenizer: the huggingface tokenizer to encode the input text
        :param max_len: the max length of the input text
        :param dataset_file: the file where the dataset is stored
        :param labels_file: the file where the labels are stored
        :param mode: if mode is not train that it will not return the labels
        """
        super(TaptValueEvalDataset, self).__init__(tokenizer=tokenizer,
                                                   max_len=max_len,
                                                   dataset_file=dataset_file,
                                                   labels_file=labels_file,
                                                   mode=mode)

    def __getitem__(self, item):
        conclusion = self.dataset.iloc[item]['Conclusion']
        premise = self.dataset.iloc[item]['Premise']
        encoded_features = self.tokenizer(conclusion, premise,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          return_token_type_ids=True,
                                          padding="max_length",
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_tensors="pt")
        if self.mode == 'train':
            labels = np.asarray(self.dataset.iloc[item][4:], dtype=np.int32)
            return dict(
                input_ids=encoded_features['input_ids'].flatten(),
                token_type_ids=encoded_features['token_type_ids'].flatten(),
                attention_mask=encoded_features['attention_mask'].flatten(),
                labels=torch.FloatTensor(labels))
        else:
            return dict(
                input_ids=encoded_features['input_ids'].flatten(),
                token_type_ids=encoded_features['token_type_ids'].flatten(),
                attention_mask=encoded_features['attention_mask'].flatten())


class TaptValueEvalDataLoader(ValueEvalDataLoader):
    def __init__(self,
                 train_dataset_file: str,
                 train_labels_file: str,
                 val_dataset_file: str,
                 val_labels_file: str,
                 batch_size: int,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int,
                 mode: str = 'train'):
        """
        Initialize the dataloader for the train and val sets.
        """
        super(TaptValueEvalDataLoader, self).__init__(train_dataset_file=train_dataset_file,
                                                      train_labels_file=train_labels_file,
                                                      val_dataset_file=val_dataset_file,
                                                      val_labels_file=val_labels_file,
                                                      batch_size=batch_size,
                                                      tokenizer=tokenizer,
                                                      max_len=max_len,
                                                      mode=mode)

    def _setup_dataloaders(self) -> None:
        """
        Set up Train and Validation Datasets.
        """

        self.train_dataset = TaptValueEvalDataset(tokenizer=self.tokenizer,
                                                  max_len=self.max_len,
                                                  dataset_file=self.train_dataset_file,
                                                  labels_file=self.train_labels_file,
                                                  mode=self.mode)

        self.val_dataset = TaptValueEvalDataset(tokenizer=self.tokenizer,
                                                max_len=self.max_len,
                                                dataset_file=self.val_dataset_file,
                                                labels_file=self.val_labels_file,
                                                mode=self.mode)


class UnifiedTaptValueEvalDataset(ValueEvalDataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int,
                 dataset_file: str,
                 labels_file: str,
                 against_phrases: List[str],
                 in_favor_phrases: List[str],
                 ova: bool,
                 label_index: int,
                 downsample: bool,
                 mode: str = 'train'):
        """
        Initialize the class by loading the dataset and its labels
        :param tokenizer: the huggingface tokenizer to encode the input text
        :param max_len: the max length of the input text
        :param dataset_file: the file where the dataset is stored
        :param labels_file: the file where the labels are stored
        :param against_phrases: a List of phrases to replace the 'against' stance with
        :param in_favor_phrases: a list of phrases to replace the 'in favor of' stance with
        :param downsample: if ova is active, opt the downsapling of the negative class
        :param mode: if mode is not train that it will not return the labels
        """
        super(UnifiedTaptValueEvalDataset, self).__init__(tokenizer=tokenizer,
                                                          max_len=max_len,
                                                          dataset_file=dataset_file,
                                                          labels_file=labels_file,
                                                          mode=mode)
        self.ova = ova
        self.downsample = downsample
        self.label_index = label_index
        if self.ova:
            if self.mode == 'train' or self.mode == 'val':
                # print(self.dataset.head())
                self.arguments = self.dataset.iloc[:, :4]
                self.labels = self.dataset.iloc[:, 4 + self.label_index].to_frame()
                self.dataset = pd.concat([self.arguments, self.labels], axis=1)
                if self.downsample:
                    self.__downsampling()
        self.against_phrases = against_phrases
        self.in_favor_phrases = in_favor_phrases
        self.no_against_phrases = len(against_phrases) - 1
        self.no_in_favor_phrases = len(in_favor_phrases) - 1

    def __downsampling(self):
        """
        If ova is True then downsample the negative class in the train dataloader
        """
        positives = self.dataset[self.dataset.iloc[:, -1] == 1]
        negatives = self.dataset[self.dataset.iloc[:, -1] == 0]
        pos_shape = positives.shape
        pos_len = pos_shape[0]
        negatives = negatives.sample(n=pos_len)
        self.dataset = pd.concat([positives, negatives], axis=0)

    def __getitem__(self, item):
        conclusion = self.dataset.iloc[item]['Conclusion']
        premise = self.dataset.iloc[item]['Premise']
        stance = self.dataset.iloc[item]['Stance']
        if stance == 'against':
            input_text = premise + self.against_phrases[random.randint(0, self.no_against_phrases)] + conclusion
        else:
            input_text = premise + self.in_favor_phrases[random.randint(0, self.no_in_favor_phrases)] + conclusion
        encoded_features = self.tokenizer(input_text,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          return_token_type_ids=True,
                                          padding="max_length",
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_tensors="pt")
        if self.mode == 'train' or self.mode == 'val':
            labels = np.asarray(self.dataset.iloc[item][4:], dtype=np.int32)
            return dict(
                input_ids=encoded_features['input_ids'].flatten(),
                token_type_ids=encoded_features['token_type_ids'].flatten(),
                attention_mask=encoded_features['attention_mask'].flatten(),
                labels=torch.FloatTensor(labels))
        else:
            return dict(
                input_ids=encoded_features['input_ids'].flatten(),
                token_type_ids=encoded_features['token_type_ids'].flatten(),
                attention_mask=encoded_features['attention_mask'].flatten())


class UnifiedTaptValueEvalDataLoader(ValueEvalDataLoader):
    def __init__(self,
                 train_dataset_file: str,
                 train_labels_file: str,
                 val_dataset_file: str,
                 val_labels_file: str,
                 batch_size: int,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int,
                 against_phrases: List[str],
                 in_favor_phrases: List[str],
                 ova: bool,
                 label_index: int,
                 downsampling: bool,
                 mode: str = 'train'):
        """
        Initialize the dataloader for the train and val sets.
        """
        self.ova = ova
        self.label_index = label_index
        self.against_phrases = against_phrases
        self.in_favour_phrases = in_favor_phrases
        self.downsampling = downsampling
        super(UnifiedTaptValueEvalDataLoader, self).__init__(train_dataset_file=train_dataset_file,
                                                             train_labels_file=train_labels_file,
                                                             val_dataset_file=val_dataset_file,
                                                             val_labels_file=val_labels_file,
                                                             batch_size=batch_size,
                                                             tokenizer=tokenizer,
                                                             max_len=max_len,
                                                             mode=mode)

    def _setup_dataloaders(self) -> None:
        """
        Set up Train and Validation Datasets.
        """
        if self.mode == 'train':
            self.train_mode = 'train'
            self.val_mode = 'val'
        else:
            self.train_mode = 'train'
            self.val_mode = 'test'
        self.train_dataset = UnifiedTaptValueEvalDataset(tokenizer=self.tokenizer,
                                                         max_len=self.max_len,
                                                         dataset_file=self.train_dataset_file,
                                                         labels_file=self.train_labels_file,
                                                         against_phrases=self.against_phrases,
                                                         in_favor_phrases=self.in_favour_phrases,
                                                         ova=self.ova,
                                                         label_index=self.label_index,
                                                         mode=self.train_mode,
                                                         downsample=self.downsampling)

        self.val_dataset = UnifiedTaptValueEvalDataset(tokenizer=self.tokenizer,
                                                       max_len=self.max_len,
                                                       dataset_file=self.val_dataset_file,
                                                       labels_file=self.val_labels_file,
                                                       against_phrases=self.against_phrases,
                                                       in_favor_phrases=self.in_favour_phrases,
                                                       ova=self.ova,
                                                       label_index=self.label_index,
                                                       mode=self.val_mode,
                                                       downsample=self.downsampling)
