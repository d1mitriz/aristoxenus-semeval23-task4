import pandas as pd


def augment_dataset(train_dataset_path: str,
                    train_labels_path: str,
                    multiplier: int,
                    train_aug_dataset_path: str,
                    train_aug_labels_path: str) -> None:
    """
    Copy each instance of the dataset 'multiplier' times
    :param train_dataset_path: the path of the training data
    :param train_labels_path: the path of the labels of the training data
    :param train_aug_labels_path: the path to store the augmented labels set
    :param train_aug_dataset_path: the path to store the augmented train dataset
    :param multiplier: how many times to copy each instance of the dataset
    """
    dataset = pd.read_csv(filepath_or_buffer=train_dataset_path, sep='\t', index_col=0)
    labels = pd.read_csv(filepath_or_buffer=train_labels_path, sep='\t', index_col=0)
    dataset_list = multiplier * [dataset]
    labels_list = multiplier * [labels]
    dataset = pd.concat(dataset_list)
    labels = pd.concat(labels_list)
    dataset.to_csv(path_or_buf=train_aug_dataset_path, sep='\t')
    labels.to_csv(path_or_buf=train_aug_labels_path, sep='\t')
