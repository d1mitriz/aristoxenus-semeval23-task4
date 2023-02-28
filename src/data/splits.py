import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix


def label_combination_dist(y):
    return Counter(combination for row in get_combination_wise_output_matrix(y, order=2) for combination in row)


def stratified_split(x, y, y_split_size: float = 0.1):
    return iterative_train_test_split(x, y, test_size=y_split_size)


def split_dataset(configs_file: str = 'train_configs.yaml'):
    root = Path('../..')
    configuration_file = root / 'configs' / configs_file

    with open(configuration_file, 'r') as cf:
        configs: dict = yaml.safe_load(cf)

    dataset_folder = configs.get('dataset_folder', 'data').split('/')[0]

    dataset_file: Path = root / dataset_folder / configs['train_dataset_file']
    labels_file: Path = root / dataset_folder / configs['train_labels_file']

    dataset = pd.read_csv(filepath_or_buffer=dataset_file, sep='\t')
    labels = pd.read_csv(filepath_or_buffer=labels_file, sep='\t', index_col=0)
    # labels_only = labels.iloc[:, 1:]

    train_data, train_labels, val_data, val_labels = stratified_split(dataset.to_numpy(), labels.to_numpy())

    # Insert the Argument ID column back in to the numpy array
    train_labels = np.insert(train_labels.astype(object), 0, train_data[:, 0], axis=-1)
    val_labels = np.insert(val_labels.astype(object), 0, val_data[:, 0], axis=-1)

    for np_array, name in zip(
            [train_data, train_labels, val_data, val_labels],
            ['arguments-training', 'labels-training', 'arguments-val', 'labels-val']
    ):
        _path = root / 'data' / 'splits' / f'{name}.tsv'
        pd.DataFrame(np_array).to_csv(
            path_or_buf=_path,
            sep='\t',
            header=dataset.columns if name.split('-')[0] == 'arguments' else labels.columns.insert(0, 'Argument ID'),
            index=False,
            encoding='utf8'
        )
        x, task = name.split("-")
        print(f'Created {task} {x} split at {str(_path.resolve())}.')
    print('Done!')


if __name__ == '__main__':
    split_dataset()
