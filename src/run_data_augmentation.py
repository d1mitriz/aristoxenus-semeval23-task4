from data.augment_dataset import augment_dataset

if __name__ == '__main__':
    augment_dataset(train_dataset_path='../data/splits/arguments-training.tsv',
                    train_labels_path='../data/splits/labels-training.tsv',
                    multiplier=4,
                    train_aug_dataset_path='../data/splits/augmented_argument-training.tsv',
                    train_aug_labels_path='../data/splits/augmented_labels-training.tsv')
