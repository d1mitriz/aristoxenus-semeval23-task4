import argparse
import optuna
from utils.configs import load_configurations
from utils.path_helper import set_project_root_path
from utils.train_utils import set_seed
from optimization.objective import Objective
from utils.callbacks import EarlyStoppingCallback


if __name__ == '__main__':

    set_project_root_path()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configuration_file',
        '-cf',
        help='The path of the configuration file',
        default="configs/optimization_configs.yaml"
    )
    parser.add_argument(
        '--starting_label',
        '-sl',
        help='Start of OVA label',
        default=0
    )
    parser.add_argument(
        '--ending_label',
        '-el',
        help='End of OVA label',
        default=20
    )
    args = parser.parse_args()
    configs = load_configurations(path=args.configuration_file)
    set_seed(configs.training.seed)

    starting_label: int = int(args.starting_label)  # ex: 0 or 10
    ending_label: int = int(args.ending_label)  # ex: 10 or 20

    for label_index in range(starting_label, ending_label):  # configs.model.n_classes
        early_stopping = EarlyStoppingCallback(patience=configs.training.early_stopping_patience) \
            if configs.training.early_stopping_patience else None
        objective = Objective(
            tracking_uri=configs.tracking.tracking_uri,
            experiment_name=configs.tracking.experiment_name + f'_{label_index}',
            train_dataset_path=configs.data.train_dataset_path,
            train_labels_path=configs.data.train_labels_path,
            val_dataset_path=configs.data.val_dataset_path,
            val_labels_path=configs.data.val_labels_path,
            against_phrases=configs.against_phrases,
            in_favor_phrases=configs.in_favor_phrases,
            pretrained_name=configs.model.pretrained_name,
            pretrained_model=configs.model.pretrained,
            n_classes=1,
            ova=configs.training.ova,
            label_index=label_index,
            max_len=configs.training.max_len,
            epochs=configs.training.epochs,
            pooling=configs.model.pooling,
            normalization=configs.model.normalization,
            beta1=configs.training.beta1,
            beta2=configs.training.beta2,
            early_stopping=early_stopping,
            tapt=configs.tapt,
            downsampling=configs.training.downsample,
            clipping=configs.model.clipping,
        )
        study = optuna.create_study(
            direction='maximize',
            study_name=configs.tracking.experiment_name
        )
        study.optimize(
            func=objective,
            n_trials=configs.optuna.n_trials,
            n_jobs=configs.optuna.n_jobs
        )
