import mlflow
import os
from optuna import Trial
from model.trainer import Trainer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from model.model import TransformerClassifier, TaptTransformerClassifier
from model.dataset import ValueEvalDataLoader, TaptValueEvalDataLoader, UnifiedTaptValueEvalDataLoader
from utils.mlflow_utils import get_or_create_experiment
from utils.callbacks import EarlyStoppingCallback
from typing import List


class Objective:
    """
    Class describing optuna objective under optimization
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        train_dataset_path: str,
        train_labels_path: str,
        val_dataset_path: str,
        val_labels_path: str,
        ova: bool,
        label_index: int,
        pretrained_name: str,
        pretrained_model: str,
        n_classes: int,
        against_phrases: List[str],
        in_favor_phrases: List[str],
        max_len: int,
        epochs: int,
        pooling: bool,
        normalization: bool,
        beta1: float,
        beta2: float,
        early_stopping: EarlyStoppingCallback,
        tapt: bool,
        downsampling: bool,
        clipping: bool
    ) -> None:
        """Initializes objective with params that will not be optimized"""
        self.experiment_id = get_or_create_experiment(
            tracking_uri=tracking_uri,
            name=experiment_name
        )
        self.experiment_name = experiment_name
        self.pretrained_model = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model_config = AutoConfig.from_pretrained(pretrained_model)
        self.pretrained_name_exp = pretrained_name
        self.pr_model = AutoModel.from_pretrained(pretrained_model, config=self.model_config)
        self.n_classes = n_classes
        self.against_phrases = against_phrases
        self.in_favor_phrases = in_favor_phrases
        self.train_dataset_path = train_dataset_path
        self.train_labels_path = train_labels_path
        self.val_dataset_path = val_dataset_path
        self.val_labels_path = val_labels_path
        self.ova = ova
        self.label_index = label_index
        self.max_len = max_len
        self.epochs = epochs
        self.pooling = pooling
        self.normalization = normalization
        self.beta1 = beta1
        self.beta2 = beta2
        self.tapt = tapt
        self.clipping = clipping
        self.downsampling = downsampling
        self.early_stopping_cb = early_stopping
        # Create folder for specific experiment to store models inside
        if os.path.isdir(os.path.join('models', self.experiment_name, str(self.label_index))):
            raise ValueError(f'Cannot save models to models/{self.experiment_name}, path already exists. \
                Change experiment name of delete folder after backing up its contents.')
        else:
            os.makedirs(os.path.join('models', self.experiment_name, str(self.label_index)))

    def __call__(self, trial: Trial) -> float:
        """Called by study.optimize()"""
        with mlflow.start_run(experiment_id=self.experiment_id):
            return self.do_trial(trial)

    def do_trial(self, trial: Trial) -> float:
        """Runs an evaluation for a suggestion of hyperparams"""
        dropout = trial.suggest_float('dropout', 0.2, 0.25, log=True)
        lr = trial.suggest_float('lr', 1e-6, 1e-5, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, step=1e-4)
        warmup_steps = trial.suggest_int('warmup_steps', 1, 2, log=True)
        batch_size = trial.suggest_int('batch_size', 160, 224, 32)
        max_norm = trial.suggest_float('max_norm', 1.0, 3.0, step=1.0)
        threshold = trial.suggest_float('threshold', 0.3, 0.45, step=0.01)
        model_folder = f'model_{self.pretrained_name_exp}_{str(trial.number)}'
        os.mkdir(os.path.join('models', self.experiment_name, str(self.label_index), model_folder))

        params = {
            'label_index': self.label_index,
            'pretrained_model': self.pretrained_model,
            'dropout': dropout,
            'lr': lr,
            # 'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'batch_size': batch_size,
            'threshold': threshold,
            'max_norm': max_norm
        }

        mlflow.log_params(params)

        print('\nParameter'.rjust(35), 'Value')
        for param_name, param_value in params.items():
            print(f'{param_name.rjust(35)} {param_value}')

        if self.tapt:
            model = TaptTransformerClassifier(
                model=self.pr_model,
                dropout=dropout,
                n_classes=self.n_classes,
                pooling=self.pooling,
                normalization=self.normalization
            )
            dataloaders = UnifiedTaptValueEvalDataLoader(
                train_dataset_file=self.train_dataset_path,
                train_labels_file=self.train_labels_path,
                val_dataset_file=self.val_dataset_path,
                val_labels_file=self.val_labels_path,
                against_phrases=self.against_phrases,
                in_favor_phrases=self.in_favor_phrases,
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                max_len=self.max_len,
                ova=self.ova,
                label_index=self.label_index,
                downsampling=self.downsampling
            )
        else:
            model = TransformerClassifier(
                model=self.pr_model,
                dropout=dropout,
                n_classes=self.n_classes,
                pooling=self.pooling,
                normalization=self.normalization
            )
            dataloaders = ValueEvalDataLoader(
                train_dataset_file=self.train_dataset_path,
                train_labels_file=self.train_labels_path,
                val_dataset_file=self.val_dataset_path,
                val_labels_file=self.val_labels_path,
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                max_len=self.max_len
            )
        train_dataloader = dataloaders.train_dataloader()
        validation_dataloader = dataloaders.val_dataloader()
        trainer = Trainer(
            tokenizer=self.tokenizer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            epochs=self.epochs,
            pooling=self.pooling,
            clipping=self.clipping,
            max_norm=max_norm,  # self.max_norm,
            learning_rate=lr,
            beta1=self.beta1,
            beta2=self.beta2,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            threshold=threshold,  # self.threshold,
            model=model,
            model_folder=os.path.join('models', self.experiment_name, str(self.label_index), model_folder),
            model_name=f'model_label_{self.label_index}',
            early_stopping_cb=self.early_stopping_cb
        )
        max_val_f1 = trainer.train()
        mlflow.log_metrics(
            metrics={
                'max_val_f1': max_val_f1
            }
        )
        return max_val_f1
