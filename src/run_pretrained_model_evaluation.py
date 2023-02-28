import os
import json
import argparse
import yaml
import mlflow
from dotenv import load_dotenv, set_key
from transformers import AutoModel, AutoTokenizer, AutoConfig, logging
from utils.path_helper import set_project_root_path
from utils.train_utils import set_seed
from utils.mlflow_utils import get_or_create_experiment
from model.dataset import ValueEvalDataLoader, TaptValueEvalDataLoader, UnifiedTaptValueEvalDataLoader
from model.trainer import Trainer
from model.model import TransformerClassifier, TaptTransformerClassifier
from utils.callbacks import EarlyStoppingCallback
from pathlib import Path


pretrained_models = [
    'bert-base-uncased',
    'distilbert-base-uncased',
    'albert-base-v2',
    'sentence-transformers/all-mpnet-base-v2',
    'xlnet-base-cased',
    'sentence-transformers/all-distilroberta-v1',
    'roberta-large'
]

pretrained_tapt_models = [
    f'models/TAPT/{ptm.replace("/", "--")}/checkpoint-219' for ptm in pretrained_models
]

_params = ["batch_size", "dropout", "label_index", "lr", "max_norm", "threshold", "warmup_steps"]
downsample_on = [0, 1, 4, 6, 9, 11, 18, 19]


def update_config(_label_index: int, _configs: dict) -> dict:
    with open(f'models/best_models_per_label/{label_index}/run_config.json') as per_label_config:
        best_params = json.load(per_label_config)

    params = {k: best_params[k] for k in _params}
    params['downsampling'] = True if _label_index in downsample_on else False
    for param, val in params.items():
        _configs.update({param: val})
    return _configs


def train_ova(__configs: dict, __label_index: int) -> None:

    set_seed(__configs['seed'])

    tokenizer = AutoTokenizer.from_pretrained(__configs['pretrained_model'])
    pr_model = AutoModel.from_pretrained(__configs['pretrained_model'])

    if __configs['tapt']:
        model = TaptTransformerClassifier(
            model=pr_model,
            dropout=__configs['dropout'],
            n_classes=__configs['n_classes'],
            pooling=__configs['pooling'],
            normalization=__configs['normalization']
        )
        dataloaders = UnifiedTaptValueEvalDataLoader(
            train_dataset_file=__configs['train_dataset_path'],
            train_labels_file=__configs['train_labels_path'],
            val_dataset_file=__configs['val_dataset_path'],
            val_labels_file=__configs['val_labels_path'],
            tokenizer=tokenizer,
            batch_size=__configs['batch_size'],
            max_len=__configs['max_len'],
            against_phrases=__configs['against_phrases'],
            in_favor_phrases=__configs['in_favor_phrases'],
            ova=__configs['ova'],
            downsampling=__configs['downsampling'],
            label_index=__label_index  # __configs['label_index'] works too, is updated in the config
        )
    else:
        model = TransformerClassifier(
            model=pr_model,
            dropout=__configs['dropout'],
            n_classes=__configs['n_classes'],
            pooling=__configs['pooling'],
            normalization=__configs['normalization']
        )
        dataloaders = ValueEvalDataLoader(
            train_dataset_file=__configs['train_dataset_path'],
            train_labels_file=__configs['train_labels_path'],
            val_dataset_file=__configs['val_dataset_path'],
            val_labels_file=__configs['val_labels_path'],
            tokenizer=tokenizer,
            batch_size=__configs['batch_size,'],
            max_len=__configs['max_len']
        )
    train_dataloader = dataloaders.train_dataloader()
    validation_dataloader = dataloaders.val_dataloader()

    early_stopping = EarlyStoppingCallback(patience=__configs['early_stopping_patience']) \
        if __configs['early_stopping_patience'] else None

    # Create folder for specific experiment to store models inside
    os.makedirs(__configs['model_folder'], exist_ok=True)

    trainer = Trainer(tokenizer=tokenizer,
                      train_dataloader=train_dataloader,
                      validation_dataloader=validation_dataloader,
                      epochs=__configs['epochs'],
                      pooling=__configs['pooling'],
                      clipping=__configs['clipping'],
                      max_norm=__configs['max_norm'],
                      learning_rate=__configs['lr'],
                      beta1=__configs['beta1'],
                      beta2=__configs['beta2'],
                      weight_decay=__configs['weight_decay'],
                      warmup_steps=__configs['warmup_steps'],
                      threshold=__configs['threshold'],
                      model=model,
                      model_folder=__configs['model_folder'],
                      model_name=__configs['model_name'],
                      early_stopping_cb=early_stopping)

    experiment_id = get_or_create_experiment(
        tracking_uri=__configs['tracking_uri'], name=__configs['experiment_name']
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name=__configs['run_name']):
        mlflow.log_params(__configs)
        trainer.train()


if __name__ == '__main__':

    set_project_root_path()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configuration_file',
        '-cf',
        help='The path of the configuration file',
        default="configs/train_configs.yaml")
    args = parser.parse_args()
    configuration_file = str(args.configuration_file)

    with open(configuration_file, 'r') as configs_file:
        configs = yaml.safe_load(configs_file)

    if configs['tracking_uri'].startswith('http'):
        dotenv_file = 'configs/.env'
        load_dotenv(dotenv_file)

        # Set the experiment name env from the config file (will write to file)
        set_key(
            dotenv_path=dotenv_file,
            key_to_set="MLFLOW_EXPERIMENT_NAME",
            value_to_set=configs['experiment_name']
        )

    labels = 20
    all_pretrained_models = pretrained_models + pretrained_tapt_models
    all_pretrained_models.pop(0)  # Ignore bert-base-cased since it run successfully
    for ptm in all_pretrained_models:
        print(f'Training using {ptm}...')
        ptm_escaped = ptm.replace("/", "--")
        for label_index in range(labels):
            # use best params per label from config file
            configs = update_config(label_index, configs)
            if 'large' in ptm and configs['batch_size'] > 160:
                configs.update({
                    "batch_size": 160
                })
            # update pretrained model related configs
            configs.update({
                'pretrained_model': ptm,
                'model_folder': f'models/multilabel_train/pretrained_eval/{ptm_escaped}/{label_index}',
                'model_name': f'{ptm_escaped}_label_{label_index}_',
                'experiment_name': f'Pretrained Eval {ptm_escaped}',
                'run_name': f'{ptm_escaped}_label_{label_index}'
            })

            train_ova(configs, label_index)
