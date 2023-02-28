import argparse
import yaml
import evaluate
import mlflow
import numpy as np
from pathlib import Path
from transformers import (
    TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig)
from transformers.integrations import MLflowCallback
from utils.path_helper import set_project_root_path
from utils.train_utils import set_seed
from utils.mlflow_utils import get_or_create_experiment
from task_adaption.tapt_dataset import ValueEvalTAPTDataset


def adapt_to_stance(pretrained_model: str = None):
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

    if pretrained_model is not None:
        configs.update({'pretrained_model': pretrained_model})

    print(configs['pretrained_model'])

    set_seed(configs['seed'])

    tokenizer = AutoTokenizer.from_pretrained(configs['pretrained_model'])

    train_dataset_file = configs['train_dataset_path']
    val_dataset_file = configs['val_dataset_path']

    train_dataset = ValueEvalTAPTDataset(tokenizer=tokenizer, max_len=100, dataset_file=train_dataset_file)
    val_dataset = ValueEvalTAPTDataset(tokenizer=tokenizer, max_len=100, dataset_file=val_dataset_file)

    model_config = AutoConfig.from_pretrained(configs['pretrained_model'])
    model = AutoModelForSequenceClassification.from_pretrained(
        configs['pretrained_model'], config=model_config)

    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    output_dir = str(Path(configs['model_folder']) / 'TAPT' / configs["pretrained_model"].replace('/', '--'))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        label_smoothing_factor=0.0
    )

    experiment_id = get_or_create_experiment(
        tracking_uri=configs['tracking_uri'], name=f'Task Adaption on {configs["pretrained_model"]}'
    )

    tokenizer = AutoTokenizer.from_pretrained(configs["pretrained_model"])

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback]
    )

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(training_args.to_dict())
        trainer.train()


if __name__ == '__main__':
    adapt_to_stance()
