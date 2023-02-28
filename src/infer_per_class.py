import sys

import yaml
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from model.inference import classify
from model.dataset import UnifiedTaptValueEvalDataLoader
from model.model import (TaptTransformerClassifier,
                         TaptTransformerForAttentionPoolingClassifier,
                         TaptTransformerForMeanMaxPoolingClassifier)

from utils.path_helper import set_project_root_path


_params = ["batch_size", "dropout", "label_index", "lr", "max_norm", "threshold", "warmup_steps"]
downsample_on = [0, 1, 4, 6, 9, 11, 18, 19]


def update_config(_label_index: int, _configs: dict) -> dict:
    with open(f'models/best_models_per_label/{_label_index}/run_config.json') as per_label_config:
        best_params = json.load(per_label_config)

    params = {k: best_params[k] for k in _params}
    params['downsampling'] = True if _label_index in downsample_on else False
    for param, val in params.items():
        _configs.update({param: val})
    return _configs


def infer_ova(__configs: dict, pr_model, tokenizer, __label_index: int, argument_ids):
    model = TaptTransformerClassifier(model=pr_model,
                                          dropout=__configs['dropout'],
                                          n_classes=1,
                                          pooling=__configs['pooling'],
                                          normalization=__configs['normalization'])

    device = 'cuda'  # 'cpu'
    dataloader = UnifiedTaptValueEvalDataLoader(
        train_dataset_file=__configs['train_dataset_path'],
        train_labels_file=__configs['train_labels_path'],
        val_dataset_file=__configs['val_dataset_path'],
        val_labels_file=__configs['val_labels_path'],
        tokenizer=tokenizer,
        batch_size=__configs['batch_size'],
        max_len=__configs['max_len'],
        against_phrases=__configs['against_phrases'],
        in_favor_phrases=__configs['in_favor_phrases'],
        downsampling=__configs['downsampling'],
        ova=__configs['ova'],
        label_index=label_idx,
        mode='test'
    )
    if argument_ids is None:
        argument_ids = dataloader.val_dataset.dataset["Argument ID"].to_numpy(dtype=str).reshape(-1, 1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        print(f'Prediction for label {label_idx}: ')
        results = classify(test_dataloader=dataloader.val_dataloader(), model=model, device=device)
    label_preds = [1 if pred > __configs['threshold'] else 0 for pred in results]

    return label_preds, argument_ids


if __name__ == '__main__':
    set_project_root_path()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configuration_file',
        '-cf',
        help='The path of the configuration file',
        default="configs/infer.yaml")
    args = parser.parse_args()
    configuration_file = str(args.configuration_file)

    with open(configuration_file, 'r') as configs_file:
        configs = yaml.safe_load(configs_file)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=configs['pretrained_model'])
    pr_model = AutoModel.from_pretrained(pretrained_model_name_or_path=configs['pretrained_model'])

    argument_ids = None

    test_preds = []
    for label_idx in range(20):
        model_parent_folder = Path(configs['model_folder']) / str(label_idx)
        model_path = list(model_parent_folder.rglob('*.pt'))[0].resolve()
        model_run_config = list(model_parent_folder.rglob('*.json'))[0].resolve()
        _configs = update_config(label_idx, configs)
        output, argument_ids = infer_ova(_configs, pr_model, tokenizer, label_idx, argument_ids)
        test_preds.append(output)

    final_preds = np.asarray(test_preds, dtype=str).T
    final_preds = np.hstack((argument_ids, final_preds))

    headers = ["Argument ID"] + configs["labels_column"]
    np.savetxt(
        fname=(Path(configs['results_folder']) / configs['results_file_name']).resolve(),
        X=final_preds,
        fmt='%s',  # save as int (defaults to float)
        delimiter='\t',
        header='\t'.join(headers),
        comments=''
    )
