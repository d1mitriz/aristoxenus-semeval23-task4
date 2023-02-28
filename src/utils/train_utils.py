import os
import json
import pathlib
import random
import warnings
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional

from transformers import BertConfig
from transformers import set_seed as set_transformers_seed


def set_warn_lvl() -> None:
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_transformers_seed(seed)


def enable_full_determinism(seed: int = 42) -> None:
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    # set seed first
    set_seed(seed)

    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def set_device(config: dict) -> None:
    if config['device'] == 'cuda' or 'gpu':
        torch.cuda.set_device(0)


def load_config(path: Union[str, Path]) -> dict:
    with open(path) as json_file:
        json_decoded = json.load(json_file)
    return json_decoded


def save_config(decoded_json: dict, path: Union[str, Path]):
    with open(path, 'w') as json_file:
        json.dump(decoded_json, json_file, indent=2, separators=(',', ': '))

