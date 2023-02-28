"""A module to define functions to return the predicted classes"""
from tqdm import tqdm
import numpy as np
from model.model import TransformerClassifier
from model.dataset import TaptValueEvalDataLoader
from typing import List, Union
import torch
from torch.utils.data import DataLoader


def classify(test_dataloader: TaptValueEvalDataLoader,
             model: TransformerClassifier,
             device: str) -> Union[np.ndarray, List]:
    """
    Classify instances feedforwarding to the model
    :param test_dataloader: stores the test set in a Pytorch Dataloader object
    :param model: the trained model
    :param device: the device (cpu or gpu)
    :return a list with the predicted tags
    """
    batches = len(test_dataloader)
    predictions = []
    model.to(device)
    for batch in tqdm(test_dataloader, total=batches):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        model_out = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        prediction_output = torch.sigmoid(model_out).squeeze()
        if device == 'cpu':
            prediction_output = prediction_output.detach().numpy()
        else:
            prediction_output = prediction_output.cpu().detach().numpy()
        predictions += prediction_output.tolist()
    return predictions  # np.asarray(predictions)


def ensemble_inference(models: List[TransformerClassifier],
                       test_dataloader: TaptValueEvalDataLoader,
                       device: str) -> np.ndarray:
    """
    Classify instances using voting technique
    :param test_dataloader: stores the test set in a Pytorch Dataloader object
    :param models: a list with the trained models
    :param device: the device (cpu or gpu)
    :return a list with the predicted tags
    """
    num_models = len(models)
    predictions = []
    for model in models:
        model_prediction = classify(test_dataloader=test_dataloader,
                                    model=model,
                                    device=device)
        predictions.append(model_prediction)
    predictions = np.asarray(predictions)  # .squeeze() #we need the binaries to decide if it is necessary
    """
    predictions:
    [model1[[], [], [], ...],
     model2[[], [], [], ...]
    """

    ensemble_prediction = np.zeros((predictions.shape[1], predictions.shape[2]))  # shape(test_size, labels)
    for model_pred in predictions:
        ensemble_prediction += model_pred
    ensemble_prediction /= num_models
    return ensemble_prediction
