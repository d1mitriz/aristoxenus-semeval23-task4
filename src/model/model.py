from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel

class TransformerClassifier(nn.Module):
    """
    A general multi-class classifier that utilize transformer LMs as backbone architecture.
    """

    def __init__(self,
                 model: PreTrainedModel,
                 n_classes: int,
                 pooling: bool,
                 normalization: bool = True,
                 dropout: float = 0.2) -> None:
        """
        :param model: The pretrained model which will be used as the backbone model
        :param dropout: the dropout ration
        :param normalization: decide if we will apply l2 normalization on the output of the pretrained model
        :param n_classes: number of classes
        """
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(p=dropout)
        self.pooling = pooling
        self.normalization = normalization
        #self.classifier = nn.Linear(3*self.model.config.hidden_size, n_classes)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self,
                conclusion_input_ids: torch.Tensor,
                conclusion_attention_mask: torch.Tensor,
                premise_input_ids: torch.Tensor,
                premise_attention_mask: torch.Tensor,
                stance: str) -> torch.Tensor:
        """
        Feedforward inputs to the model. The input here is text span provided by DataLoaders
        :param conclusion_input_ids: the input ids of the conclusion
        :param conclusion_attention_mask: the attetion mask of the conclusion
        :param premise_attention_mask: the attention mask of the premise
        :param premise_input_ids: the input ids of the premise
        :param stance: the stance of the argument
        """
        premise_output = self.model(input_ids=premise_input_ids, attention_mask=premise_attention_mask)
        conclusion_output = self.model(input_ids=conclusion_input_ids, attention_mask=conclusion_attention_mask)
        if self.pooling:
            premise_pooler_output = self.dropout(self.mean_pooling(model_output=premise_output,
                                                                   attention_mask=premise_attention_mask))
            conclusion_pooler_output = self.dropout(self.mean_pooling(model_output=conclusion_output,
                                                                      attention_mask=conclusion_attention_mask))
        else:
            premise_pooler_output = self.dropout(premise_output.pooler_output)
            conclusion_pooler_output = self.dropout(conclusion_output.pooler_output)
        if self.normalization:
            premise_pooler_output = nn.functional.normalize(input=premise_pooler_output)
            conclusion_pooler_output = nn.functional.normalize(input=conclusion_pooler_output)
        if stance == 'in favor of':
            stance_output = conclusion_pooler_output - premise_pooler_output
        else:
            stance_output = conclusion_pooler_output + premise_pooler_output
        #concat_output = torch.cat((conclusion_pooler_output,
        #                           premise_pooler_output,
        #                           stance_output), dim=1)
        #classifier_output = self.classifier(concat_output)
        classifier_output = self.classifier(stance_output)
        return classifier_output

    def mean_pooling(self,
                     model_output: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean Pooling - Take attention mask into account for correct averaging
        :param model_output:
        :param attention_mask:
        :return:
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def parameters(self) -> List[torch.Tensor]:
        """
        Return the model parameters.
        :return: Returns a list containing all the parameters of the model
        """
        return [param for _, param in self.model.named_parameters() if param.requires_grad]


class TaptTransformerClassifier(TransformerClassifier):
    def __init__(self,
                 model: PreTrainedModel,
                 n_classes: int,
                 pooling: bool,
                 normalization: bool = True,
                 dropout: float = 0.2) -> None:
        """
        :param model: The pretrained model which will be used as the backbone model
        :param dropout: the dropout ration
        :param normalization: decide if we will apply l2 normalization on the output of the pretrained model
        :param n_classes: number of classes
        """
        super(TaptTransformerClassifier, self).__init__(model=model,
                                                        n_classes=n_classes,
                                                        pooling=pooling,
                                                        normalization=normalization,
                                                        dropout=dropout)

    def forward(self, # noqa
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Tapt transformer classifier
        :param input_ids: the input ids of the input text
        :param attention_mask: the attention_mask of the input text
        """
        try:
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        except TypeError:
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask)
        if self.pooling:
            pooler_output = self.dropout(self.mean_pooling(model_output=output,
                                                           attention_mask=attention_mask))
        else:
            pooler_output = self.dropout(output.pooler_output)
        if self.normalization:
            pooler_output = nn.functional.normalize(input=pooler_output)
        classifier_output = self.classifier(pooler_output)
        return classifier_output
