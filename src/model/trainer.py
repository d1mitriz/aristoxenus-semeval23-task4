import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from transformers import get_linear_schedule_with_warmup

from model.model import TransformerClassifier, TaptTransformerClassifier
from utils.callbacks import EarlyStoppingCallback
# from src.model.dataset import ValueEvalDataLoader


class Trainer(object):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 learning_rate: float,
                 epochs: int,
                 threshold: float,
                 pooling: bool,
                 clipping: bool,
                 max_norm: float,
                 beta1: float,
                 beta2: float,
                 weight_decay: float,
                 warmup_steps: int,
                 model: Union[TransformerClassifier, TaptTransformerClassifier],
                 model_folder: str,
                 model_name: str,
                 early_stopping_cb: Optional[EarlyStoppingCallback] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = validation_dataloader
        self.lr = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.clipping = clipping
        self.max_norm = max_norm
        self.pooling = pooling
        self.beta1 = beta1
        self.beta2 = beta2
        self.betas = (beta1, beta2)
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.model = model.to(self.device)
        self.model_folder = model_folder
        os.makedirs(self.model_folder, exist_ok=True)
        self.model_name = model_name
        self.__setup_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_loss = []
        self.val_loss = []
        self.train_macro_f1 = []
        self.train_macro_precision = []
        self.train_macro_recall = []
        self.val_macro_f1 = []
        self.val_macro_precision = []
        self.val_macro_recall = []

        self.train_micro_f1 = []
        self.train_micro_precision = []
        self.train_micro_recall = []
        self.val_micro_f1 = []
        self.val_micro_precision = []
        self.val_micro_recall = []
        self.max_val_f1 = 0.  # max macro F1
        self.early_stopping = early_stopping_cb

        self.__post_init()

    def __post_init(self, avg: str = 'macro'):
        if avg == 'macro':
            self.train_f1 = self.train_macro_f1
            self.val_f1 = self.val_macro_f1

    def __setup_optimizer(self):
        params = self.model.parameters()
        self.optimizer = Adam(
            params=params,
            lr=self.lr,
            betas=self.betas,
            # weight_decay=self.weight_decay
        )
        warmup_steps = (self.warmup_steps * self.epochs) // 2  # 100 * 20 / 2 = 1000
        training_steps = 2 * (self.epochs * len(self.train_dataloader)) - warmup_steps
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=training_steps)

    def __compute_batch_loss(self,
                             model_output: torch.Tensor,
                             labels: torch.Tensor,
                             mode: str) -> None:
        loss = self.loss_fn(model_output, labels)
        if mode == 'train':
            loss.backward()
            self.train_loss.append(loss.item())
        else:
            self.val_loss.append(loss.item())

    def __compute_batch_metrics(self,
                                model_output: torch.Tensor,
                                labels: torch.Tensor,
                                mode: str) -> None:
        model_prediction = nn.Sigmoid()(model_output).cpu().data.numpy()
        numpy_labels = labels.cpu().data.numpy()
        predictions = np.where(model_prediction > self.threshold, 1, 0)
        micro_f1 = f1_score(y_true=numpy_labels, y_pred=predictions, average='micro')
        macro_f1 = f1_score(y_true=numpy_labels, y_pred=predictions, average='macro')
        micro_precision = precision_score(y_true=numpy_labels, y_pred=predictions, average='micro')
        macro_precision = precision_score(y_true=numpy_labels, y_pred=predictions, average='macro')
        micro_recall = recall_score(y_true=numpy_labels, y_pred=predictions, average='micro')
        macro_recall = recall_score(y_true=numpy_labels, y_pred=predictions, average='macro')
        if mode == 'train':
            self.train_micro_f1.append(micro_f1)
            self.train_macro_f1.append(macro_f1)
            self.train_micro_precision.append(micro_precision)
            self.train_macro_precision.append(macro_precision)
            self.train_micro_recall.append(micro_recall)
            self.train_macro_recall.append(macro_recall)
        else:
            self.val_micro_f1.append(micro_f1)
            self.val_macro_f1.append(macro_f1)
            self.val_micro_precision.append(micro_precision)
            self.val_macro_precision.append(macro_precision)
            self.val_micro_recall.append(micro_recall)
            self.val_macro_recall.append(macro_recall)

    def __update_progress_bar(self,
                              batch_step: int,
                              progress_bar: tqdm,
                              epoch: int,
                              mode: str,
                              loss: List,
                              f1: List):
        if batch_step % 100 == 0:
            progress_bar.set_description(
                'Epoch:{} - {}_loss {:.3f} | {}_micro_f1: {:.3f}'.format(epoch,
                                                                         mode,
                                                                         float(np.mean(loss)),
                                                                         mode,
                                                                         float(np.mean(f1))))

    def __on_train(self, epoch: int) -> None:
        self.model.train()
        train_progress_bar = tqdm(self.train_dataloader, total=len(self.train_dataloader))
        for batch_step, batch in enumerate(train_progress_bar):
            self.__on_train_step(batch=batch)
            self.__on_train_step_end(batch_step=batch_step,
                                     progress_bar=train_progress_bar,
                                     epoch=epoch)

    def __on_train_step(self, batch: Dict):
        self.optimizer.zero_grad()
        if isinstance(self.model, TaptTransformerClassifier):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            model_out = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
            labels = batch['labels'].to(self.device)
        else:
            p_input_ids = batch['premise_input_ids'].to(self.device)
            p_attention_mask = batch['premise_attention_mask'].to(self.device)
            c_input_ids = batch['conclusion_input_ids'].to(self.device)
            c_attention_mask = batch['conclusion_attention_mask'].to(self.device)
            stance = batch['stance']
            labels = batch['labels'].to(self.device)
            model_out = self.model(conclusion_input_ids=c_input_ids,
                                   conclusion_attention_mask=c_attention_mask,
                                   premise_input_ids=p_input_ids,
                                   premise_attention_mask=p_attention_mask,
                                   stance=stance)

        self.__compute_batch_loss(model_output=model_out,
                                  labels=labels,
                                  mode='train')
        if self.clipping:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=self.max_norm,
                                     norm_type=2)
        self.optimizer.step()
        self.scheduler.step()
        self.__compute_batch_metrics(model_output=model_out,
                                     labels=labels,
                                     mode='train')

    def __on_train_step_end(self, batch_step: int,
                            progress_bar: tqdm,
                            epoch: int):
        self.__update_progress_bar(batch_step=batch_step,
                                   progress_bar=progress_bar,
                                   epoch=epoch,
                                   mode='train',
                                   f1=self.train_f1,
                                   loss=self.train_loss)

    def __on_validation(self, epoch: int) -> None:
        self.model.eval()
        val_progress_bar = tqdm(self.val_dataloader, total=len(self.val_dataloader))
        for batch_step, batch in enumerate(val_progress_bar):
            self.__on_val_step(batch=batch)
            self.__on_val_step_end(batch_step=batch_step,
                                   progress_bar=val_progress_bar,
                                   epoch=epoch)

    @torch.no_grad()
    def __on_val_step(self, batch: Dict):
        if isinstance(self.model, TaptTransformerClassifier):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            model_out = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
            labels = batch['labels'].to(self.device)
        else:
            p_input_ids = batch['premise_input_ids'].to(self.device)
            p_attention_mask = batch['premise_attention_mask'].to(self.device)
            c_input_ids = batch['conclusion_input_ids'].to(self.device)
            c_attention_mask = batch['conclusion_attention_mask'].to(self.device)
            stance = batch['stance']
            labels = batch['labels'].to(self.device)
            model_out = self.model(conclusion_input_ids=c_input_ids,
                                   conclusion_attention_mask=c_attention_mask,
                                   premise_input_ids=p_input_ids,
                                   premise_attention_mask=p_attention_mask,
                                   stance=stance)
        self.__compute_batch_loss(model_output=model_out,
                                  labels=labels,
                                  mode='val')
        self.__compute_batch_metrics(model_output=model_out,
                                     labels=labels,
                                     mode='val')

    def __on_val_step_end(self,
                          batch_step: int,
                          progress_bar: tqdm,
                          epoch: int):
        self.__update_progress_bar(batch_step=batch_step,
                                   progress_bar=progress_bar,
                                   epoch=epoch,
                                   mode='val',
                                   f1=self.val_f1,
                                   loss=self.val_loss)

    def __on_epoch_end(self, epoch: int):
        epoch_val_micro_f1 = float(np.mean(self.val_micro_f1))
        epoch_val_micro_recall = float(np.mean(self.val_micro_recall))
        epoch_val_micro_precision = float(np.mean(self.val_micro_precision))
        epoch_val_macro_f1 = float(np.mean(self.val_macro_f1))
        epoch_val_macro_recall = float(np.mean(self.val_macro_recall))
        epoch_val_macro_precision = float(np.mean(self.val_macro_precision))
        epoch_val_loss = float(np.mean(self.val_loss))
        epoch_val_f1 = epoch_val_macro_f1

        epoch_train_micro_f1 = float(np.mean(self.train_micro_f1))
        epoch_train_micro_recall = float(np.mean(self.train_micro_recall))
        epoch_train_micro_precision = float(np.mean(self.train_micro_precision))
        epoch_train_macro_f1 = float(np.mean(self.train_macro_f1))
        epoch_train_macro_recall = float(np.mean(self.train_macro_recall))
        epoch_train_macro_precision = float(np.mean(self.train_macro_precision))
        epoch_train_loss = float(np.mean(self.train_loss))

        saved_model_path: Optional[str] = self.__maybe_save_model(
            epoch_val_macro_f1=epoch_val_macro_f1,
            keep='best_only'
        )

        self.train_loss = []
        self.val_loss = []
        self.train_macro_f1 = []
        self.train_macro_precision = []
        self.train_macro_recall = []
        self.val_macro_f1 = []
        self.val_macro_precision = []
        self.val_macro_recall = []

        self.train_micro_f1 = []
        self.train_micro_precision = []
        self.train_micro_recall = []
        self.val_micro_f1 = []
        self.val_micro_precision = []
        self.val_micro_recall = []
        self.__report_epoch(val_micro_precision=epoch_val_micro_precision,
                            val_micro_f1=epoch_val_micro_f1,
                            val_micro_recall=epoch_val_micro_recall,
                            val_macro_precision=epoch_val_macro_precision,
                            val_macro_recall=epoch_val_macro_recall,
                            val_macro_f1=epoch_val_macro_f1,
                            val_loss=epoch_val_loss,
                            train_loss=epoch_train_loss,
                            train_micro_precision=epoch_train_micro_precision,
                            train_micro_f1=epoch_train_micro_f1,
                            train_micro_recall=epoch_train_micro_recall,
                            train_macro_precision=epoch_train_macro_precision,
                            train_macro_recall=epoch_train_macro_recall,
                            train_macro_f1=epoch_train_macro_f1,
                            epoch=epoch)
        # Early stopping callback
        if self.early_stopping:
            self.early_stopping(current_loss=epoch_val_loss, current_f1=epoch_val_f1, model_dir=saved_model_path)

    def __maybe_save_model(self, epoch_val_macro_f1: float, keep: str = 'best_only') -> Optional[str]:
        if self.max_val_f1 < epoch_val_macro_f1:
            self.max_val_f1 = epoch_val_macro_f1
            already_saved_models = list(Path(self.model_folder).glob(pattern='*.pt'))
            if self.__should_save_model(already_saved_models):
                if keep == 'best_only':
                    self.__remove_saved_models(already_saved_models)
                saved_model_path = os.path.join(self.model_folder, self.model_name + f'_{self.max_val_f1}.pt')
                torch.save(obj=self.model.state_dict(), f=saved_model_path)
                return saved_model_path

    def __should_save_model(self, model_files: List[Path]) -> bool:
        _scores = [np.float64(model.parts[-1].split('_')[-1].rstrip('.pt')) for model in model_files]
        return True if not _scores or np.max(_scores) < self.max_val_f1 else False

    def __remove_saved_models(self, model_files: List[Path]) -> None:
        for model in model_files:
            model.unlink(missing_ok=True)

    def __report_epoch(self,
                       val_macro_f1: float,
                       val_macro_precision: float,
                       val_macro_recall: float,
                       val_micro_f1: float,
                       val_micro_precision: float,
                       val_micro_recall: float,
                       val_loss: float,
                       train_macro_f1: float,
                       train_macro_precision: float,
                       train_macro_recall: float,
                       train_micro_f1: float,
                       train_micro_precision: float,
                       train_micro_recall: float,
                       train_loss: float,
                       epoch: int):
        learning_rate = self.optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch} finished. '
            f'Learning rate: {learning_rate} - '
            f'train_loss: {train_loss} - '
            f'train_macro_f1: {train_macro_f1} - '
            f'validation_loss: {val_loss} - '
            f'validation_macro_f1: {val_macro_f1} - '
            f'Best validation_f1: {self.max_val_f1} - '
        )
        log_dict = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_micro_f1": train_micro_f1,
            "train_micro_precision": train_micro_precision,
            "train_micro_recall": train_micro_recall,
            "train_macro_f1": train_macro_f1,
            "train_macro_precision": train_macro_precision,
            "train_macro_recall": train_macro_recall,
            "validation_loss": val_loss,
            "validation_micro_f1": val_micro_f1,
            "val_micro_precision": val_micro_precision,
            "val_micro_recall": val_micro_recall,
            "val_macro_f1": val_macro_f1,
            "val_macro_precision": val_macro_precision,
            "val_macro_recall": val_macro_recall
        }
        # mlflow logger
        mlflow.log_metrics(metrics=log_dict, step=epoch)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.__on_train(epoch=epoch)
            self.__on_validation(epoch=epoch)
            self.__on_epoch_end(epoch=epoch)
            if self.early_stopping:
                print(self.early_stopping.epoch_summary())
                if self.early_stopping.stop():
                    print(f'Maximum patience reached. Stopping early, after {epoch} epochs.')
                    self.early_stopping.reset()
                    return self.max_val_f1
        return self.max_val_f1
