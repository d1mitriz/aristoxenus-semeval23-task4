import os
import torch
from transformers import AutoModel, AutoTokenizer
from model.inference import classify
from model.dataset import TaptValueEvalDataLoader
from model.model import TaptTransformerClassifier

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='roberta-base')
    pr_model = AutoModel.from_pretrained(pretrained_model_name_or_path='roberta-base')
    dataloader = TaptValueEvalDataLoader(train_dataset_file=os.path.join('data/splits', 'arguments-training.tsv'),
                                         train_labels_file=os.path.join('data/splits', 'labels-training.tsv'),
                                         val_dataset_file=os.path.join('data/splits', 'arguments-training.tsv'),
                                         val_labels_file=os.path.join('data/splits', 'labels-training.tsv'),
                                         tokenizer=tokenizer,
                                         batch_size=2,
                                         max_len=100,
                                         mode='test')
    model = TaptTransformerClassifier(model=pr_model,
                                      dropout=0.,
                                      n_classes=20,
                                      pooling=True,
                                      normalization=False)
    checkpoint = torch.load('<MODEL_PATH>')
    model.load_state_dict(checkpoint)
    model.eval()
    results = classify(test_dataloader=dataloader.train_dataloader(), model=model, device='cuda:0')
    print(results)
