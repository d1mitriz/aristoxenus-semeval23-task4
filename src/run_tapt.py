from task_adaption import tapt

if __name__ == '__main__':
    pretrained_models = [
        'bert-base-uncased',
        'distilbert-base-uncased',
        'albert-base-v2',
        'roberta-large',
        'sentence-transformers/all-mpnet-base-v2',
        'xlnet-base-cased',
        'sentence-transformers/all-distilroberta-v1'
    ]

    for ptm in pretrained_models:
        tapt.adapt_to_stance(ptm)
