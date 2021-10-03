# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 09:19:00 2021

@author: Shadow
"""


def main():
    
    from transformers import AutoTokenizer
    
    MODEL_CHECKPOINT = 'facebook/muppet-roberta-base'
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    
    import datasets
    DATASET = 'ag_news'
    raw_dataset = datasets.load_dataset(DATASET)
    
    COLUMN_NAME = 'text'
    def tokenize(examples):
        return tokenizer(examples[COLUMN_NAME], truncation=True)
    
    encoded_dataset = raw_dataset.map(tokenize, batched=True)
    
    print('train data shape: ', encoded_dataset["train"].shape)
    print('train val shape: ', encoded_dataset["test"].shape)
    print('dataset type: ', type(encoded_dataset))
    
    import itertools
    from torch.utils.data import Subset, Dataset
    import numpy as np
    
    '''
    train_dataset = Subset(encoded_dataset["train"], np.arange(start=0, stop=5000))
    val_dataset =  Subset(encoded_dataset["test"], np.arange(start=0, stop=3000))
    test_dataset = Subset(encoded_dataset["test"], np.arange(start=3000, stop=7600))
    '''
    
    train_dataset = encoded_dataset["train"]
    val_dataset =  encoded_dataset["test"]
    
    from transformers import AutoModelForSequenceClassification
    
    NUM_LABELS = 3
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels = NUM_LABELS)
    
    
    import numpy as np
    from sklearn.metrics import f1_score
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1_score(y_true=labels, y_pred = predictions, average='macro')
    
    from transformers import Trainer
    from transformers import TrainingArguments
    
    args = TrainingArguments(output_dir = 'results', do_eval = True)
    
    trainer = Trainer(model=model,
                      args = args,
                      train_dataset = train_dataset,
                      eval_dataset = val_dataset,
                      tokenizer = tokenizer,
                      compute_metrics=compute_metrics)
    
    trainer.train()
    
if __name__=="__main__":
    main()