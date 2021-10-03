# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 11:27:02 2021

@author: Shadow
"""

from transformers import AutoTokenizer
import datasets
    
import itertools
from torch.utils.data import Subset, Dataset
import numpy as np
from transformers import AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score
from transformers import Trainer
from transformers import TrainingArguments
import flaml
import time
import ray
from sklearn.metrics import mean_absolute_percentage_error

def main():
    max_num_epoch = 64
    search_space = {
            # You can mix constants with search space objects.
            "num_train_epochs": flaml.tune.loguniform(1, max_num_epoch),
            "learning_rate": flaml.tune.loguniform(1e-6, 1e-4),
            "adam_epsilon": flaml.tune.loguniform(1e-9, 1e-7),
            "adam_beta1": flaml.tune.uniform(0.8, 0.99),
            "adam_beta2": flaml.tune.loguniform(98e-2, 9999e-4),
    }
    
    HP_METRIC, MODE = "matthews_correlation", "max"

    # resources
    num_cpus = 1
    num_gpus = 4
    
    # constraints
    num_samples = -1    # number of trials, -1 means unlimited
    time_budget_s = 120    # time budget in seconds
    
    start_time = time.time()
    ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    
    print("Tuning started...")
    analysis = flaml.tune.run(
        train_MuppetRoberta,
        search_alg=flaml.CFO(
            space=search_space,
            metric=HP_METRIC,
            mode=MODE,
            low_cost_partial_config={"num_train_epochs": 1}),
        report_intermediate_result=False,
        # uncomment the following if report_intermediate_result = True
        # max_resource=max_num_epoch, min_resource=1,
        resources_per_trial={"gpu": num_gpus, "cpu": num_cpus},
        local_dir='logs/',
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        use_ray=True,
    )
    
    ray.shutdown()
    
    best_trial = analysis.get_best_trial(HP_METRIC, MODE, "all")
    metric = best_trial.metric_analysis[HP_METRIC][MODE]
    print(f"n_trials={len(analysis.trials)}")
    print(f"time={time.time()-start_time}")
    print(f"Best model eval {HP_METRIC}: {metric:.4f}")
    print(f"Best model parameters: {best_trial.config}")
    return
    

def train_MuppetRoberta(config: dict):

    # Load CoLA dataset and apply tokenizer
    MODEL_CHECKPOINT = 'facebook/muppet-roberta-base'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    
    DATASET = 'ag_news'
    raw_dataset = datasets.load_dataset(DATASET)
    
    COLUMN_NAME = 'text'
    def tokenize(examples):
        return tokenizer(examples[COLUMN_NAME], truncation=True)
    
    encoded_dataset = raw_dataset.map(tokenize, batched=True)

    train_dataset = encoded_dataset["train"]
    val_dataset =  encoded_dataset["test"]
    
    
    NUM_LABELS = 4
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels = NUM_LABELS)
    
    
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1_score(y_true=labels, y_pred = predictions, average='macro')
    
    training_args = TrainingArguments(
        output_dir='.',
        do_eval=False,
        disable_tqdm=True,
        logging_steps=20000,
        save_total_limit=0,
        **config,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_output = trainer.evaluate()

    # report the metric to optimize
    flaml.tune.report(
        loss=eval_output["eval_loss"],
        matthews_correlation=eval_output["eval_matthews_correlation"],
    )
    
if __name__=="__main__":
    main()