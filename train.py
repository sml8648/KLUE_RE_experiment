import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from data_loader.data_loader import *
from utils import *
import argparse
import wandb

def main(args):

    print(args)

    wandb.init(project="klue_re_experiment", entity="sangmun")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = load_dataset(tokenizer, args.train_data)
    valid_dataset = load_dataset(tokenizer, './data/valid.csv')

    model_config =  AutoConfig.from_pretrained(args.pre_trained)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(args.pre_trained, config=model_config)
    model.parameters

    training_args = TrainingArguments(
    output_dir='./results',
    save_total_limit=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    report_to='wandb',
    run_name=args.model
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained('./best_model')

    model.eval()
    metrics = trainer.evaluate(valid_dataset)
    print("==================== Test metric score ====================")
    print("eval loss: ", metrics["eval_loss"])
    print("eval auprc: ", metrics["eval_auprc"])
    print("eval micro f1 score: ", metrics["eval_micro f1 score"])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='klue/roberta-large')
    parser.add_argument('--pre_trained', type=str, default='klue/roberta-large')
    parser.add_argument('--train_data', type=str, default='./data/train.csv')

    args = parser.parse_args()
    main(args)
