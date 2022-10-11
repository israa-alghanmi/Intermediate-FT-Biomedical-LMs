import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from transformers import *
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import math
import os
import random
from datasets import Dataset
from sklearn.metrics import classification_report,f1_score,accuracy_score
from sklearn.metrics import average_precision_score
import argparse


def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--train', dest='training_path', type=str, default="./MedQA_train.csv",
        help='The path of training data')
    parser.add_argument('--dev', dest='dev_path', type=str, default="./MedQA_dev.csv",
        help='The path of dev data ')
    parser.add_argument('--testing', dest='testing_path', type=str, default="./MedQA_test.csv",help='The path of testing data ')
    parser.add_argument('--seed_val', dest='seed_val', type=int, default="42", help='Seed value')
    parser.add_argument('--model_path', dest='model_path', type=str, default="", help='LM path')
    parser.add_argument('--pretraining_data', dest='pretraining_data', type=str, default="", help='pretraining_data ')    
    parser.add_argument('--cased', dest='cased', type=str, default="cased", help='if the model cased or uncased ') 
    args = parser.parse_args()
    return args
    
    
def read_csv_to_df(input_path):
    df= pd.read_csv(input_path)
    df.label= df.label.apply(lambda x: int(x))
    return df
    
def prepare_data():    
    training_df= read_csv_to_df(args.training_path)
    dev_df= read_csv_to_df(args.dev_path)
    test_df= read_csv_to_df(args.testing_path)
    return test_df,training_df,dev_df
    
    
def tokenize_function(example):
    return tokenizer(example["question"], example["option"], truncation=True,pad_to_max_length=True, max_length=512)


def tokenize_function_pretrain(example):
    return tokenizer(example["p1"], example["p2"], truncation=True,pad_to_max_length=True, max_length=512)
    


def preprocess_df(df):
    df = df.replace(np.nan, '', regex=True)
    print('Case: ', args.cased)
    if args.cased=='cased':
        my_dict = {"question": list(df['question']), 'option': list(df['option']), 'labels':list(df['label']) }
    if args.cased=='uncased':
        df.question=df.question.apply(lambda x: x.lower())
        df.option=df.option.apply(lambda x: x.lower())
        my_dict = {"question": list(df['question']), 'option': list(df['option']), 'labels':list(df['label']) }
    dataset_=Dataset.from_dict(my_dict)
    dataset_=dataset_.map(tokenize_function, batched=True)
    return dataset_  

def preprocess_pretraindf(df):
    df = df.replace(np.nan, '', regex=True)
    df = df.reset_index(drop=True)
    df.mask_label= df.mask_label.apply(lambda x: int(x))
    if args.cased=='cased':
        my_dict = {"p1": list(df['p1_masked']), 'p2': list(df['p2_masked']), 'labels':list(df['mask_label']) }
    if args.cased=='uncased':
        df.p1_masked=df.p1_masked.apply(lambda x: x.lower())
        df.p2_masked=df.p2_masked.apply(lambda x: x.lower())
        my_dict = {"p1": list(df['p1_masked']), 'p2': list(df['p2_masked']), 'labels':list(df['mask_label']) } 
    dataset_=Dataset.from_dict(my_dict)
    dataset_=dataset_.map(tokenize_function_pretrain, batched=True)
    return dataset_  

def pretrain():

    pretrain_df=pd.read_csv(args.pretraining_data)
    print('Pre-training data', args.pretraining_data)
    print('Length Pre-training data', len(pretrain_df))
    pretrain_tokenized_datasets=preprocess_pretraindf(pretrain_df)
    pretrain_tokenized_datasets = pretrain_tokenized_datasets.remove_columns(["p1", "p2"])
    pretrain_tokenized_datasets.set_format("torch")
    pretrain_dataloader=DataLoader(pretrain_tokenized_datasets, shuffle=True, batch_size=train_batch_size)
    num_training_steps = num_epochs * len(pretrain_dataloader)
    warmup_steps = math.ceil(len(pretrain_dataloader) * num_epochs * 0.1)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)  
    progress_bar = tqdm(range(num_training_steps))
    print('pretraining num_epochs', num_epochs)
    model.train()
    for epoch in range(num_epochs):
        for batch in pretrain_dataloader:
            #print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    return model 
            
                
def training():
    num_epochs=4
    num_training_steps = num_epochs * len(train_dataloader)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)  
    progress_bar = tqdm(range(num_training_steps))

    print('training num_epochs', num_epochs)
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            #print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            

def testing():

    model.eval()
    predictions , true_labels = [], []
    logits_all=[]
    test_labels=[]
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictions_s = torch.argmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        label_ids = batch["labels"].to('cpu').numpy()
        logits_all.append(logits)
        predictions.append(predictions_s)
        test_labels.append(label_ids)
    
    logits_all = [item for sublist in logits_all for item in sublist]
    logits_all =[item[1] for item in logits_all ]
    test_labels = [item for sublist in test_labels for item in sublist]
    
    return logits_all,test_labels


def testing_dev():

    model.eval()
    predictions , true_labels = [], []
    logits_all=[]
    test_labels=[]
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictions_s = torch.argmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        label_ids = batch["labels"].to('cpu').numpy()
        logits_all.append(logits)
        predictions.append(predictions_s)
        test_labels.append(label_ids)
    
    logits_all = [item for sublist in logits_all for item in sublist]
    logits_all =[item[1] for item in logits_all ]
    test_labels = [item for sublist in test_labels for item in sublist]
    
    return logits_all,test_labels


def compute_scores(df):
    labels=[]
    Question_scores=[] 
    predictions_per_question=[] 
    Question_scores_by_index=[]
    predictions_by_index=[] 
    temp_labels=[]
    labels_by_index=[]
    for i in range(0,len(logits_all)):    
        label= test_labels[i]
        labels.append(label)
        temp_labels.append(label)
        Question_scores.append(logits_all[i])
        Question_scores_by_index.append(logits_all[i])
        if len(Question_scores)==4:
            Question_scores=[1 if score == max(Question_scores) else 0 for score in Question_scores] 
            predictions_per_question=predictions_per_question+Question_scores
            Question_scores=[]
            Question_scores_by_index=Question_scores_by_index.index(max(Question_scores_by_index))
            predictions_by_index.append(Question_scores_by_index)
            Question_scores_by_index=[]
            temp_labels=temp_labels.index(1)
            labels_by_index.append(temp_labels)
            temp_labels=[]
    
    
    print('<------------------------------->')   
    print("Accuracy by index:{}".format(accuracy_score(labels_by_index, predictions_by_index)))
    acc_dev.append(accuracy_score(labels_by_index, predictions_by_index))
    
if __name__ == '__main__':  
    
    if torch.cuda.is_available():        
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args = parse_arguments()

    train_batch_size=8
    num_epochs_list = [2,3,4]
    seed_val=args.seed_val
    print('seed: ', seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val) 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    test_df,training_df,dev_df=prepare_data()
    
    model_path_org=args.model_path
    print('model_path_org', model_path_org)
    tokenizer = AutoTokenizer.from_pretrained(model_path_org)
    #############prepare datasets############
    train_tokenized_datasets=preprocess_df(training_df)
    dev_tokenized_datasets=preprocess_df(dev_df)
    test_tokenized_datasets=preprocess_df(test_df)
    train_tokenized_datasets3 = train_tokenized_datasets.remove_columns(["question", "option"])
    train_tokenized_datasets3.set_format("torch")
    dev_tokenized_datasets3 = dev_tokenized_datasets.remove_columns(["question", "option"])
    dev_tokenized_datasets3.set_format("torch")
    test_tokenized_datasets3 = test_tokenized_datasets.remove_columns(["question", "option"])
    test_tokenized_datasets3.set_format("torch")
    ###############prepare dataloaders############################
    train_dataloader=DataLoader(train_tokenized_datasets3, shuffle=True, batch_size=train_batch_size)
    eval_dataloader=DataLoader(dev_tokenized_datasets3, shuffle=False, batch_size=train_batch_size) 
    test_dataloader=DataLoader(test_tokenized_datasets3, shuffle=False, batch_size=train_batch_size)
    acc_dev=[]
    models=[]
    for num_epochs in num_epochs_list:
    #################################################
        model = AutoModelForSequenceClassification.from_pretrained(model_path_org, num_labels=2)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.to(device)
        model= pretrain()
        optimizer = AdamW(model.parameters(), lr=2e-5)
        training()
        logits_all,test_labels=testing_dev()
        compute_scores(dev_df)
        models.append(model)
    #################################################
    
    max_acc_index=np.argmax(acc_dev)
    model=models[max_acc_index]
    logits_all,test_labels=testing()
    compute_scores(test_df)
    
    
    
    