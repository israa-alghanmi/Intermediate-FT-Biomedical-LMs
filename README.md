#  Self-Supervised Intermediate Fine-Tuning of Biomedical Language Models for Interpreting Patient Case Descriptions

This repository contains the official implementation of the paper [Self-Supervised Intermediate Fine-Tuning of Biomedical Language
Models for Interpreting Patient Case Descriptions](add_link_here) which has been accepted by the COLING 2022 conference.


## Pre-processing
We convert the multiple-choice question answering datasets to binary classification. 
Here is an example for MedQA:
```
python3 "preprocessing/preprocessing.py" --training "./phrases_no_exclude_train.jsonl" --dev "./phrases_no_exclude_dev.jsonl" --test "./phrases_no_exclude_test.jsonl"
```



## Evaluation

```
python3 "evaluate.py" --train "./MedQA_training.csv" --dev "./MedQA_dev.csv" --testing "./MedQA_test.csv" -seed_val=12345 --model_path "./scibert_scivocab_cased/" --cased 'cased' --pretraining_data "./SplitDis.csv"
```

## Citation
```
COMING SOON
```
