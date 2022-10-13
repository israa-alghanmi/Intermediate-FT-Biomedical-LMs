#  Self-Supervised Intermediate Fine-Tuning of Biomedical Language Models for Interpreting Patient Case Descriptions

This repository contains the official implementation of the paper [Self-Supervised Intermediate Fine-Tuning of Biomedical Language
Models for Interpreting Patient Case Descriptions](https://aclanthology.org/2022.coling-1.123.pdf) which has been accepted by the COLING 2022 conference.


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
@inproceedings{alghanmi-etal-2022-self,
    title = "Self-Supervised Intermediate Fine-Tuning of Biomedical Language Models for Interpreting Patient Case Descriptions",
    author = "Alghanmi, Israa  and
      Espinosa-Anke, Luis  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.123",
    pages = "1432--1441",
    abstract = "Interpreting patient case descriptions has emerged as a challenging problem for biomedical NLP, where the aim is typically to predict diagnoses, to recommended treatments, or to answer questions about cases more generally. Previous work has found that biomedical language models often lack the knowledge that is needed for such tasks. In this paper, we aim to improve their performance through a self-supervised intermediate fine-tuning strategy based on PubMed abstracts. Our solution builds on the observation that many of these abstracts are case reports, and thus essentially patient case descriptions. As a general strategy, we propose to fine-tune biomedical language models on the task of predicting masked medical concepts from such abstracts. We find that the success of this strategy crucially depends on the selection of the medical concepts to be masked. By ensuring that these concepts are sufficiently salient, we can substantially boost the performance of biomedical language models, achieving state-of-the-art results on two benchmarks.",
}

```
