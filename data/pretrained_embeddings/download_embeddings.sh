#!/bin/bash

# Script for downloading pretrained word embeddings.
# Skip downloading a model by commenting out the corresponding commands.


# BERT base, uncased
mkdir bert-base-uncased
curl https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json -o bert-base-uncased/config.json
curl https://cdn.huggingface.co/bert-base-uncased-vocab.txt -o bert-base-uncased/vocab.txt
curl https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin -o bert-base-uncased/pytorch_model.bin


# BERT large, uncased (whole word masking)
mkdir bert-large-uncased-whole-word-masking
curl https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json -o bert-large-uncased-whole-word-masking/config.json
curl https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-vocab.txt -o bert-large-uncased-whole-word-masking/vocab.txt
curl https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-pytorch_model.bin -o bert-large-uncased-whole-word-masking/pytorch_model.bin


# RoBERTa base
mkdir roberta-base
curl https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json -o roberta-base/config.json
curl https://cdn.huggingface.co/roberta-base-dict.txt -o roberta-base/dict.txt
curl https://cdn.huggingface.co/roberta-base-merges.txt -o roberta-base/merges.txt
curl https://cdn.huggingface.co/roberta-base-modelcard.json -o roberta-base/modelcard.json
curl https://cdn.huggingface.co/roberta-base-vocab.json -o roberta-base/vocab.json
curl https://cdn.huggingface.co/roberta-base-pytorch_model.bin -o roberta-base/pytorch_model.bin


# RoBERTa large
mkdir roberta-large
curl https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json -o roberta-large/config.json
curl https://cdn.huggingface.co/roberta-large-merges.txt -o roberta-large/merges.txt
curl https://cdn.huggingface.co/roberta-large-modelcard.json -o roberta-large/modelcard.json
curl https://cdn.huggingface.co/roberta-large-vocab.json -o roberta-large/vocab.json
curl https://cdn.huggingface.co/roberta-large-pytorch_model.bin -o roberta-large/pytorch_model.bin

