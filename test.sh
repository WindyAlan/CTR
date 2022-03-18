#!/bin/bash

export TRANSFORMERS_CACHE='./model_cache'
export HF_DATASETS_CACHE='./dataset_cache'



python run.py  --bert_model 'bert-base-uncased' --backbone bert_adapter --baseline ctr --task dsc --eval_batch_size 64 --train_batch_size 32 --scenario til_classification --idrandom 0 --use_predefine_args

# python run.py \  
# --bert_model 'bert-base-uncased' \   
# --backbone bert_adapter \  
# --baseline ctr \  
# --task dsc \  
# --eval_batch_size 128 \  
# --train_batch_size 32 \  
# --scenario til_classification \  
# --idrandom 0  \
# --use_predefine_args

# python  run.py \  
#     --bert_model 'bert-base-uncased' \   
#     --backbone bert_adapter \  
#     --baseline ctr \  
#     --task asc \  
#     --eval_batch_size 128 \  
#     --train_batch_size 32 \  
#     --scenario til_classification \  
#     --idrandom 0  \
#     --use_predefine_args