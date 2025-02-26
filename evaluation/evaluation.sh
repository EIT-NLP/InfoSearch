#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python models/rerankers/evaluate_reranker.py \
  --model_name_or_path /data/zyl_data/InfoSearch/wo_language \
  --output_dir result/InfoSeach \
  --batch_size 100 \
  --task_names Clarity-v1

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python models/rerankers/evaluate_reranker.py \
  --model_name_or_path /data/zyl_data/InfoSearch/wo_language \
  --output_dir result/InfoSeach \
  --batch_size 16