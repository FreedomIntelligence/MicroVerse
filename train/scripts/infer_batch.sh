#!/bin/bash
# Batch inference from JSON metadata

python inference.py \
  --model_size 1.3B \
  --mode lora \
  --checkpoint ./outputs/lora_1.3B \
  --metadata ./data/your_dataset/test_samples.json \
  --output ./results/batch_output
