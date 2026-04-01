#!/bin/bash
# Inference with full fine-tuned checkpoint (1.3B)

python inference.py \
  --model_size 1.3B \
  --mode full \
  --checkpoint ./outputs/full_1.3B \
  --prompt "A doctor explaining a medical procedure to a patient in a clinic" \
  --output ./results/full_1.3B/output.mp4
