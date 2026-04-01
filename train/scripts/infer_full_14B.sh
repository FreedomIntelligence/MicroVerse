#!/bin/bash
# Inference with full fine-tuned checkpoint (14B)

python inference.py \
  --model_size 14B \
  --mode full \
  --checkpoint ./outputs/full_14B \
  --prompt "A doctor explaining a medical procedure to a patient in a clinic" \
  --output ./results/full_14B/output.mp4
