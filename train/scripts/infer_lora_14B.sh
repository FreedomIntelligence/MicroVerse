#!/bin/bash
# Inference with LoRA checkpoint (14B)

python inference.py \
  --model_size 14B \
  --mode lora \
  --checkpoint ./outputs/lora_14B \
  --prompt "A doctor explaining a medical procedure to a patient in a clinic" \
  --output ./results/lora_14B/output.mp4
