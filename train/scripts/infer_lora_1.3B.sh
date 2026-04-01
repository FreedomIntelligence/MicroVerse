#!/bin/bash
# Inference with LoRA checkpoint (1.3B)

python inference.py \
  --model_size 1.3B \
  --mode lora \
  --checkpoint ./outputs/lora_1.3B \
  --prompt "A doctor explaining a medical procedure to a patient in a clinic" \
  --output ./results/lora_1.3B/output.mp4
