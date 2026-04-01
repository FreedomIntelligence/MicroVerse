#!/bin/bash
# Full Fine-tuning for Wan2.1-T2V-14B
# Requires DeepSpeed ZeRO Stage 2 with 2+ GPUs (80GB each)

accelerate launch --config_file configs/accelerate_deepspeed_2gpu.yaml \
  train.py \
  --dataset_base_path ./data/your_dataset \
  --dataset_metadata_path ./data/your_dataset/metadata.json \
  --height 480 --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./outputs/full_14B" \
  --trainable_models "dit" \
  --use_gradient_checkpointing_offload \
  --save_steps 1000
