#!/bin/bash
# Full Fine-tuning for Wan2.1-T2V-1.3B

accelerate launch --config_file configs/accelerate_2gpu.yaml \
  train.py \
  --dataset_base_path ./data/your_dataset \
  --dataset_metadata_path ./data/your_dataset/metadata.json \
  --height 480 --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./outputs/full_1.3B" \
  --trainable_models "dit" \
  --use_gradient_checkpointing_offload \
  --save_steps 5000
