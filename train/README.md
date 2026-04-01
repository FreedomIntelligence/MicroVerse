Fine-tuning the MicroVerse video generation model on microscopic biological simulation data, supporting both **LoRA** and **full parameter** training with automatic checkpoint resume.

## Quick Start

### 1. Installation

```bash
cd train
pip install -r requirements.txt
```

### 2. Prepare Dataset

Create a directory with your videos and a `metadata.json` file:

```
data/your_dataset/
├── metadata.json
└── videos/
    ├── video_001.mp4
    ├── video_002.mp4
    └── ...
```

The `metadata.json` should be a JSON array:

```json
[
  {
    "id": "video_001",
    "video": "videos/video_001.mp4",
    "prompt": "A detailed description of the microscopic process..."
  }
]
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Unique identifier for the sample |
| `video` | string | Yes | Relative path to the video file |
| `prompt` | string | Yes | Text description of the microscopic process |

**Video requirements:**
- Format: MP4 (H.264 recommended)
- Resolution: Any (will be resized to training resolution)
- Duration: 2–10 seconds recommended
- FPS: Any (frames will be sampled automatically)

### 3. Training

#### LoRA Training (Recommended for getting started)

```bash
# 1.3B model — works on 1× 24 GB GPU
bash scripts/train_lora_1.3B.sh

# 14B model — works on 2× 24 GB GPUs
bash scripts/train_lora_14B.sh
```

#### Full Parameter Training

```bash
# 1.3B model — requires 2× 40 GB GPUs
bash scripts/train_full_1.3B.sh

# 14B model — requires 2× 80 GB GPUs with DeepSpeed
bash scripts/train_full_14B.sh
```

Edit the scripts to set:
- `--dataset_base_path` — path to your dataset directory
- `--dataset_metadata_path` — path to your `metadata.json`
- `--output_path` — where to save checkpoints
- `--learning_rate` — learning rate (1e-4 for LoRA, 1e-5 for full)
- `--num_epochs` — number of training epochs
- `--save_steps` — checkpoint save interval

### 4. Inference

#### Single video generation

```bash
python inference.py \
  --model_size 1.3B \
  --mode lora \
  --checkpoint ./outputs/lora_1.3B \
  --prompt "DNA replication inside a eukaryotic cell nucleus" \
  --output ./results/output.mp4
```

#### Batch generation from JSON

```bash
python inference.py \
  --model_size 14B \
  --mode full \
  --checkpoint ./outputs/full_14B \
  --metadata ./data/test_samples.json \
  --output ./results/batch_output/
```

## Training Configurations

### GPU Requirements

| Mode | Model | Min GPUs | Min VRAM per GPU | Accelerate Config |
|------|-------|----------|-------------------|-------------------|
| LoRA | 1.3B | 1 | 24 GB | `accelerate_1gpu.yaml` |
| LoRA | 14B | 2 | 24 GB | `accelerate_2gpu.yaml` |
| Full | 1.3B | 2 | 40 GB | `accelerate_2gpu.yaml` |
| Full | 14B | 2 | 80 GB | `accelerate_deepspeed_2gpu.yaml` |

### LoRA vs Full Fine-tuning

| Aspect | LoRA | Full |
|--------|------|------|
| VRAM | Low | High |
| Speed | Fast | Slow |
| Checkpoint size | ~100–500 MB | ~3–27 GB |
| Quality | Good | Best |
| Recommended LR | 1e-4 | 1e-5 |
| Recommended epochs | 3–5 | 1–2 |

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--learning_rate` | 1e-4 | Learning rate |
| `--num_epochs` | 1 | Number of training epochs |
| `--height` | 480 | Video height (pixels) |
| `--width` | 832 | Video width (pixels) |
| `--num_frames` | 81 | Number of frames per video |
| `--save_steps` | None | Save checkpoint every N steps |
| `--lora_rank` | 32 | LoRA rank |
| `--lora_target_modules` | q,k,v,o,ffn.0,ffn.2 | LoRA target modules |
| `--use_gradient_checkpointing_offload` | False | Offload gradient checkpoints to CPU |
| `--dataset_repeat` | 1 | Repeat dataset N times per epoch |

Training automatically resumes from the latest checkpoint if interrupted. Checkpoints are saved as `step-{N}.safetensors` in the output directory.

## Project Structure

```
train/
├── train.py                 # Training entry point
├── inference.py             # Inference entry point
├── requirements.txt         # Python dependencies
├── configs/                 # Accelerate/DeepSpeed configs
│   ├── accelerate_1gpu.yaml
│   ├── accelerate_2gpu.yaml
│   ├── accelerate_deepspeed_2gpu.yaml
│   └── accelerate_deepspeed_4gpu.yaml
├── scripts/                 # Ready-to-use shell scripts
│   ├── train_lora_1.3B.sh
│   ├── train_lora_14B.sh
│   ├── train_full_1.3B.sh
│   ├── train_full_14B.sh
│   ├── infer_lora_1.3B.sh
│   ├── infer_lora_14B.sh
│   ├── infer_full_1.3B.sh
│   ├── infer_full_14B.sh
│   └── infer_batch.sh
├── data/                    # Dataset directory
│   └── example/
│       ├── metadata.json    # Example metadata
│       └── videos/
└── diffsynth/               # Core training/inference engine
```

## Acknowledgments

This training framework builds upon [Wan2.1](https://github.com/Wan-Video/Wan2.1) and is adapted for MicroVerse microscale biological simulation.
