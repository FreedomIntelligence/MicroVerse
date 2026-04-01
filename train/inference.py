"""Inference script for Wan2.1 fine-tuned models (LoRA and full fine-tuning)."""
import torch
import argparse
import json
import os
import glob
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict


DEFAULT_NEGATIVE_PROMPT = (
    "Overexposed, static, blurry details, subtitles, watermark, "
    "worst quality, low quality, JPEG artifacts, ugly, deformed, "
    "extra fingers, poorly drawn hands, poorly drawn face, mutated, "
    "fused fingers, frozen frame, cluttered background"
)


def load_pipeline(model_size: str, device: str = "cuda"):
    model_id = f"Wan-AI/Wan2.1-T2V-{model_size}"
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
    )
    return pipe


def load_checkpoint(pipe, checkpoint_path: str, mode: str = "lora"):
    if mode == "lora":
        print(f"Loading LoRA weights from {checkpoint_path}")
        pipe.load_lora(pipe.dit, checkpoint_path, alpha=1)
    else:
        print(f"Loading full model weights from {checkpoint_path}")
        state_dict = load_state_dict(checkpoint_path)
        pipe.dit.load_state_dict(state_dict)
    return pipe


def find_latest_checkpoint(output_path: str):
    step_ckpts = sorted(glob.glob(os.path.join(output_path, "step-*.safetensors")))
    epoch_ckpts = sorted(glob.glob(os.path.join(output_path, "epoch-*.safetensors")))
    all_ckpts = step_ckpts + epoch_ckpts
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoint found in {output_path}")
    return all_ckpts[-1]


def generate_single(pipe, prompt, output_path, negative_prompt=None, seed=42):
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        tiled=True,
    )
    save_video(video, output_path, fps=15, quality=5)
    print(f"Video saved to {output_path}")


def generate_batch(pipe, metadata_path, output_dir, negative_prompt=None, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path, "r") as f:
        samples = json.load(f)

    print(f"Generating {len(samples)} videos...")
    for i, sample in enumerate(samples):
        video_id = sample.get("id", f"{i:05d}")
        prompt = sample.get("prompt", "")
        output_path = os.path.join(output_dir, f"{video_id}.mp4")

        if os.path.exists(output_path):
            print(f"[{i+1}/{len(samples)}] Skipping {video_id} (exists)")
            continue

        print(f"[{i+1}/{len(samples)}] Generating {video_id}...")
        try:
            generate_single(pipe, prompt, output_path, negative_prompt, seed)
        except Exception as e:
            print(f"  ERROR: {e}")

    generated = len(glob.glob(os.path.join(output_dir, "*.mp4")))
    print(f"Done! Generated {generated}/{len(samples)} videos in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Wan2.1 Video Inference")
    parser.add_argument("--model_size", type=str, required=True, choices=["1.3B", "14B"], help="Model size")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file or directory")
    parser.add_argument("--mode", type=str, required=True, choices=["lora", "full"], help="Training mode")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for generation")
    parser.add_argument("--metadata", type=str, default=None, help="Path to JSON metadata for batch generation")
    parser.add_argument("--output", type=str, default="./output", help="Output path (file for single, directory for batch)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    if args.prompt is None and args.metadata is None:
        parser.error("Either --prompt or --metadata must be provided")

    # Load pipeline
    print(f"Loading Wan2.1-T2V-{args.model_size} pipeline...")
    pipe = load_pipeline(args.model_size, args.device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = find_latest_checkpoint(ckpt_path)
        print(f"Using latest checkpoint: {ckpt_path}")
    pipe = load_checkpoint(pipe, ckpt_path, args.mode)

    # Generate
    if args.prompt:
        output_path = args.output if args.output.endswith(".mp4") else os.path.join(args.output, "output.mp4")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        generate_single(pipe, args.prompt, output_path, args.negative_prompt, args.seed)
    else:
        generate_batch(pipe, args.metadata, args.output, args.negative_prompt, args.seed)


if __name__ == "__main__":
    main()
