## 🔬MicroVerse

[ICLR 2026] MicroVerse: A Preliminary Exploration Toward a Micro-World Simulation

[![arXiv](https://img.shields.io/badge/arXiv-2603.00585-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2603.00585)
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2603.00585)
[![benchmark](https://img.shields.io/badge/📊-MicroWorldBench-red.svg)](eval/)
[![dataset](https://img.shields.io/badge/📦-Dataset-red.svg)]([eval/](https://huggingface.co/datasets/wangrongsheng/MicroVerse))

**MicroVerse** is the first AI framework designed to simulate microscopic biological processes at organ, cellular, and subcellular levels. Unlike existing video generation models that mostly mimic visual textures, MicroVerse integrates physical and biological constraints to faithfully reproduce the **mechanisms of life**. Leveraging the **MicroSim-10K dataset** and evaluated against **MicroWorldBench**, it sets a new benchmark for microscale AI simulations, enabling applications in biomedical research, education, and interactive visualization.

This work represents a leap from “macro-scale visual imitation” to **mechanism-aware micro-world modeling**, making AI a tool for understanding life at its most fundamental scales.

## ✨ Highlights

* **🌱 True Microscale Simulation:** Accurately models organ, cellular, and subcellular dynamics, capturing DNA replication, cell division, and apoptosis with scientific fidelity.
* **📊 Benchmarking Science:** Evaluated with **MicroWorldBench**, the first expert-annotated rubric-based test for microscopic simulations.
* **💾 Expert-Verified Dataset:** **MicroSim-10K** provides 9,601 high-resolution, semantically rich video clips guiding AI toward mechanism-level understanding.
* **🧠 Mechanism over Visuals:** Moves beyond realistic textures to **faithfully reproduce underlying biological laws and temporal dynamics**.
* **🎓 Research & Education Ready:** Enables cost-effective, interactive exploration of life’s hidden processes, bridging AI, biology, and learning.

## 📣 Latest News!!

* `[2026.05.18]` 🎉 [Dataset (9.6k Samples)](https://huggingface.co/datasets/wangrongsheng/MicroVerse) released.
* `[2026.04.01]` 🎉 [MicroWorldBench](eval/) evaluation and [MicroVerse](train/) training code released.
* `[2026.03.03]` 🚀 [Paper](https://arxiv.org/abs/2603.00585) released.
* `[2026.01.26]` 🎉 MicroVerse has been officially accepted as a **Poster at ICLR 2026**.

## 🚀 Training

We provide fine-tuning scripts for MicroVerse under the `train/` directory, supporting **LoRA** and **full parameter** training on the MicroSim-10K dataset.

**Quick Start:**

```bash
cd train
pip install -r requirements.txt

# Prepare your dataset (see train/README.md for metadata format)
# then run LoRA fine-tuning:
bash scripts/train_lora_1.3B.sh   # 1x 24GB GPU
bash scripts/train_lora_14B.sh    # 2x 24GB GPUs

# Inference after training:
python inference.py \
  --model_size 1.3B --mode lora \
  --checkpoint ./outputs/lora_1.3B \
  --prompt "DNA replication inside a eukaryotic cell nucleus" \
  --output ./results/output.mp4
```

See [`train/README.md`](train/README.md) for dataset format, GPU requirements, and all training arguments.

---

## 🧪 Evaluation (MicroWorldBench)

We provide the evaluation pipeline for **MicroWorldBench** under the `eval/` directory. It assesses generated videos across three dimensions: **Scientific Accuracy**, **Visual Quality**, and **Instruction Following**.

**Quick Start:**

```bash
# 1. Install dependencies
pip install requests opencv-python numpy tqdm

# 2. Configure API credentials
cp eval/.env.example eval/.env  # then fill in EVAL_API_KEY

# 3. Place your model's videos
mkdir eval/<ModelName>
# Add videos named by task index: 1.mp4, 2.mp4, ...

# 4. Run evaluation
cd eval
python video_evaluator.py   # Step 1: extract frames & judge
python calculate_scores.py  # Step 2: aggregate scores & report
```

Results are saved to `eval/eval_result/`. See [`eval/README.md`](eval/README.md) for full details.

## ⚙️ Acknowledgments

This project builds upon the [Wan2.1](https://github.com/Wan-Video/Wan2.1) video generation model, with its codebase based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝:

```BibTeX
@article{wang2026microverse,
  title={MicroVerse: A Preliminary Exploration Toward a Micro-World Simulation},
  author={Wang, Rongsheng and Wu, Minghao and Zhou, Hongru and Yu, Zhihan and Cai, Zhenyang and Chen, Junying and Wang, Benyou},
  journal={arXiv preprint arXiv:2603.00585},
  year={2026}
}
```
