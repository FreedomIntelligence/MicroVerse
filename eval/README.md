# MicroWorldBench Evaluation Code

Evaluation pipeline for **MicroWorldBench (MWBench)** — a benchmark for assessing AI-generated microscopic world videos across three dimensions: **Scientific Accuracy**, **Visual Quality**, and **Instruction Following**.

---

## Repository Structure

```
.
├── video_evaluator.py       # Step 1 — extract frames & call the judge model
├── calculate_scores.py      # Step 2 — aggregate scores & generate report
├── MWBenchRubrics.json      # Per-task rubrics used by video_evaluator.py
├── MWBenchRubrics_norm.json # Weighted rubrics used by calculate_scores.py
├── finaltasks.json          # Task metadata (category labels, prompts)
├── eval_result/             # Output directory
│   ├── <model>_result.json  # Raw evaluation output (auto-generated)
│   └── evaluation_report.md # Summary report (auto-generated)
└── .env.example             # Environment variable template
```

---

## Requirements

```bash
pip install requests opencv-python numpy tqdm
```

---

## Setup

Copy the environment variable template and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
# API key for the evaluation model (GPT-4o or any OpenAI-compatible endpoint)
EVAL_API_KEY=your_key_here

# Optional: override the API base URL (defaults to https://api.openai.com)
EVAL_BASE_URL=https://api.openai.com
```

Export the variables before running:

```bash
export $(cat .env | xargs)
```

---

## Usage

### Step 1 — Evaluate Videos

Place your generated videos under a subdirectory named after the model:

```
./
└── <ModelName>/
    ├── 1.mp4
    ├── 2.mp4
    └── ...
```

Video filenames must be the task index (e.g. `42.mp4`).

Edit the `model_names` list in `video_evaluator.py` (or pass via `EvalConfig`) then run:

```bash
python video_evaluator.py
```

Results are saved incrementally to `eval_result/<ModelName>_result.json`.  
Re-running skips already-completed tasks, so the script is safe to interrupt and resume.

#### Key configuration options (`EvalConfig` in `video_evaluator.py`)

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `"gpt-4o"` | Judge model name |
| `max_workers` | `100` | Parallel threads |
| `num_frames` | `8` | Frames extracted per video |
| `retry_times` | `3` | API retry attempts per request |
| `test_mode` | `False` | Set `True` to process only `test_limit` videos |
| `test_limit` | `4` | Videos to process in test mode |

---

### Step 2 — Calculate Scores

Once evaluation results exist in `eval_result/`, run:

```bash
python calculate_scores.py
```

This reads all `*_result.json` files, computes normalised scores per dimension and category, and writes `eval_result/evaluation_report.md`.

---

## Scoring Formula

For each task and dimension:

```
S      = Σ (score_i × weight_i × sign_i)
S_norm = max(0, S / Σ w_i+) × 100
```

where `score_i ∈ {0, 1}`, `sign_i ∈ {+1, −1}` (positive/penalty criterion), and `Σ w_i+` sums the weights of positive criteria only.  
The overall task score is the normalised combined score across all three dimensions.

### Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| Scientific | Accuracy of the depicted biological/physical processes |
| Visual | Realism, clarity, and quality of the rendered visuals |
| Instruction | Adherence to the task prompt and detailed requirements |

### Task Categories

| Category | Description |
|----------|-------------|
| Organ-level | Organ-scale structures and physiological processes |
| Cellular-level | Cell-scale structures and interactions |
| Subcellular-level | Subcellular organelles and molecular-level processes |

---

## Output Format

`eval_result/<ModelName>_result.json` — array of objects, one per video:

```json
{
  "index": 42,
  "video_path": "./ModelName/42.mp4",
  "scientific_eval": {
    "raw_response": "...",
    "parsed_result": { "scores": [1, 0, 1], "reasoning": "..." }
  },
  "visual_eval": { ... },
  "instruction_eval": { ... }
}
```
