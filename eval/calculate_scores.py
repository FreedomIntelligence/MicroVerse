#!/usr/bin/env python3
"""
calculate_scores.py
-------------------
Aggregates per-video evaluation results produced by video_evaluator.py into
summary statistics and writes a Markdown report to eval_result/evaluation_report.md.

Scoring formula
---------------
For each task and each dimension (Scientific / Visual / Instruction):

    S      = sum(score_i * weight_i * sign_i)   where sign_i ∈ {+1, -1}
    S_norm = max(0, S / sum(w_i for positive items)) * 100

The overall task score is the normalised sum across all three dimensions combined.
Tasks missing any dimension are skipped.

Input files
-----------
eval_result/<model>_result.json  - Raw evaluation output from video_evaluator.py
MWBenchRubrics_norm.json         - Rubrics with per-criterion weights
finaltasks.json                  - Task metadata including category labels
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _calculate_dimension_score(
    scores: List[int],
    rubrics: Dict[str, Any],
    dimension: str,
) -> Tuple[float, float]:
    """
    Return (raw_score, max_possible) for one dimension of one task.

    Parameters
    ----------
    scores    : Binary list returned by the evaluator (0 or 1 per criterion).
    rubrics   : Full rubric entry for the task.
    dimension : One of 'scientific', 'visual', 'instruction'.
    """
    rubric_key = f"rubrics_{dimension}"
    dim_rubrics = rubrics.get(rubric_key)
    if not dim_rubrics or not scores:
        return 0.0, 0.0

    # Collect valid criterion indices
    criterion_indices = {
        int(k[1:]) for k in dim_rubrics if k.startswith("s") and k[1:].isdigit()
    }

    raw_score = 0.0
    max_possible = 0.0

    for i, score in enumerate(scores):
        n = i + 1
        if n not in criterion_indices:
            continue
        s_val_str = str(dim_rubrics.get(f"s{n}", "0"))
        w_val = float(dim_rubrics.get(f"w{n}", 0))
        sign = 1 if s_val_str == "+1" else -1 if s_val_str == "-1" else int(s_val_str)

        if score == 1:
            raw_score += sign * w_val
        if sign > 0:
            max_possible += w_val

    return raw_score, max_possible


def _normalise(raw: float, max_possible: float) -> float:
    """Normalise raw score to [0, 100]; clamp negatives to 0."""
    if max_possible <= 0:
        return 0.0
    return max(0.0, raw / max_possible * 100)


# ---------------------------------------------------------------------------
# Per-model aggregation
# ---------------------------------------------------------------------------

DIMENSIONS = [
    ("scientific_eval",  "scientific"),
    ("visual_eval",      "visual"),
    ("instruction_eval", "instruction"),
]

CATEGORIES = ["Organ-level", "Cellular-level", "Subcellular-level"]


def _aggregate_model(
    model_data: List[Dict[str, Any]],
    rubrics_index: Dict[int, Any],
    tasks_index: Dict[int, str],
) -> Dict[str, Any]:
    """Return aggregated score lists for one model."""
    stats: Dict[str, Any] = {
        "valid_tasks": 0,
        "total_scores": [],
        "scientific_scores": [],
        "visual_scores": [],
        "instruction_scores": [],
        "category_scores": {c: [] for c in CATEGORIES},
    }

    for item in model_data:
        task_idx = item.get("index")
        if task_idx not in rubrics_index:
            continue

        rubrics = rubrics_index[task_idx]
        category = tasks_index.get(task_idx, "Unknown")

        # Require all three dimensions to have valid parsed results
        dim_scores: List[float] = []
        task_raw = 0.0
        task_max = 0.0
        valid = True

        for eval_key, dim_name in DIMENSIONS:
            eval_data = item.get(eval_key, {})
            parsed = eval_data.get("parsed_result") if eval_data else None
            if not parsed:
                valid = False
                break
            scores = parsed.get("scores", [])
            if not scores:
                valid = False
                break

            raw, mx = _calculate_dimension_score(scores, rubrics, dim_name)
            dim_scores.append(_normalise(raw, mx))
            task_raw += raw
            task_max += mx

        if not valid or len(dim_scores) != 3:
            continue

        stats["valid_tasks"] += 1
        task_total = _normalise(task_raw, task_max)
        stats["total_scores"].append(task_total)
        stats["scientific_scores"].append(dim_scores[0])
        stats["visual_scores"].append(dim_scores[1])
        stats["instruction_scores"].append(dim_scores[2])
        if category in stats["category_scores"]:
            stats["category_scores"][category].append(task_total)

    return stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _avg(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _generate_report(results: Dict[str, Any], output_path: str) -> None:
    """Write a Markdown summary report."""
    total_tasks = 459  # Expected benchmark size

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("# MicroWorldBench Model Evaluation Report\n\n")

        # --- Coverage table ---
        f.write("## Coverage\n\n")
        f.write("| Model | Valid Tasks | Total Tasks | Coverage |\n")
        f.write("|-------|-------------|-------------|----------|\n")
        for name, data in sorted(results.items()):
            rate = data["valid_tasks"] / total_tasks * 100 if total_tasks else 0
            f.write(f"| {name} | {data['valid_tasks']} | {total_tasks} | {rate:.1f}% |\n")
        f.write("\n")

        # --- Table 1: overall + category breakdown ---
        f.write("## Table 1: Overall and Category Scores\n\n")
        f.write("| Model | Average | Organ-level | Cellular-level | Subcellular-level |\n")
        f.write("|-------|---------|-------------|----------------|-------------------|\n")

        ranked = sorted(
            [
                (
                    _avg(d["total_scores"]),
                    name,
                    {c: _avg(d["category_scores"][c]) for c in CATEGORIES},
                )
                for name, d in results.items()
            ],
            reverse=True,
        )
        for avg_total, name, cat in ranked:
            f.write(
                f"| {name} | {avg_total:.1f}"
                f" | {cat['Organ-level']:.1f}"
                f" | {cat['Cellular-level']:.1f}"
                f" | {cat['Subcellular-level']:.1f} |\n"
            )
        f.write("\n")

        # --- Table 2: overall + dimension breakdown ---
        f.write("## Table 2: Overall and Dimension Scores\n\n")
        f.write("| Model | Average | Scientific | Visual | Instruction |\n")
        f.write("|-------|---------|------------|--------|-------------|\n")

        ranked2 = sorted(
            [
                (
                    _avg(d["total_scores"]),
                    name,
                    _avg(d["scientific_scores"]),
                    _avg(d["visual_scores"]),
                    _avg(d["instruction_scores"]),
                )
                for name, d in results.items()
            ],
            reverse=True,
        )
        for avg_total, name, sci, vis, ins in ranked2:
            f.write(f"| {name} | {avg_total:.1f} | {sci:.1f} | {vis:.1f} | {ins:.1f} |\n")
        f.write("\n")

        # --- Detailed per-model stats ---
        f.write("## Detailed Statistics\n\n")
        for name, data in sorted(results.items()):
            f.write(f"### {name}\n\n")
            f.write(f"- **Valid tasks**: {data['valid_tasks']}\n")
            f.write(f"- **Overall average**: {_avg(data['total_scores']):.2f}\n")
            f.write(f"- **Scientific average**: {_avg(data['scientific_scores']):.2f}\n")
            f.write(f"- **Visual average**: {_avg(data['visual_scores']):.2f}\n")
            f.write(f"- **Instruction average**: {_avg(data['instruction_scores']):.2f}\n\n")
            f.write("**Category breakdown:**\n")
            for cat in CATEGORIES:
                sc = data["category_scores"][cat]
                f.write(f"- {cat}: {_avg(sc):.2f} ({len(sc)} tasks)\n")
            f.write("\n")

        # --- Scoring methodology ---
        f.write("## Scoring Methodology\n\n")
        f.write("### Formula\n\n")
        f.write("1. **Dimension raw score**: `S = Σ(s_i × w_i)` where `s_i ∈ {0, 1}` and `w_i` is the criterion weight\n")
        f.write("2. **Normalised score**: `S_norm = max(0, S / Σw_i+) × 100` where `Σw_i+` sums weights of positive criteria only\n")
        f.write("3. **Task score**: normalised combined score across all three dimensions\n\n")
        f.write("### Dimensions\n\n")
        f.write("- **Scientific**: Scientific accuracy of the video content\n")
        f.write("- **Visual**: Visual quality and realism\n")
        f.write("- **Instruction**: Adherence to the task prompt\n\n")
        f.write("### Categories\n\n")
        f.write("- **Organ-level**: Tasks depicting organ-scale structures and processes\n")
        f.write("- **Cellular-level**: Tasks depicting cell-scale structures and processes\n")
        f.write("- **Subcellular-level**: Tasks depicting subcellular structures and processes\n")

    print(f"Report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    eval_dir = "./eval_result"
    rubrics_path = "./MWBenchRubrics_norm.json"
    tasks_path = "./finaltasks.json"

    print("Loading rubrics and task metadata ...")
    rubrics_data = _load_json(rubrics_path)
    tasks_data = _load_json(tasks_path)

    rubrics_index: Dict[int, Any] = {item["index"]: item for item in rubrics_data}
    tasks_index: Dict[int, str] = {item["index"]: item.get("final_label", "Unknown") for item in tasks_data}

    print(f"Rubrics: {len(rubrics_index)} entries  |  Tasks: {len(tasks_index)} entries")

    results: Dict[str, Any] = {}
    for filename in os.listdir(eval_dir):
        if not filename.endswith("_result.json"):
            continue
        model_name = filename.replace("_result.json", "")
        print(f"\nProcessing: {model_name}")
        try:
            model_data = _load_json(os.path.join(eval_dir, filename))
            results[model_name] = _aggregate_model(model_data, rubrics_index, tasks_index)
            print(f"  Valid tasks: {results[model_name]['valid_tasks']}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print(f"\nAggregated {len(results)} model(s).  Generating report ...")
    _generate_report(results, os.path.join(eval_dir, "evaluation_report.md"))
    print("Done.")


if __name__ == "__main__":
    main()
