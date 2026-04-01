#!/usr/bin/env python3
"""
video_evaluator.py
------------------
Evaluates generated videos against MWBench rubrics using a vision-language
model.  For each video it extracts N evenly-spaced frames,
then calls the API once per rubric dimension (Scientific / Visual / Instruction)
and saves the raw + parsed scores to eval_result/<model>_result.json.

Environment variables
---------------------
EVAL_API_KEY   : API key for the evaluation model endpoint  (required)
EVAL_BASE_URL  : Base URL of the OpenAI-compatible API      (optional,
                  defaults to https://api.openai.com)
"""

import json
import os
import time
import requests
import cv2
import base64
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from pathlib import Path
import threading
from tqdm import tqdm


@dataclass
class EvalConfig:
    """Centralised evaluation configuration."""

    # --- API ---
    api_key: str = field(default_factory=lambda: os.environ.get("EVAL_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("EVAL_BASE_URL", "https://api.openai.com"))
    model: str = "gpt-4o"

    # --- Concurrency / reliability ---
    max_workers: int = 100      # Thread-pool size
    save_interval: int = 100    # Checkpoint every N completed items
    retry_times: int = 3        # Per-request retry attempts
    request_delay: float = 0.5  # Seconds between consecutive requests
    request_timeout: int = 180  # Per-request timeout in seconds

    # --- Quick-test mode ---
    test_mode: bool = False
    test_limit: int = 4         # Max videos when test_mode=True

    # --- Paths ---
    rubrics_file: str = "./MWBenchRubrics.json"
    base_input_dir: str = "."          # Parent directory that contains per-model folders
    output_dir: str = "./eval_result"  # Directory for output JSON files

    # --- Models to evaluate (populated in __post_init__) ---
    model_names: List[str] = None

    # --- Frame sampling ---
    num_frames: int = 8  # Frames extracted per video

    def __post_init__(self) -> None:
        if self.model_names is None:
            self.model_names = [
                "Runway_Gen4",
            ]


class VideoEvaluator:
    """Handles frame extraction and per-dimension API evaluation."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.lock = threading.Lock()
        self.rubrics_data = self._load_rubrics()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_rubrics(self) -> Dict[int, Any]:
        """Load and index rubrics by task index."""
        try:
            with open(self.config.rubrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {item["index"]: item for item in data}
        except Exception as e:
            print(f"Error: cannot load rubrics file {self.config.rubrics_file}: {e}")
            return {}

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------

    def _compress_images(self, base64_images: List[str], quality: int = 85) -> List[str]:
        """Re-encode base64 JPEG images at a lower quality to reduce payload size."""
        compressed = []
        try:
            for b64 in base64_images:
                img_data = base64.b64decode(b64)
                arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    compressed.append(base64.b64encode(buf).decode("utf-8"))
                else:
                    compressed.append(b64)  # Keep original if decode fails
        except Exception as e:
            print(f"Error compressing images: {e}")
            return base64_images
        return compressed

    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[str]:
        """
        Extract *num_frames* evenly-spaced frames from *video_path*.

        Returns a list of base64-encoded JPEG strings (empty list on failure).
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video file is empty or corrupted: {video_path}")

            # Sample positions spread evenly across the video duration
            if num_frames == 1:
                positions = [total_frames // 2]
            else:
                step = total_frames / num_frames
                positions = [int(i * step + step / 2) for i in range(num_frames)]

            frames_b64 = []
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    frames_b64.append(base64.b64encode(buf).decode("utf-8"))

            cap.release()
            return frames_b64

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return []

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, task_info: Dict[str, Any], dimension: str, rubrics: Dict[str, Any]) -> str:
        """Compose the evaluation prompt for a single dimension."""
        task_desc = task_info.get("任务", "")
        detailed_req = task_info.get("具体要求", "")

        dimension_names = {
            "rubrics_scientific": "Scientific Accuracy",
            "rubrics_visual": "Visual Quality",
            "rubrics_instruction": "Instruction Following",
        }
        dim_name = dimension_names.get(dimension, dimension)

        # Build the rubric checklist
        rubrics_text = ""
        for i in range(1, 10):
            d_key, s_key = f"d{i}", f"s{i}"
            if d_key in rubrics and s_key in rubrics:
                point_type = "Positive Point" if rubrics[s_key] == "+1" else "Penalty Point"
                rubrics_text += f"{i}. [{point_type}] {rubrics[d_key]}\n"

        return f"""You are a professional video evaluation expert. Please evaluate this microscopic world scientific animation video in the {dim_name} dimension according to the following criteria.

**Task Description:**
{task_desc}

**Detailed Requirements:**
{detailed_req}

**{dim_name} Evaluation Criteria:**
{rubrics_text}

**Evaluation Instructions:**
1. Carefully observe the 8 video screenshots arranged in chronological order.
2. Judge each evaluation point: mark 1 if the criterion is met, 0 otherwise.
3. Positive Points: score 1 if the video meets the standard.
4. Penalty Points: score 1 if the video exhibits the described problem.
5. Provide detailed reasoning for every judgment.

**Output Format (strict JSON):**
{{
    "scores": [score1, score2, ...],
    "reasoning": "Detailed reasoning explaining the judgment for each criterion."
}}

The length of "scores" must equal the number of evaluation points; each element is 0 or 1."""

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def call_api_for_evaluation(
        self,
        base64_images: List[str],
        task_info: Dict[str, Any],
        dimension: str,
        rubrics: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Send frames + prompt to the evaluation API and return parsed result.

        Automatically retries with progressively lower JPEG quality if the
        request is rejected with HTTP 413 (payload too large).
        """
        if not base64_images:
            return None

        prompt = self._build_prompt(task_info, dimension, rubrics)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        # Try sending with decreasing image quality until the request succeeds
        compression_levels = [100, 30, 20, 10]

        for quality in compression_levels:
            current_images = (
                base64_images if quality == 100
                else self._compress_images(base64_images, quality)
            )
            if quality < 100:
                print(f"Retrying with compression quality={quality} ...")

            content = [{"type": "text", "text": prompt}]
            for b64 in current_images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            payload = {"model": self.config.model, "messages": [{"role": "user", "content": content}]}

            for attempt in range(self.config.retry_times):
                try:
                    time.sleep(self.config.request_delay)
                    response = requests.post(
                        f"{self.config.base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.config.request_timeout,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("choices"):
                            raw = result["choices"][0]["message"]["content"].strip()
                            parsed = self._parse_json_response(raw)
                            return {"raw_response": raw, "parsed_result": parsed}
                        print(f"Unexpected API response format: {result}")

                    elif response.status_code == 413:
                        print(f"HTTP 413: payload too large at quality={quality}, will compress further.")
                        break  # Move to next (lower) quality level

                    elif response.status_code == 429:
                        wait = 2 ** attempt
                        print(f"HTTP 429: rate limited, waiting {wait}s ...")
                        time.sleep(wait)

                    else:
                        print(f"API error {response.status_code}: {response.text}")

                except requests.exceptions.Timeout:
                    print(f"Timeout on attempt {attempt + 1}/{self.config.retry_times}")
                    if attempt < self.config.retry_times - 1:
                        time.sleep(2 ** attempt)

                except requests.exceptions.RequestException as e:
                    print(f"Request error on attempt {attempt + 1}: {e}")
                    if attempt < self.config.retry_times - 1:
                        time.sleep(2 ** attempt)

                except Exception as e:
                    print(f"Unexpected error: {e}")
                    break

        return None

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
        """Extract and parse the first JSON object found in *raw*."""
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None


class EvaluationGenerator:
    """Orchestrates multi-threaded evaluation across one or more models."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.video_evaluator = VideoEvaluator(config)
        self.processed_videos: set = set()
        self.results: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _load_existing_results(self, model_name: str) -> List[Dict[str, Any]]:
        """Load previously saved results and mark completed videos."""
        output_file = os.path.join(self.config.output_dir, f"{model_name}_result.json")
        if not os.path.exists(output_file):
            return []
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                idx = item.get("index")
                if idx and item.get("scientific_eval") and item.get("visual_eval") and item.get("instruction_eval"):
                    self.processed_videos.add(idx)
            print(f"Loaded {len(data)} existing results ({len(self.processed_videos)} complete) for {model_name}")
            return data
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return []

    def _get_video_files(self, model_name: str) -> List[Tuple[int, str]]:
        """Return a sorted list of (index, path) for unprocessed videos of *model_name*."""
        video_ext = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
        files: List[Tuple[int, str]] = []

        input_path = Path(self.config.base_input_dir) / model_name
        if input_path.is_dir():
            for p in input_path.iterdir():
                if p.is_file() and p.suffix.lower() in video_ext:
                    try:
                        files.append((int(p.stem), str(p)))
                    except ValueError:
                        print(f"Warning: cannot parse index from filename '{p.name}'")

        files.sort(key=lambda x: x[0])
        pending = [(i, p) for i, p in files if i not in self.processed_videos]

        if self.config.test_mode:
            pending = pending[: self.config.test_limit]

        print(f"Found {len(files)} videos for {model_name}; {len(self.processed_videos)} already done; {len(pending)} to process")
        return pending

    def _save_results(self, model_name: str, force: bool = False) -> None:
        """Merge in-memory results with the existing output file and write to disk."""
        if not force and len(self.results) % self.config.save_interval != 0:
            return
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            output_file = os.path.join(self.config.output_dir, f"{model_name}_result.json")

            existing: List[Dict[str, Any]] = []
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)

            result_map = {item.get("index"): item for item in existing}
            with self.lock:
                for r in self.results:
                    result_map[r["index"]] = r

            merged = sorted(result_map.values(), key=lambda x: x.get("index", 0))
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(merged)} results to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Core evaluation logic
    # ------------------------------------------------------------------

    def _evaluate_single_video(self, index: int, video_path: str) -> Optional[Dict[str, Any]]:
        """Evaluate one video across all three dimensions; returns None on failure."""
        if index not in self.video_evaluator.rubrics_data:
            print(f"Warning: no rubrics found for index {index}")
            return None

        task_info = self.video_evaluator.rubrics_data[index]
        frames = self.video_evaluator.extract_frames(video_path, self.config.num_frames)
        if not frames:
            print(f"Cannot extract frames: {video_path}")
            return None

        result: Dict[str, Any] = {"index": index, "video_path": video_path}
        dimensions = [
            ("rubrics_scientific", "scientific_eval"),
            ("rubrics_visual", "visual_eval"),
            ("rubrics_instruction", "instruction_eval"),
        ]

        for dim, key in dimensions:
            if dim not in task_info:
                print(f"Warning: dimension '{dim}' missing in rubrics for index {index}")
                return None

            print(f"Evaluating [{index}] {dim} ...")
            eval_result = self.video_evaluator.call_api_for_evaluation(frames, task_info, dim, task_info[dim])
            if eval_result is None:
                print(f"  ✗ Failed [{index}] {dim}")
                return None

            result[key] = eval_result
            print(f"  ✓ Done    [{index}] {dim}")
            time.sleep(1)  # Brief pause to reduce rate-limit pressure

        print(f"✓ Completed [{index}]: {Path(video_path).name}")
        return result

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run_single_model(self, model_name: str) -> None:
        """Run the full evaluation pipeline for one model."""
        print(f"\n=== Processing model: {model_name} ===")
        self.processed_videos = set()
        self.results = []

        self._load_existing_results(model_name)
        pending = self._get_video_files(model_name)
        if not pending:
            print(f"No videos to process for {model_name}")
            return

        pbar = tqdm(total=len(pending), desc=f"Evaluating {model_name}")
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(self._evaluate_single_video, idx, path): (idx, path)
                for idx, path in pending
            }
            for future in as_completed(future_map):
                try:
                    result = future.result()
                    if result:
                        with self.lock:
                            self.results.append(result)
                        if len(self.results) % self.config.save_interval == 0:
                            self._save_results(model_name)
                except Exception as e:
                    print(f"Worker error: {e}")
                finally:
                    pbar.update(1)
        pbar.close()

        if self.results:
            self._save_results(model_name, force=True)

        print(f"=== {model_name} done: {len(pending)} attempted, {len(self.results)} succeeded ===")
        print(f"Results: {os.path.join(self.config.output_dir, f'{model_name}_result.json')}")

    def run(self) -> None:
        """Evaluate all configured models sequentially."""
        print("=== MWBench Video Evaluation System ===")
        print(f"Model : {self.config.model}")
        print(f"Threads: {self.config.max_workers}")
        print(f"Input  : {self.config.base_input_dir}")
        print(f"Output : {self.config.output_dir}")
        print(f"Test   : {self.config.test_mode}")
        print(f"Models : {self.config.model_names}")
        print("-" * 50)

        for model_name in self.config.model_names:
            try:
                self.run_single_model(model_name)
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                traceback.print_exc()

        print(f"\n=== All done. {len(self.config.model_names)} model(s) processed. ===")
        print(f"Results saved in: {self.config.output_dir}")


def main() -> None:
    """Entry point.  Adjust EvalConfig fields below as needed."""
    config = EvalConfig(
        # Uncomment to enable quick-test mode:
        # test_mode=True,
        # test_limit=4,
        # max_workers=10,
    )
    EvaluationGenerator(config).run()


if __name__ == "__main__":
    main()
