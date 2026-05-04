#!/usr/bin/env python3
"""Evaluate conversations in a folder with prompts from prompts_evaluation.

For each conversation (.txt) in the input folder, run each prompt (except
panas_before/after) and parse the numeric score output. Produce per-conversation
scores and the average score per metric across all conversations.
"""
import argparse
import json
import logging
import re
from pathlib import Path
from statistics import mean

from src.llm import LLM

from src.notify import notify_start_process, notify_end_process

ROOT = Path(__file__).resolve().parents[0]
PROMPTS_DIR = ROOT / "prompts_evaluation"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_folder")


def load_prompts(prompts_dir: Path):
    prompts = {}
    for p in sorted(prompts_dir.glob("*.txt")):
        # skip panas prompts
        if p.name in ("panas_after.txt", "panas_before.txt"):
            continue
        prompts[p.name] = p.read_text(encoding="utf-8")
    return prompts


_score_re = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*(.*)", re.S)


def parse_score(text: str):
    """Parse a generated evaluator response of form "<score>, <explanation>".

    Returns (score: float, explanation: str) or (None, full_text) if parsing fails.
    """
    if not text:
        return None, ""
    m = _score_re.match(text.strip())
    if m:
        try:
            return float(m.group(1)), m.group(2).strip()
        except Exception:
            return None, text.strip()
    # try to find first number anywhere
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if m2:
        try:
            return float(m2.group(1)), text.strip()
        except Exception:
            return None, text.strip()
    return None, text.strip()


def evaluate_conversation(llm: LLM, convo_text: str, prompts: dict, max_new_tokens=256):
    scores = {}
    for fname, template in prompts.items():
        prompt_filled = template.format(conversation=convo_text)
        try:
            print(prompt_filled)
            _, generated = llm.run_prompt(prompt_filled, do_sample=False, keep_history=False, max_new_tokens=max_new_tokens)    # greedy decoding for evaluation
            logger.info(f"Generated response: {generated}")
        except Exception as e:
            logger.exception(f"LLM run failed for prompt {fname}: {e}")
            generated = ""
        score, explanation = parse_score(generated)
        metric_name = Path(fname).stem
        scores[metric_name] = {"score": score, "explanation": explanation, "raw": generated}
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=str(ROOT / "data_out" / "expand_2"), help="Folder with conversation .txt files")
    parser.add_argument("--prompts", default=str(PROMPTS_DIR), help="Prompts directory")
    parser.add_argument("--out", default=str(ROOT / "data_out" / "folder_eval_results.jsonl"))
    parser.add_argument("--model", default="qwen3", help="Model name for src.llm.LLM")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    logger.info(f"Loaded {len(prompts)} prompts (excluding panas) from {args.prompts}")

    folder = Path(args.folder)
    if not folder.exists():
        logger.error("Conversation folder does not exist: %s", folder)
        return

    convo_files = sorted(folder.glob("*.txt"))
    if not convo_files:
        logger.error("No .txt conversation files found in %s", folder)
        return
    logger.info("Found %d conversations in %s", len(convo_files), folder)

    llm = LLM(model_name=args.model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_metric_values = {}  # metric -> list of scores (one per conversation if parsed)

    with out_path.open("w", encoding="utf-8") as fout:
        for convo_file in convo_files:
            logger.info(f"Evaluating conversation: {convo_file.name}")
            convo_text = convo_file.read_text(encoding="utf-8")
            convo_scores = evaluate_conversation(llm, convo_text, prompts)

            # accumulate
            for metric, info in convo_scores.items():
                score = info.get("score")
                per_metric_values.setdefault(metric, []).append(score if score is not None else None)

            record = {"conversation_file": convo_file.name, "scores": convo_scores}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # compute averages (ignore None)
    averages = {}
    for metric, vals in per_metric_values.items():
        numeric = [v for v in vals if v is not None]
        averages[metric] = mean(numeric) if numeric else None

    summary_path = out_path.parent / (out_path.stem + "_summary.json")
    summary = {"num_conversations": len(convo_files), "average_scores": averages}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Finished. Per-conversation results: %s", out_path)
    logger.info("Summary saved to %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # python3 evaluate_folder.py --folder data_out/doctor_instructions_2 --out data_eval/doctor_instructions_2_eval.jsonl
    notify_start_process("evaluate_folder")
    main()
    notify_end_process("evaluate_folder")
