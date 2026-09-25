"""Conditional efficiency from evaluate.py scored files (Table 1 protocol).

evaluate.py writes <iter>.jsonl and <iter>_scored.jsonl (per-record
is_correct) but only reports unconditional averages. This script filters the
scored records by is_correct==True and recomputes avg_traj_length /
avg_tool_total over the correct subset only, following the ReSum paper
(Appendix F.2) conditional-efficiency protocol. Tokenization and tool
counting follow evaluate.single_round_statistics verbatim so conditional and
unconditional values are directly comparable.

Usage:
  python -m offline.conditional_efficiency \
      --input_folder <output>/<model_dir>/<dataset_parent> --out efficiency.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_tokenizer(path: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path)
    except Exception:
        import tiktoken
        return tiktoken.encoding_for_model("gpt-4o")


def traj_stats(sample, tokenizer, available_tools) -> dict:
    """Per-record (traj_length, tool_total, answer_length) with evaluate.py's
    conventions: full concatenation of message contents for tokens, parsed
    <tool_call> blocks in assistant messages for tool counts."""
    msgs = sample.get("messages", [])
    traj_length = len(tokenizer.encode("".join(m["content"] for m in msgs)))
    tool_total, invalid = 0, 0
    for msg in msgs:
        if msg.get("role") != "assistant" or "<tool_call>" not in msg["content"]:
            continue
        try:
            call = json.loads(msg["content"].split("<tool_call>")[1]
                              .split("</tool_call>")[0].strip())
            # evaluate.py counts valid AND invalid-name calls in "total"
            tool_total += 1
            if call["name"] not in available_tools:
                invalid += 1
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    final_msg = msgs[-1]["content"] if msgs else ""
    if "<answer>" in final_msg and "</answer>" in final_msg:
        answer = final_msg.split("<answer>")[1].split("</answer>")[0].strip()
        answer_length = len(tokenizer.encode(answer))
    else:
        answer_length = 0
    return {"traj_length": traj_length, "tool_total": tool_total,
            "invalid": invalid, "answer_length": answer_length}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True,
                        help="folder containing iter*.jsonl / iter*_scored.jsonl")
    parser.add_argument("--tokenizer", default="/path/to/your/Qwen2.5-72B-Instruct",
                        help="stats tokenizer (evaluate.py convention)")
    parser.add_argument("--available_tools", default="search,visit")
    parser.add_argument("--out", required=True, help="output json path")
    args = parser.parse_args()

    available_tools = set(args.available_tools.split(","))
    tokenizer = load_tokenizer(args.tokenizer)

    scored_files = sorted(glob.glob(os.path.join(args.input_folder,
                                                 "iter*_scored.jsonl")))
    if not scored_files:
        raise ValueError(f"INPUT_ERROR: no iter*_scored.jsonl under "
                         f"{args.input_folder}; run evaluate.py first")

    per_iter, all_stats, correct_stats = [], [], []
    for path in scored_files:
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        scored = [(traj_stats(r, tokenizer, available_tools),
                   bool(r.get("is_correct"))) for r in records]
        correct = [s for s, ok in scored if ok]
        per_iter.append({
            "file": os.path.basename(path),
            "n_correct": len(correct),
            "n_total": len(records),
            "tokens_per_correct_trajectory":
                round(sum(s["traj_length"] for s in correct) / len(correct), 2)
                if correct else None,
            "tool_calls_per_correct_trajectory":
                round(sum(s["tool_total"] for s in correct) / len(correct), 2)
                if correct else None,
        })
        all_stats.extend(s for s, _ in scored)
        correct_stats.extend(correct)

    def _avg(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 2) if rows else None

    result = {
        "input_folder": args.input_folder,
        "n_scored_files": len(scored_files),
        "n_correct_rollouts": len(correct_stats),
        "conditional_on_correct": {
            "tokens_per_correct_trajectory": _avg(correct_stats, "traj_length"),
            "tool_calls_per_correct_trajectory": _avg(correct_stats, "tool_total"),
            "avg_answer_length_tokens": _avg(correct_stats, "answer_length"),
        },
        "unconditional_crosscheck": {
            "tokens_all_trajectories_avg": _avg(all_stats, "traj_length"),
            "tool_calls_all_trajectories_avg": _avg(all_stats, "tool_total"),
            "avg_tool_invalid_per_trajectory":
                round(sum(r["invalid"] for r in all_stats) / len(all_stats), 2),
        },
        "per_iteration": per_iter,
        "conditioning": ("conditional on judge-correct final answer "
                         "(is_correct==True in evaluate.py *_scored.jsonl)"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in
                      ("n_correct_rollouts", "conditional_on_correct")},
                     indent=2))


if __name__ == "__main__":
    main()
