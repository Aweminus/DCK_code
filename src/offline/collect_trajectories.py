"""Collect DCK-Train trajectories on the main training host (Spec 6.4).

One resum-host rollout per question with DCK off, using the exact main
evaluation stack (agent, search/visit, page extractor, trigger, summary
tool). A snapshot is saved before every real compaction event; trajectories
without any event contribute their terminal horizon. Infrastructure-failed
questions are retried once with the same seed and otherwise replaced by the
next question of the same language from the candidate pool.

Usage:
  python -m offline.collect_trajectories --model <path> \
      --questions data/dck_train/questions.jsonl \
      --snapshot_dir data/dck_train/snapshots_<agent>
"""
from __future__ import annotations

import argparse
import json
import os

from prompt import SYSTEM_PROMPT
from dck_agent import DCKReactAgent
from dck.entities import StopTables


def build_agent(model: str, snapshot_dir: str, rollout_seed: int,
                stop_tables_path: str) -> DCKReactAgent:
    llm_cfg = {
        'model': model,
        'generate_cfg': {'max_input_tokens': 320000, 'max_retries': 10,
                         'temperature': 0.85, 'top_p': 0.95},
        'model_type': 'qwen_dashscope',
    }
    stop_tables = StopTables.load(stop_tables_path) if stop_tables_path else None
    return DCKReactAgent(
        llm=llm_cfg,
        function_list=["search", "visit"],
        system_message=SYSTEM_PROMPT,
        paradigm='resum',
        dck_mode='off',
        rollout_seed=rollout_seed,
        stop_tables=stop_tables,
        snapshot_dir=snapshot_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--snapshot_dir", required=True)
    parser.add_argument("--rollout_seed", type=int, default=42)
    parser.add_argument("--stop_tables", default="")
    parser.add_argument("--summary_iteration", type=int, default=1000)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    os.makedirs(args.snapshot_dir, exist_ok=True)
    agent = build_agent(args.model, args.snapshot_dir, args.rollout_seed,
                        args.stop_tables)

    results_path = os.path.join(args.snapshot_dir, "rollouts.jsonl")
    with open(results_path, "a", encoding="utf-8") as out:
        for q in questions:
            task = {"item": {"question": q["question"], "answer": q["answer"],
                             "question_id": q["question_id"]},
                    "rollout_id": 1}
            result = agent._run(task, args.model, args.summary_iteration)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[collect] {q['question_id']}: "
                  f"termination={result['termination']}")

    print(f"rollout results appended to {results_path}")


if __name__ == "__main__":
    main()
