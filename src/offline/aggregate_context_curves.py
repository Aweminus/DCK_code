"""Survivor-conditioned working-context curves from the context log.

Aggregates the per-tool-turn JSONL written by --dck_context_log into the
context_curves.json curves: value at turn t = mean ACTIVE working-context
token_count over the trajectories that still have a running record at turn t
(survivor conditioning). The terminal running=False record marks trajectory
end and carries the termination reason.

Usage:
  python -m offline.aggregate_context_curves \
      --arm react:<react_context.jsonl> --arm resum:<resum_context.jsonl> \
      --arm resum_dck:<dck_context.jsonl> --out context_curves.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def read_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def survivor_curve(records: list) -> dict:
    """One arm's curve: per-turn mean over trajectories alive at that turn."""
    running = [r for r in records if r.get("running")]
    terminated = [r for r in records if not r.get("running")]
    # a trajectory is (question_id, rollout_seed); turns are its tool turns
    by_turn: dict = defaultdict(list)
    for r in running:
        by_turn[r["turn"]].append(r["token_count"])
    turns = sorted(by_turn)
    curve = [round(sum(by_turn[t]) / len(by_turn[t]), 1) for t in turns]
    compaction_turns = [r["turn"] for r in running if r.get("compaction_index", 0) >= 1]
    return {
        "n_trajectories": len({(r["question_id"], r["rollout_seed"])
                               for r in running}),
        "n_terminal_records": len(terminated),
        "turns": turns,
        "mean_token_count": curve,
        "survivors_per_turn": [len(by_turn[t]) for t in turns],
        "first_compaction_turn": min(compaction_turns) if compaction_turns else None,
        "terminations": dict(sorted(
            (term, sum(1 for r in terminated if r.get("termination") == term))
            for term in {r.get("termination") for r in terminated})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True,
                        help="name:<context_log.jsonl> (repeatable)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split(":", 1)
        arms[name] = survivor_curve(read_jsonl(path))

    max_turn = max((c["turns"][-1] for c in arms.values()
                    if c["turns"]), default=0)
    result = {
        "definition": (
            "Value at tool turn t = average ACTIVE working-context tokens "
            "after turn t, averaged over the trajectories still running at "
            "turn t (survivor conditioning); source: per-tool-turn context "
            "log written by main.py --dck_context_log."),
        "tool_turns": list(range(1, max_turn + 1)),
        "arms": arms,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {"n_traj": v["n_trajectories"],
                          "first_compaction_turn": v["first_compaction_turn"],
                          "terminations": v["terminations"]}
                      for k, v in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
