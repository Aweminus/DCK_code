"""Compaction-difficulty slices from event logs + scored files.

Questions are stratified by the TOTAL number of compaction events across
the paired HOST BASELINE (dck_mode=off) rollouts: 0, 1, 2+. All arms are
then evaluated on the same fixed slices, so per-slice pass@1 isolates the
protection effect from question difficulty. Baseline compaction counts come
from the event log (baseline arms write one event record per compaction,
aggregated across rollouts per question); per-arm correctness comes from
evaluate.py's *_scored.jsonl.

Usage:
  python -m offline.aggregate_compaction_slices \
      --baseline_event_log <baseline_events.jsonl> \
      --arm resum:<resum_scored_dir> --arm resum_dck:<dck_scored_dir> \
      --out compaction_slices.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

from dck.event_log import read_events


def baseline_compaction_counts(event_log_path: str) -> dict:
    """question_id -> number of compaction events in the baseline rollout."""
    counts: dict = defaultdict(int)
    for rec in read_events(event_log_path):
        if rec.get("type") == "event":
            counts[str(rec["question_id"])] += 1
    if not counts:
        raise ValueError(f"INPUT_ERROR: no event records in {event_log_path}; "
                         "the baseline (off) arm must run with --dck_event_log")
    return dict(counts)


def slice_of(n_compactions: int) -> str:
    return "0" if n_compactions == 0 else "1" if n_compactions == 1 else "2+"


def load_arm(folder: str) -> dict:
    """question -> [is_correct per iteration]."""
    files = sorted(glob.glob(os.path.join(folder, "iter*_scored.jsonl")))
    if not files:
        raise ValueError(f"INPUT_ERROR: no iter*_scored.jsonl under {folder}")
    per_q: dict = defaultdict(list)
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    per_q[rec["question"]].append(bool(rec.get("is_correct")))
    return dict(per_q)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_event_log", required=True,
                        help="event log of the host baseline (dck_mode=off) run")
    parser.add_argument("--arm", action="append", required=True,
                        help="name:<scored_dir> (repeatable)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    counts = baseline_compaction_counts(args.baseline_event_log)
    slices = {q: slice_of(n) for q, n in counts.items()}
    slice_sizes = defaultdict(int)
    for s in slices.values():
        slice_sizes[s] += 1

    result: dict = {}
    for spec in args.arm:
        name, folder = spec.split(":", 1)
        per_q = load_arm(folder)
        # questions absent from the baseline log (e.g. 0-compaction baselines
        # that logged no events only if the log covers them) fall in "0"
        for q in per_q:
            if q not in slices:
                slices[q] = "0"
                slice_sizes["0"] += 1
        correct = defaultdict(int)
        for q, flags in per_q.items():
            correct[slices[q]] += sum(flags)
        n_iters = len(next(iter(per_q.values())))
        arm_out = {}
        for s in ("0", "1", "2+"):
            size = slice_sizes[s]
            arm_out[s] = {
                "correct_rollouts": correct[s],
                "pass_at_1": round(100.0 * correct[s] / (size * n_iters), 2)
                if size else None,
            }
        arm_out["total_correct_rollouts"] = sum(correct.values())
        result[name] = arm_out

    # slice_sizes is finalized only here: the arm loop above folds in the
    # 0-compaction questions that never appear in the baseline event log
    result["slice_sizes"] = dict(sorted(slice_sizes.items()))
    result["definition"] = (
        "Questions stratified by the total number of compaction events across "
        "the paired host baseline (dck_mode=off) rollouts (0, 1, 2+). All "
        "arms are evaluated on the same fixed question slices; DCK/Random/"
        "Single are inert on questions that never compacted in any baseline "
        "rollout, hence identical counts on the 0 slice.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {s: v[s]["pass_at_1"] for s in ("0", "1", "2+")}
                      for k, v in result.items()
                      if isinstance(v, dict) and isinstance(v.get("0"), dict)},
                     indent=2))


if __name__ == "__main__":
    main()
