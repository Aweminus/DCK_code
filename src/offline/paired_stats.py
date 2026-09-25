"""Question-level paired statistics between two arms (Table 1 protocol).

Per question i, d_i = mean over the paired rollout iterations of
(y_A - y_B), y in {0, 1} from evaluate.py's *_scored.jsonl is_correct.
Reports: mean pass@1 difference (pp), 95% percentile bootstrap interval
(question as resampling unit), one-sided sign-flip p-value over the nonzero
d_i, and Holm step-down correction over a declared family of comparisons.

Usage:
  python -m offline.paired_stats --dataset browsecomp \
      --comparison dck_vs_resum:<armA_dir>:<armB_dir> \
      --comparison dck_vs_single:<armC_dir>:<armB_dir> \
      --holm_family dck_vs_resum,dck_vs_single \
      --out paired_statistics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict


def load_per_question(folder: str) -> dict:
    """question -> [is_correct per iteration] from iter*_scored.jsonl."""
    files = sorted(glob.glob(os.path.join(folder, "iter*_scored.jsonl")))
    if not files:
        raise ValueError(f"INPUT_ERROR: no iter*_scored.jsonl under {folder}; "
                         "run evaluate.py first")
    per_q: dict = defaultdict(list)
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                per_q[rec["question"]].append(bool(rec.get("is_correct")))
    # all iterations must have the same length per question; pad is a config
    # error the caller should see, not silently average over
    lengths = {len(v) for v in per_q.values()}
    if len(lengths) != 1:
        raise ValueError(f"INPUT_ERROR: uneven iteration counts {lengths} in {folder}")
    return dict(per_q)


def paired_diffs(folder_a: str, folder_b: str) -> list:
    a, b = load_per_question(folder_a), load_per_question(folder_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("INPUT_ERROR: no shared questions between the two arms")
    dropped = len(set(a) ^ set(b))
    return [(sum(y1 for y1 in a[q]) / len(a[q]) - sum(y2 for y2 in b[q]) / len(b[q]))
            for q in shared], shared, a, b, dropped


def bootstrap_ci(diffs: list, n_boot: int, rng: random.Random) -> list:
    """95% percentile interval of the mean, question as resampling unit."""
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return [round(lo * 100, 2), round(hi * 100, 2)]


def sign_flip_p(diffs: list, n_flips: int, rng: random.Random) -> float:
    """One-sided sign-flip test on the nonzero per-question differences:
    p = P(mean of sign-flipped d >= observed mean), empirical exceedance
    (count+1)/(B+1) so zero exceedances reports the resolution floor."""
    nz = [d for d in diffs if d != 0.0]
    if not nz:
        return 1.0
    observed = sum(nz)
    exceed = 0
    for _ in range(n_flips):
        flipped = sum(d if rng.random() < 0.5 else -d for d in nz)
        if flipped >= observed:
            exceed += 1
    return round((exceed + 1) / (n_flips + 1), 4)


def holm(pvals: dict) -> dict:
    """Holm step-down: adjusted p_(i) = max_{j<=i} (m-j+1) * p_(j), clipped
    at 1 and enforced monotone non-decreasing."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, (m - i) * pvals[key])
        adjusted[key] = round(min(1.0, running), 4)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--comparison", action="append", required=True,
                        help="label:<armA_scored_dir>:<armB_scored_dir> (repeatable)")
    parser.add_argument("--holm_family", default="",
                        help="comma-separated labels forming the Holm family")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    family = {s.strip() for s in args.holm_family.split(",") if s.strip()}
    comparisons, pvals = {}, {}
    for spec in args.comparison:
        label, folder_a, folder_b = spec.split(":", 2)
        rng = random.Random(args.seed)
        diffs, shared, a, b, dropped = paired_diffs(folder_a, folder_b)
        mean_pp = round(100.0 * sum(diffs) / len(diffs), 2)
        n_iters = len(next(iter(a.values())))
        decomp: dict = defaultdict(int)
        for q in shared:
            delta = round(sum(a[q]) - sum(b[q]))
            if delta > 0:
                decomp[f"plus{delta}"] += 1
            elif delta < 0:
                decomp[f"minus{abs(delta)}"] += 1
        rec = {
            "dataset": args.dataset,
            "n_paired_questions": len(shared),
            "n_iterations": n_iters,
            "questions_dropped_by_mismatch": dropped,
            "mean_pass_at_1_diff_pp": mean_pp,
            "ci95_percentile": bootstrap_ci(diffs, args.bootstrap, rng),
            "sign_flip_p_one_sided": sign_flip_p(diffs, args.bootstrap, rng),
            "questions_better": sum(1 for d in diffs if d > 0),
            "questions_worse": sum(1 for d in diffs if d < 0),
            "questions_tied": sum(1 for d in diffs if d == 0),
            "rollout_diff_decomposition": dict(sorted(decomp.items())),
        }
        if label in family:
            pvals[label] = rec["sign_flip_p_one_sided"]
        else:
            rec["holm_adjusted_p"] = None
            rec["note"] = ("Reference/boundary analysis; not part of the "
                           "Holm family.")
        comparisons[label] = rec

    adjusted = holm(pvals) if pvals else {}
    for label, p in adjusted.items():
        comparisons[label]["holm_adjusted_p"] = p

    result = {
        "protocol": (
            f"Question-level cluster bootstrap ({args.bootstrap} resamples, "
            "question as resampling unit), 95% percentile interval, one-sided "
            "sign-flip test at question level over the nonzero per-question "
            "mean differences; Holm step-down correction over the declared "
            f"family ({', '.join(sorted(family)) if family else 'none'}). "
            "d_i = mean over the paired rollout iterations of (y_A - y_B). "
            "Sign-flip p-values are empirical exceedance rates "
            "((count+1)/(B+1)); zero exceedances reports the Monte Carlo "
            f"resolution floor 1/({args.bootstrap}+1)."),
        "holm_family": sorted(family),
        "seed": args.seed,
        "comparisons": comparisons,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {"mean_diff_pp": v["mean_pass_at_1_diff_pp"],
                          "holm_p": v.get("holm_adjusted_p")}
                      for k, v in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
