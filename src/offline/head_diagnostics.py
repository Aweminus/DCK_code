"""Head diagnostics: oracle top-8 agreement and scoring latency.

Oracle agreement (validation split): per labeled compaction event, the oracle
top-8 is the top-k set by the materialized closure_delta label; each selector
(dck = closure head, single = single-knockout head, random = the seeded
RandomSelector draw) picks its top-k, and the reported value is the mean
fraction of the oracle set recovered, |selected n oracle| / k. Latency: mean
wall-clock of the per-event head scoring pass (scaler transform + MLP
forward over the event's observations); the frozen agent model's hidden
pooling is excluded — it is not head cost.

Usage:
  python -m offline.head_diagnostics --labels data/dck_train/labels_7b.jsonl \
      --closure_head checkpoints/closure_head_7b \
      [--single_head checkpoints/single_head_7b] --out head_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from dck.config import DEFAULT_CONFIG
from dck.features import FeatureScaler
from dck.head import load_head


def iter_events(labels_path: str):
    """Yield (record, bank_indices) with the hidden-bank alignment of
    train_head.load_examples: one bank row per label entry in file order."""
    bank_index = 0
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            indices = {}
            for lb in rec["labels"]:
                indices[lb["obs_id"]] = bank_index
                bank_index += 1
            yield rec, indices


def top_k_by(pairs, k: int) -> list:
    """Top-k obs ids by (value desc, obs_id asc) — the serving sort order."""
    ranked = sorted(pairs, key=lambda p: (-p[1], p[0]))
    return [obs for obs, _ in ranked[:k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True,
                        help="labels JSONL from materialize_labels.py")
    parser.add_argument("--hidden_bank", default="",
                        help="pooled hidden-state bank .pt (default: alongside --labels)")
    parser.add_argument("--closure_head", required=True)
    parser.add_argument("--single_head", default="",
                        help="single-knockout head dir (enables the single arm)")
    parser.add_argument("--split", default="val", help="split to evaluate on")
    parser.add_argument("--k", type=int, default=0,
                        help="protection set size (default: config.k_protect)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    k = args.k or DEFAULT_CONFIG.k_protect
    bank_path = args.hidden_bank or os.path.join(
        os.path.dirname(args.labels) or ".", "hidden_states.pt")
    bank = torch.load(bank_path).float()

    models = {"dck_closure": load_head(args.closure_head)[0]}
    scalers = {"dck_closure":
               FeatureScaler.load(os.path.join(args.closure_head, "scaler.json"))}
    if args.single_head:
        models["single_knockout"] = load_head(args.single_head)[0]
        single_scaler = os.path.join(args.single_head, "scaler.json")
        scalers["single_knockout"] = (
            FeatureScaler.load(single_scaler)
            if os.path.exists(single_scaler) else scalers["dck_closure"])

    overlap = {name: 0.0 for name in list(models) + ["random"]}
    n_events = 0
    latency_ms = {name: [] for name in models}
    with torch.no_grad():
        for rec, indices in iter_events(args.labels):
            if rec.get("split", "train") != args.split:
                continue
            usable = {lb["obs_id"]: lb.get("closure_delta")
                      for lb in rec["labels"] if lb.get("closure_delta") is not None}
            if not usable:
                continue
            candidates = sorted(usable)
            k_event = min(k, len(candidates))
            oracle = top_k_by([(o, usable[o]) for o in candidates], k_event)
            n_events += 1

            for name, head in models.items():
                start = time.perf_counter()
                scores = {}
                scaler = scalers[name]
                for obs in candidates:
                    scalars = torch.tensor(
                        [scaler.transform(rec["features"][obs])],
                        dtype=torch.float32)
                    hidden = bank[indices[obs]].unsqueeze(0)
                    scores[obs] = head(hidden, scalars).item()
                latency_ms[name].append(
                    (time.perf_counter() - start) * 1000.0)
                selected = top_k_by(list(scores.items()), k_event)
                overlap[name] += (len(set(selected) & set(oracle)) / k_event)

            rng = random.Random(
                f"{rec['question_id']}|{rec['rollout_seed']}|"
                f"{rec['compaction_index']}")
            drawn = rng.sample(candidates, k_event)
            overlap["random"] += len(set(drawn) & set(oracle)) / k_event

    if n_events == 0:
        raise ValueError(f"INPUT_ERROR: no usable {args.split} events in "
                         f"{args.labels}")

    agreement = {name: round(v / n_events, 4) for name, v in overlap.items()}
    # hypergeometric sanity floor for the random selector
    mean_candidates = 0
    n_seen, total = 0, 0
    for rec, indices in iter_events(args.labels):
        if rec.get("split", "train") != args.split:
            continue
        if any(lb.get("closure_delta") is not None for lb in rec["labels"]):
            total += sum(1 for lb in rec["labels"]
                         if lb.get("closure_delta") is not None)
            n_seen += 1
    mean_candidates = total / n_seen
    hyper = round((min(k, mean_candidates) / mean_candidates
                   if mean_candidates else 0.0), 4)

    result = {
        "split": args.split,
        "n_events": n_events,
        "k": k,
        "mean_candidates_per_event": round(mean_candidates, 2),
        "oracle_top8_agreement_val": {
            **agreement,
            "hypergeometric_expected_random": hyper,
            "definition": (
                "mean fraction of the oracle top-k closure-influence "
                "observations recovered by each selector, "
                "|selected n oracle| / k, on the validation split; a uniform "
                "random selector drawing k of the mean candidate count has "
                "hypergeometric expected overlap k*k/n"),
        },
        "head_latency_ms_per_compaction_event": {
            name: round(sum(v) / len(v), 2) for name, v in latency_ms.items()
        },
        "latency_note": (
            "scaler transform + head forward over the event's observations, "
            "torch.no_grad, wall-clock mean per compaction event; excludes "
            "the frozen agent model's hidden-state pooling"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"oracle_top8_agreement_val":
                      result["oracle_top8_agreement_val"],
                      "head_latency_ms_per_compaction_event":
                      result["head_latency_ms_per_compaction_event"]},
                     indent=2))


if __name__ == "__main__":
    main()
