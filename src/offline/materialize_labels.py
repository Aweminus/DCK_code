"""Materialize closure/single knockout labels from collected snapshots
(Spec 6.5).

For every snapshot: rebuild registry + lineage, compute the factual score
once and per sampled root one closure (+ optionally one single) teacher-
forcing score, and store features alongside. Output is one JSONL of labeled
examples ready for train_head.py.

Per-event score budget: 1 + 2*m_s <= 49 for the 7B anchor (--include_single),
1 + m_s <= 25 for 3B/30B (default).

Usage:
  python -m offline.materialize_labels --local_model <hf checkpoint> \
      --questions data/dck_train/questions.jsonl \
      --snapshot_dir data/dck_train/snapshots_websailor7b \
      --out data/dck_train/labels_7b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from dck.config import DEFAULT_CONFIG
from dck.labels import OVERFLOW, score_event
from dck.model_backend import LocalBackend
from dck.selector import observation_features
from offline.snapshot_utils import iter_snapshots, rebuild_lineage, rebuild_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_model", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--snapshot_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include_single", action="store_true",
                        help="also materialize single-knockout labels (7B anchor)")
    parser.add_argument("--hidden_out", default="",
                        help="path of the pooled hidden-state bank (.pt); "
                             "defaults to <out dir>/hidden_states.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        split = {q["question_id"]: q.get("split", "train")
                 for q in map(json.loads, f) if q}

    backend = LocalBackend(args.local_model, device=args.device)
    config = DEFAULT_CONFIG
    hidden_out = args.hidden_out or os.path.join(
        os.path.dirname(args.out) or ".", "hidden_states.pt")

    n_ok = n_overflow = 0
    hidden_bank = []          # one [d_h] vector per emitted label row, in order
    with open(args.out, "w", encoding="utf-8") as out:
        for snap in iter_snapshots(args.snapshot_dir):
            registry, messages = rebuild_registry(snap)
            lineage = rebuild_lineage(snap)
            candidates = registry.candidates()
            if not candidates:
                continue
            horizon = max(o.time_index for o in candidates)
            features = observation_features(registry, lineage, horizon)
            labels = score_event(
                backend, registry, lineage, messages, snap["answer"],
                config, snap["question_id"], snap["rollout_seed"],
                snap["compaction_index"], include_single=args.include_single,
            )
            if labels.status == OVERFLOW:
                n_overflow += 1
                print(f"[labels] {snap['question_id']} event "
                      f"{snap['compaction_index']}: {OVERFLOW}, skipped")
                continue
            n_ok += 1
            record = {
                "question_id": snap["question_id"],
                "split": split.get(snap["question_id"], "train"),
                "rollout_seed": snap["rollout_seed"],
                "compaction_index": snap["compaction_index"],
                "kind": snap["kind"],
                "factual_logprob": labels.factual_logprob,
                "features": features,
                "labels": labels.to_dict()["labels"],
            }
            # pool hidden states for the labeled roots, same order as the
            # sorted labels list (train_head consumes this alignment)
            for lb in record["labels"]:
                obs = registry.get(lb["obs_id"])
                hidden_bank.append(
                    backend.hidden_state(messages, obs.visible_text).cpu())
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[labels] {snap['question_id']} event "
                  f"{snap['compaction_index']}: {len(labels.labels)} roots")

    torch.save(torch.stack(hidden_bank), hidden_out)
    print(f"materialized {n_ok} events ({n_overflow} overflow-skipped) -> {args.out}")
    print(f"hidden bank ({len(hidden_bank)} x "
          f"{hidden_bank[0].shape[0] if hidden_bank else 0}) -> {hidden_out}")


if __name__ == "__main__":
    main()
