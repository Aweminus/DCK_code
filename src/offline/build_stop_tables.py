"""Freeze per-(agent, language) stop-entity tables (Spec 5.3, 6).

G_train^(A,l) = { e : df_train(e) / N_obs > 0.30 }, where document frequency
counts an entity once per observation and only training-split observations
enter the statistics. Validation and evaluation data never do.

Usage:
  python -m offline.build_stop_tables --questions data/dck_train/questions.jsonl \
      --snapshot_dir data/dck_train/snapshots_websailor7b \
      --agent websailor7b --out data/dck_train/stop_tables_7b.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from dck.entities import EntityExtractor, StopTables, build_stop_table
from dck.textnorm import is_cjk_dominant
from offline.snapshot_utils import iter_snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--snapshot_dir", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        split = {q["question_id"]: q.get("split", "train")
                 for q in map(json.loads, f) if q}

    extractor = EntityExtractor()
    by_lang = defaultdict(list)
    for snap in iter_snapshots(args.snapshot_dir):
        if split.get(snap["question_id"]) != "train":
            continue
        lang = "zh" if is_cjk_dominant(snap["question"]) else "en"
        for rec in snap["archive"]:
            by_lang[lang].append(rec["visible_text"])

    tables = {
        lang: build_stop_table(texts, extractor, args.threshold)
        for lang, texts in by_lang.items()
    }
    stop_tables = StopTables(
        tables={f"{args.agent}|{lang}": ents for lang, ents in tables.items()},
        extractor_backend=extractor.backend,
        threshold=args.threshold,
    )
    stop_tables.save(args.out)
    print(f"wrote {args.out}: "
          + ", ".join(f"{k}={len(v)}" for k, v in stop_tables.tables.items()))


if __name__ == "__main__":
    main()
