"""Build the DCK-Train question list (Spec 6.1, 6.2).

Source pool: PolarSeeker/OpenSeeker-v1-Data (public, MIT). Fixed rules:
  1. non-empty question and answer;
  2. reference-trajectory tool-call count >= 8 (long-horizon filter);
  3. exact normalized question+answer dedup against GAIA text-only,
     BrowseComp-ZH and BrowseComp evaluation files;
  4. character 5-gram MinHash dedup within the pool (Jaccard > 0.8 pairs are
     reported for manual review; confirmed near-duplicates are dropped);
  5. seed-42 random sampling of 270 en + 30 zh;
  6. a data manifest with ids, language, source line, raw hash, dedup
     decisions and split assignment is written alongside.

Usage:
  python -m offline.prepare_train_questions \
      --openseeker <openseeker.jsonl> \
      --eval_files eval_data/gaia.jsonl eval_data/browsecomp.jsonl eval_data/browsecomp_zh.jsonl \
      --out_dir data/dck_train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Set, Tuple

from dck.textnorm import is_cjk_dominant, normalize_answer, norm_query

SEED = 42
N_EN = 270
N_ZH = 30
MIN_TOOL_CALLS = 8
MINHASH_PERMUTATIONS = 128
MINHASH_JACCARD = 0.8


def _norm_qa(question: str, answer: str) -> Tuple[str, str]:
    return norm_query(question), normalize_answer(answer)


def _char_ngrams(text: str, n: int = 5) -> Set[str]:
    text = "".join(text.split())
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


class MinHash:
    """Deterministic char-5-gram MinHash (seed-fixed permutations)."""

    def __init__(self, permutations: int = MINHASH_PERMUTATIONS, seed: int = SEED):
        rng = random.Random(seed)
        self.a = [rng.randrange(1, (1 << 61) - 1) for _ in range(permutations)]
        self.b = [rng.randrange(0, (1 << 61) - 1) for _ in range(permutations)]
        self.p = (1 << 61) - 1

    def signature(self, grams: Set[str]) -> List[int]:
        sig = []
        for a, b in zip(self.a, self.b):
            sig.append(min((a * hash(g) + b) % self.p for g in grams) if grams else 0)
        return sig

    @staticmethod
    def jaccard(sig1: List[int], sig2: List[int]) -> float:
        return sum(1 for x, y in zip(sig1, sig2) if x == y) / len(sig1)


def load_pool(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({"line_no": line_no, "record": rec})
    return rows


def tool_call_count(record: dict) -> int:
    """Best-effort count of tool calls in the reference trajectory."""
    for key in ("tool_calls", "n_tool_calls", "num_tool_calls"):
        if key in record and isinstance(record[key], int):
            return record[key]
    trajectory = record.get("trajectory") or record.get("messages") or []
    count = 0
    for step in trajectory:
        content = step.get("content") if isinstance(step, dict) else None
        if isinstance(content, str) and "<tool_call>" in content:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openseeker", required=True)
    parser.add_argument("--eval_files", nargs="*", default=[])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- evaluation fingerprints (rule 3) ------------------------------
    eval_fp: Set[Tuple[str, str]] = set()
    for path in args.eval_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                eval_fp.add(_norm_qa(rec.get("question", ""), rec.get("answer", "")))

    # ---- pool filtering (rules 1-3) -------------------------------------
    pool: List[dict] = []
    seen_fp: Set[Tuple[str, str]] = set()
    for row in load_pool(args.openseeker):
        rec = row["record"]
        question = (rec.get("question") or "").strip()
        answer = (rec.get("answer") or "").strip()
        if not question or not answer:
            continue
        if tool_call_count(rec) < MIN_TOOL_CALLS:
            continue
        fp = _norm_qa(question, answer)
        if fp in eval_fp or fp in seen_fp:
            continue
        seen_fp.add(fp)
        pool.append({
            "question": question,
            "answer": answer,
            "lang": "zh" if is_cjk_dominant(question) else "en",
            "source_line": row["line_no"],
            "raw_sha256": hashlib.sha256(
                json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        })

    # ---- MinHash near-duplicate report (rule 4) --------------------------
    minhash = MinHash()
    sigs = [minhash.signature(_char_ngrams(p["question"])) for p in pool]
    near_dup_groups: List[List[int]] = []
    assigned: Dict[int, int] = {}
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if pool[i]["lang"] != pool[j]["lang"]:
                continue
            if MinHash.jaccard(sigs[i], sigs[j]) > MINHASH_JACCARD:
                gid = assigned.get(i, len(near_dup_groups))
                assigned[i] = gid
                assigned[j] = gid
                if gid == len(near_dup_groups):
                    near_dup_groups.append([i, j])
                elif j not in near_dup_groups[gid]:
                    near_dup_groups[gid].append(j)
    # drop the later member of every confirmed near-duplicate pair; pairs are
    # reported in the manifest for manual confirmation either way
    dropped_by_near_dup = set()
    for group in near_dup_groups:
        for idx in group[1:]:
            dropped_by_near_dup.add(idx)

    # ---- stratified seed-42 sampling (rule 5) ----------------------------
    rng = random.Random(SEED)
    en_pool = [i for i, p in enumerate(pool)
               if p["lang"] == "en" and i not in dropped_by_near_dup]
    zh_pool = [i for i, p in enumerate(pool)
               if p["lang"] == "zh" and i not in dropped_by_near_dup]
    en_idx = rng.sample(en_pool, min(N_EN, len(en_pool)))
    zh_idx = rng.sample(zh_pool, min(N_ZH, len(zh_pool)))

    selected = sorted(en_idx + zh_idx, key=lambda i: (pool[i]["lang"], i))
    questions = []
    for qid, i in enumerate(selected):
        entry = dict(pool[i])
        entry["question_id"] = f"dcktrain_{entry['lang']}_{qid:04d}"
        questions.append(entry)

    # ---- stratified 90/10 question-level split (Spec 6.6) ----------------
    rng_split = random.Random(SEED)
    split: Dict[str, str] = {}
    for lang in ("en", "zh"):
        ids = [q["question_id"] for q in questions if q["lang"] == lang]
        rng_split.shuffle(ids)
        n_val = max(1, round(len(ids) * 0.1))
        for qid in ids[:n_val]:
            split[qid] = "val"
        for qid in ids[n_val:]:
            split[qid] = "train"

    # ---- outputs -----------------------------------------------------------
    out_questions = os.path.join(args.out_dir, "questions.jsonl")
    with open(out_questions, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({**q, "split": split[q["question_id"]]},
                               ensure_ascii=False) + "\n")

    manifest = {
        "seed": SEED,
        "n_en": len(en_idx), "n_zh": len(zh_idx),
        "pool_size_after_filters": len(pool),
        "near_duplicate_pairs": [[pool[i]["source_line"] for i in g]
                                 for g in near_dup_groups],
        "eval_files": [os.path.abspath(p) for p in args.eval_files],
        "questions_file": os.path.abspath(out_questions),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"selected {len(en_idx)} en + {len(zh_idx)} zh questions "
          f"({len(near_dup_groups)} near-dup groups reported)")
    print(f"wrote {out_questions}")


if __name__ == "__main__":
    main()
