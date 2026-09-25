"""Train an agent-specific closure/single head (Spec 5.7, 6.6).

Reads the labeled JSONL from materialize_labels.py, fits the feature scaler
on the training split only, trains the MLP with the frozen recipe (AdamW
lr 1e-3, batch 256, 3 epochs, grad clip 1.0, seed 42) and selects the
checkpoint by validation Spearman, then Huber, then the earlier epoch.

Usage:
  python -m offline.train_head --labels data/dck_train/labels_7b.jsonl \
      --hidden_dim 3584 --head_kind closure --out_dir checkpoints/closure_head_7b
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from dck.features import FeatureScaler
from dck.head import TrainConfig, save_head, train_head


def load_examples(path: str, label_key: str):
    """(features, value, split, bank_index) per label entry; bank_index is the
    global row number over all entries, keeping the hidden bank aligned even
    when a head kind filters entries out."""
    rows = []
    bank_index = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for lb in rec["labels"]:
                value = lb.get(label_key)
                feats = rec["features"][lb["obs_id"]]
                if value is None:
                    rows.append((feats, None, rec["split"], bank_index))
                else:
                    rows.append((feats, value, rec["split"], bank_index))
                bank_index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--head_kind", choices=["closure", "single"], default="closure")
    parser.add_argument("--hidden_bank", default="",
                        help="pooled hidden-state bank .pt (default: alongside --labels)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    label_key = "closure_delta" if args.head_kind == "closure" else "single_delta"
    rows = load_examples(args.labels, label_key)
    usable = [r for r in rows if r[1] is not None]
    if not usable:
        raise ValueError(f"TRAIN_ERROR: no {label_key} examples in {args.labels}")

    train_rows = [r for r in usable if r[2] == "train"]
    val_rows = [r for r in usable if r[2] == "val"]
    if not val_rows:
        raise ValueError("TRAIN_ERROR: empty validation split")

    # scaler is fit on the training split only (Spec 5.6)
    scaler = FeatureScaler.fit([r[0] for r in train_rows])

    # hidden bank: [N_total, d_h] aligned with the label-row order of the
    # JSONL (materialize_labels.py guarantees this alignment)
    bank_path = args.hidden_bank or os.path.join(
        os.path.dirname(args.labels) or ".", "hidden_states.pt")
    if not os.path.exists(bank_path):
        raise ValueError(
            f"TRAIN_ERROR: hidden-state bank not found at {bank_path}. "
            "Run materialize_labels.py first."
        )
    bank = torch.load(bank_path).float()
    if bank.shape[0] != len(rows):
        raise ValueError(
            f"TRAIN_ERROR: hidden bank has {bank.shape[0]} rows but labels "
            f"file has {len(rows)}"
        )
    if bank.shape[1] != args.hidden_dim:
        raise ValueError(
            f"TRAIN_ERROR: hidden bank dim {bank.shape[1]} != --hidden_dim "
            f"{args.hidden_dim}"
        )

    def _tensors(rs):
        idx = torch.tensor([r[3] for r in rs])
        scalars = torch.tensor([scaler.transform(r[0]) for r in rs],
                               dtype=torch.float32)
        labels = torch.tensor([r[1] for r in rs], dtype=torch.float32)
        return bank[idx], scalars, labels

    train_tensors = _tensors(train_rows)
    val_tensors = _tensors(val_rows)
    head, best = train_head(args.hidden_dim, train_tensors, val_tensors,
                            cfg=TrainConfig(), device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "hidden_dim": args.hidden_dim,
        "head_kind": args.head_kind,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "label_key": label_key,
        "best_epoch": best["epoch"],
        "val_spearman": best["spearman"],
        "val_huber": best["huber"],
        "train_config": TrainConfig().__dict__,
    }
    save_head(args.out_dir, head, meta)
    scaler.save(os.path.join(args.out_dir, "scaler.json"))
    print(f"trained {args.head_kind} head: "
          f"val spearman={best['spearman']:.4f} huber={best['huber']:.4f} "
          f"(epoch {best['epoch']}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
