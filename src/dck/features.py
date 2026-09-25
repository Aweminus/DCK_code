"""Observation features and the frozen feature scaler (Spec 5.6).

Seven scalar features per (event s, observation t):
    [ t, s, s - t, |D_<=s(t) ∩ A_s|, |E_t|, Tok(visible_text), r_t ]
plus the last-layer mean-pooled hidden state over the visible_text span.
Scalar features are z-scored with statistics frozen on the training split;
a dimension whose std is < 1e-12 is mapped to 0 (constant feature).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Sequence

from .textnorm import norm_query

N_SCALAR_FEATURES = 7


def scalar_features(t: int, s: int, closure_addressable: int,
                    first_entity_count: int, visible_tokens: int,
                    repeated_query: int) -> List[float]:
    """The seven scalar features of observation t at event s."""
    return [float(t), float(s), float(s - t), float(closure_addressable),
            float(first_entity_count), float(visible_tokens),
            float(repeated_query)]


def repeated_query_flag(queries: Sequence[str],
                        earlier_queries: Sequence[str]) -> int:
    """r_t in {0, 1}: exact norm_query match against any earlier query in the
    same rollout. No embeddings, edit distance or fuzzy matching."""
    if not queries:
        return 0
    earlier = {norm_query(q) for q in earlier_queries if q}
    return int(any(norm_query(q) in earlier for q in queries))


@dataclass
class FeatureScaler:
    """Frozen z-score statistics; sigma < 1e-12 maps the dimension to 0."""

    mean: List[float]
    std: List[float]

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> "FeatureScaler":
        if not rows:
            raise ValueError("SCALER_ERROR: cannot fit on empty rows")
        d = len(rows[0])
        mean = [0.0] * d
        for row in rows:
            for j, v in enumerate(row):
                mean[j] += v
        mean = [m / len(rows) for m in mean]
        std = [0.0] * d
        for row in rows:
            for j, v in enumerate(row):
                std[j] += (v - mean[j]) ** 2
        std = [math.sqrt(v / len(rows)) for v in std]
        return cls(mean=mean, std=std)

    def transform(self, row: Sequence[float]) -> List[float]:
        if len(row) != len(self.mean):
            raise ValueError("SCALER_ERROR: feature dimension mismatch")
        return [
            (v - m) / s if s >= 1e-12 else 0.0
            for v, m, s in zip(row, self.mean, self.std)
        ]

    # ------------------------------------------------------------------ IO
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(mean=raw["mean"], std=raw["std"])
