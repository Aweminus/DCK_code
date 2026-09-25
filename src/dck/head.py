"""Closure head: MLP ranking model and its frozen training recipe (Spec 5.6, 5.7).

Architecture: MLP (d_h + 7) -> 512 -> 1, GELU, no dropout.
For WebSailor-7B (d_h = 3584) this gives 3584*512 + 512 + 512 + 1 = 1,839,617
parameters. Trained with AdamW (lr 1e-3, betas (0.9, 0.999), eps 1e-8,
weight decay 0.01), batch 256, 3 epochs, grad clip 1.0, seed 42. The
checkpoint is selected by validation Spearman, ties broken by validation
Huber loss then the earlier epoch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .features import N_SCALAR_FEATURES


class ClosureHead(nn.Module):
    """MLP (d_h + 7) -> 512 -> 1 with GELU, no dropout."""

    def __init__(self, hidden_dim: int, n_scalar: int = N_SCALAR_FEATURES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_scalar, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(self, hidden: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """hidden [B, d_h], scalars [B, 7] -> scores [B]."""
        return self.mlp(torch.cat([hidden, scalars], dim=-1)).squeeze(-1)


# ------------------------------------------------------------------ metrics

def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    def _ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(list(x)), _ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - my) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def huber(pred: Sequence[float], target: Sequence[float],
          delta: float = 1.0) -> float:
    total = 0.0
    for p, t in zip(pred, target):
        d = p - t
        total += 0.5 * d * d if abs(d) <= delta else delta * (abs(d) - 0.5 * delta)
    return total / max(1, len(pred))


# ------------------------------------------------------------------ training

@dataclass
class TrainConfig:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 3
    grad_clip: float = 1.0
    seed: int = 42
    huber_delta: float = 1.0


def train_head(hidden_dim: int, train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               val: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               cfg: Optional[TrainConfig] = None,
               device: str = "cpu") -> Tuple[ClosureHead, Dict]:
    """Train the closure head; checkpoint selection by val Spearman, ties by
    val Huber then the earlier epoch (Spec 5.7).

    train/val: (hidden [N, d_h], scalars [N, 7], labels [N]).
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    head = ClosureHead(hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, betas=cfg.betas,
                            eps=cfg.eps, weight_decay=cfg.weight_decay)
    h_tr, s_tr, y_tr = (t.to(device) for t in train)
    h_va, s_va, y_va = (t.to(device) for t in val)
    n = h_tr.shape[0]
    if n == 0:
        raise ValueError("TRAIN_ERROR: empty training split")

    best = {"epoch": -1, "spearman": float("-inf"), "huber": float("inf"),
            "state": None}
    g = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(cfg.epochs):
        head.train()
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if idx.numel() == 0:
                continue
            pred = head(h_tr[idx], s_tr[idx])
            loss = nn.functional.huber_loss(pred, y_tr[idx], delta=cfg.huber_delta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
            opt.step()
        head.eval()
        with torch.no_grad():
            val_pred = head(h_va, s_va).cpu().tolist()
        sp = spearman(val_pred, y_va.cpu().tolist())
        hb = huber(val_pred, y_va.cpu().tolist(), cfg.huber_delta)
        # Selection: Spearman desc, Huber asc, epoch asc.
        better = (
            best["epoch"] < 0
            or sp > best["spearman"] + 1e-12
            or (abs(sp - best["spearman"]) <= 1e-12 and hb < best["huber"] - 1e-12)
        )
        if better:
            best = {"epoch": epoch, "spearman": sp, "huber": hb,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}}
    head.load_state_dict(best["state"])
    return head, best


# ------------------------------------------------------------------- serving

def load_head(path: str, device: str = "cpu") -> Tuple[ClosureHead, Dict]:
    """Load a checkpoint saved by save_head()."""
    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    head = ClosureHead(meta["hidden_dim"])
    head.load_state_dict(torch.load(os.path.join(path, "head.pt"),
                                    map_location="cpu"))
    head.to(device)
    head.eval()
    return head, meta


def save_head(path: str, head: ClosureHead, meta: Dict) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(path, "head.pt"))
    clean = {k: v for k, v in meta.items() if k != "state"}
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
