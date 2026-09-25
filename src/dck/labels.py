"""Closure-knockout supervision labels (Spec 5.4, 5.5, 6.5).

Per compaction event s and root observation t in the sampled root set:

    Delta_t = clip( logP(a* | H_s) - logP(a* | H_s[Gamma_{t,s} -> stub]),
                    -5, +5 )

where Gamma_{t,s} = {t} U (D_<=s(t) intersect A_s). Positive labels mark
observations whose removal hurts the forced final answer. Any serialization
that overflows the host window marks the whole event SCORE_CONTEXT_OVERFLOW
and it is excluded from training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import DCKConfig
from .observations import ObservationRegistry
from .serialization import Message, knockout_messages
from .textnorm import normalize_answer

OVERFLOW = "SCORE_CONTEXT_OVERFLOW"


def equidistant_roots(n_candidates: int, max_roots: int) -> List[int]:
    """Deterministic equidistant 1-based indices into the chronological
    candidate list (Spec 6.5): m_s = min(max_roots, n_candidates) roots at
    idx_j = floor(j*(n-1)/(m_s-1)) + 1, endpoints included."""
    if n_candidates <= 0:
        return []
    m = min(max_roots, n_candidates)
    if m == 1:
        return [1]
    return [int(j * (n_candidates - 1) // (m - 1)) + 1 for j in range(m)]


@dataclass
class RootLabel:
    """Closure and single knockout labels of one root observation."""

    closure_delta: float
    closure_logprob: float
    gamma_size: int
    single_delta: Optional[float] = None      # 7B anchor only
    single_logprob: Optional[float] = None


@dataclass
class EventLabels:
    """Labels for one compaction event."""

    question_id: str
    rollout_seed: int
    compaction_index: int
    factual_logprob: Optional[float] = None
    labels: Dict[str, RootLabel] = field(default_factory=dict)
    status: str = "ok"                      # "ok" | OVERFLOW
    overflow_obs_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "rollout_seed": self.rollout_seed,
            "compaction_index": self.compaction_index,
            "factual_logprob": self.factual_logprob,
            "labels": [
                {
                    "obs_id": oid,
                    "closure_delta": lb.closure_delta,
                    "closure_logprob": lb.closure_logprob,
                    "gamma_size": lb.gamma_size,
                    "single_delta": lb.single_delta,
                    "single_logprob": lb.single_logprob,
                }
                for oid, lb in sorted(self.labels.items())
            ],
            "status": self.status,
            "overflow_obs_id": self.overflow_obs_id,
        }


def score_event(backend, registry: ObservationRegistry, lineage,
                messages: Sequence[Message], answer: str,
                config: DCKConfig, question_id: str, rollout_seed: int,
                compaction_index: int,
                include_single: bool = True,
                addressable_time: Optional[Dict[str, int]] = None) -> EventLabels:
    """Compute closure-knockout (and optionally single-knockout) labels for
    all roots in the sampled root set.

    `lineage` must already contain every observation up to the event horizon.
    `addressable_time` maps obs_id -> 1-based observation index if the
    registry's time_index differs from the lineage key (normally identical).
    Per event this costs 1 factual + per root 1 closure (+1 single) teacher
    forcing scores: <= 49 for the 7B anchor, <= 25 for 3B/30B (Spec 6.5).
    """
    event = EventLabels(
        question_id=question_id, rollout_seed=rollout_seed,
        compaction_index=compaction_index,
    )
    answer_norm = normalize_answer(answer)
    if not answer_norm:
        raise ValueError("LABEL_ERROR: empty normalized gold answer")

    candidates = registry.candidates()
    n = len(candidates)
    if n == 0:
        return event

    time_of = addressable_time or {obs.obs_id: obs.time_index for obs in candidates}
    horizon = max(obs.time_index for obs in candidates)

    try:
        factual = backend.loglikelihood(messages, answer_norm)
    except ValueError as exc:
        if str(exc) == OVERFLOW:
            event.status = OVERFLOW
            return event
        raise
    event.factual_logprob = factual

    def _clip(value: float) -> float:
        return max(-config.label_clip, min(config.label_clip, value))

    roots = equidistant_roots(n, config.max_label_roots)
    for r in roots:
        root = candidates[r - 1]
        t = time_of[root.obs_id]
        descendants = lineage.descendants(t, horizon)
        gamma_ids = {root.obs_id}
        for u in descendants:
            # Map lineage index u back to the addressable observation with that
            # time index (at most one per registry).
            for obs in candidates:
                if obs.time_index == u:
                    gamma_ids.add(obs.obs_id)
                    break
        try:
            ko_closure = backend.loglikelihood(
                knockout_messages(messages, gamma_ids), answer_norm)
            ko_single = None
            if include_single:
                ko_single = backend.loglikelihood(
                    knockout_messages(messages, {root.obs_id}), answer_norm)
        except ValueError as exc:
            if str(exc) == OVERFLOW:
                event.status = OVERFLOW
                event.overflow_obs_id = root.obs_id
                event.labels.clear()
                return event
            raise
        event.labels[root.obs_id] = RootLabel(
            closure_delta=_clip(factual - ko_closure),
            closure_logprob=ko_closure,
            gamma_size=len(gamma_ids),
            single_delta=_clip(factual - ko_single) if ko_single is not None else None,
            single_logprob=ko_single,
        )
    return event
