"""Protection selectors: DCK / Single / Random (Spec 5.9, 7.1).

All three selectors share the same candidate set, the same payload builder,
the same presentation order and the same budget code; they differ only in how
the selected set B_s is chosen (comparator isomorphism).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch

from .config import DCKConfig
from .features import FeatureScaler, repeated_query_flag, scalar_features
from .head import ClosureHead
from .observations import Observation, ObservationRegistry
from .serialization import Message


def observation_features(registry: ObservationRegistry, lineage,
                         horizon: int) -> Dict[str, List[float]]:
    """The seven scalar features per addressable observation at this event."""
    features: Dict[str, List[float]] = {}
    addressable = registry.addressable_observations()
    addressable_times = {obs.time_index for obs in addressable}
    for obs in addressable:
        earlier_queries: List[str] = []
        for other in registry.archive_dump_order():
            if other.time_index < obs.time_index:
                earlier_queries.extend(other.queries)
        closure_n = lineage.closure_size(obs.time_index, horizon, addressable_times)
        first_n = len(lineage.first_entities.get(obs.time_index, set()))
        tok = obs.token_length
        if tok is None:
            raise ValueError("FEATURE_ERROR: missing token_length on observation")
        features[obs.obs_id] = scalar_features(
            t=obs.time_index, s=horizon, closure_addressable=closure_n,
            first_entity_count=first_n, visible_tokens=tok,
            repeated_query=repeated_query_flag(obs.queries, earlier_queries),
        )
    return features


def score_candidates(backend, head: ClosureHead, scaler: FeatureScaler,
                     registry: ObservationRegistry, lineage,
                     messages: Sequence[Message],
                     hidden_cache: Optional[Dict[str, "torch.Tensor"]] = None,
                     current_body: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """I_hat for every addressable observation: hidden state pooled over the
    observation's current in-context span + z-scored scalars.

    hidden_cache: obs_id -> pooled vector, filled at append time (control
    order, Spec 7.1) and at compaction time for protected survivors. When an
    entry is missing the backend pools live over `current_body` (payload body
    for protected entries, visible_text otherwise)."""
    horizon = max(
        (obs.time_index for obs in registry.addressable_observations()), default=0
    )
    feats = observation_features(registry, lineage, horizon)
    hidden_cache = hidden_cache if hidden_cache is not None else {}
    current_body = current_body or {}
    scores: Dict[str, float] = {}
    with torch.no_grad():
        for obs in registry.addressable_observations():
            if obs.obs_id in hidden_cache:
                hidden = hidden_cache[obs.obs_id]
            else:
                span = current_body.get(obs.obs_id, obs.visible_text)
                hidden = backend.hidden_state(messages, span)
            scalars = torch.tensor(
                [scaler.transform(feats[obs.obs_id])], dtype=torch.float32
            )
            pred = head(hidden.unsqueeze(0), scalars).item()
            scores[obs.obs_id] = pred
    return scores


class BaseSelector:
    """Common candidate/payload/presentation plumbing."""

    def select(self, registry: ObservationRegistry,
               scores: Dict[str, float],
               config: DCKConfig,
               compaction_index: int = 0) -> List[Observation]:
        raise NotImplementedError


class DCKSelector(BaseSelector):
    """Top min(8, |C_s|) by sort_key (-I_hat, time_index, obs_id)."""

    def __init__(self, backend, head: ClosureHead, scaler: FeatureScaler):
        self.backend = backend
        self.head = head
        self.scaler = scaler

    def score(self, registry, lineage, messages,
              hidden_cache=None, current_body=None) -> Dict[str, float]:
        return score_candidates(self.backend, self.head, self.scaler,
                                registry, lineage, messages,
                                hidden_cache=hidden_cache, current_body=current_body)

    def select(self, registry, scores, config, compaction_index: int = 0):
        from .protection import select_protected

        _, p_s = select_protected(registry.candidates(), scores, config)
        return p_s


class SingleSelector(DCKSelector):
    """Identical Top min(8, |C_s|) selection and budget as DCKSelector
    (comparator isomorphism, Spec 7.1); only the supervision differs — the
    scores are produced by the single-knockout head, not the closure head."""


class RandomSelector(BaseSelector):
    """Uniform subset of size min(8, |C_s|), seeded by
    (question_id, rollout_seed, compaction_index)."""

    def __init__(self, question_id: str, rollout_seed: int):
        self.question_id = question_id
        self.rollout_seed = rollout_seed

    def select(self, registry, scores, config, compaction_index: int = 0):
        candidates = sorted(registry.candidates(),
                            key=lambda o: (o.time_index, o.obs_id))
        k = min(config.k_protect, len(candidates))
        rng = random.Random(f"{self.question_id}|{self.rollout_seed}|{compaction_index}")
        chosen = rng.sample(candidates, k) if k else []
        return sorted(chosen, key=lambda o: (o.time_index, o.obs_id))
