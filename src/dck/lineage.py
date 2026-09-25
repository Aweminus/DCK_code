"""Temporal entity lineage (Spec 5.3, Lemma 1).

The recursive temporal rule
    u in D_<=s(t)  iff  ent(q_u) ∩ ⋃_{v in {t} ∪ D_<u(t)} E_v  != ∅
is exactly the reachability set of the forward entity-reuse graph G_s, hence
unique, acyclic and computable by a single chronological scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Set, Tuple


@dataclass
class LineageState:
    """Streaming lineage state over one rollout, keyed by 1-based obs index.

    All inputs are prefix-determined, so appending future observations never
    changes existing first-entity sets or closures (prefix invariance, Spec 7.1
    test 1).
    """

    question_entities: Set[str] = field(default_factory=set)
    stop_entities: Set[str] = field(default_factory=set)
    # per observation index t (1-based)
    obs_entities: Dict[int, Set[str]] = field(default_factory=dict)      # ent(o_t)
    first_entities: Dict[int, Set[str]] = field(default_factory=dict)    # E_t
    query_entities: Dict[int, Set[str]] = field(default_factory=dict)    # ent(q_t)

    def add_observation(self, t: int, obs_entities: Set[str],
                        query_entities: Set[str]) -> None:
        """Register observation o_t together with the entities of the queries
        q_t that produced it. Call in chronological order."""
        if t in self.obs_entities:
            raise ValueError(f"observation index {t} registered twice")
        seen = set(self.question_entities)
        for j in range(1, t):
            seen |= self.obs_entities.get(j, set())
        self.obs_entities[t] = set(obs_entities)
        self.query_entities[t] = set(query_entities)
        self.first_entities[t] = set(obs_entities) - seen - self.stop_entities

    def descendants(self, t: int, horizon: int) -> Set[int]:
        """D_<=s(t): forward closure of root t at horizon s (Spec 5.3).

        Chronological scan: u is admitted iff some query of u reuses an entity
        first introduced by t or by an already-admitted descendant."""
        if t > horizon:
            return set()
        admitted: Set[int] = set()
        source_ents = set(self.first_entities.get(t, set()))
        for u in range(t + 1, horizon + 1):
            if self.query_entities.get(u, set()) & source_ents:
                admitted.add(u)
                source_ents |= self.first_entities.get(u, set())
        return admitted

    def closure_size(self, t: int, horizon: int, addressable: Set[int]) -> int:
        """|D_<=s(t) ∩ A_s|: addressable descendants only (feature 4, Spec 5.6)."""
        return len(self.descendants(t, horizon) & addressable)


def build_lineage(question: str, observations: Sequence[Tuple[int, str, str]],
                  extractor, stop_entities: Set[str]) -> LineageState:
    """Convenience constructor.

    observations: (t, visible_text, query_text_joined) per observation.
    extractor: dck.entities.EntityExtractor.
    """
    state = LineageState(
        question_entities=extractor.entity_set(question),
        stop_entities=set(stop_entities),
    )
    for t, visible_text, query_text in observations:
        state.add_observation(
            t,
            obs_entities=extractor.entity_set(visible_text),
            query_entities=extractor.entity_set(query_text),
        )
    return state
