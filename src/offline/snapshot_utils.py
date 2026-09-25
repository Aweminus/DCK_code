"""Rebuild in-memory state from a collected snapshot (Spec 6.4, 6.5).

Snapshots are written by DCKReactAgent(snapshot_dir=...): one per real
compaction event (pre-compaction context) plus terminal horizons for
trajectories without events. This module reconstructs the registry and
lineage so labels and features can be materialized offline.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from dck.observations import Observation, ObservationRegistry, parse_obs_blocks
from dck.serialization import Message


def load_snapshot(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rebuild_registry(snapshot: dict) -> Tuple[ObservationRegistry, List[Message]]:
    """Rebuild the observation registry; addressability is recovered by
    parsing the observation blocks actually present in the snapshot context
    (content absorbed into an old summary is never re-expanded, Spec 6.4.6)."""
    archive = snapshot["archive"]
    registry = ObservationRegistry(str(snapshot["question_id"]))
    messages = [dict(m) for m in snapshot["messages"]]

    in_context: Dict[str, Tuple[str, int]] = {}   # obs_id -> (tag, msg_index)
    for idx, msg in enumerate(messages):
        for oid, tag, _body in parse_obs_blocks(msg.get("content") or ""):
            in_context[oid] = (tag, idx)

    # replay the archive in time order
    obs_by_id: Dict[str, Observation] = {}
    for rec in sorted(archive, key=lambda r: r["time_index"]):
        obs = registry.register(
            tool_type=rec["tool_type"], queries=rec["queries"], urls=rec["urls"],
            raw_text=rec["raw_text"], visible_text=rec["visible_text"],
            token_length=rec.get("token_length"),
        )
        if obs.obs_id != rec["obs_id"]:
            # archive and registry must agree on ids
            raise ValueError(f"SNAPSHOT_ERROR: obs_id mismatch {obs.obs_id} != {rec['obs_id']}")
        obs_by_id[obs.obs_id] = obs

    # addressable = blocks present in the snapshot context
    present = set(in_context)
    if set(obs_by_id) != present:
        registry.mark_addressable(
            sorted(present),
            {oid: in_context[oid][1] for oid in present},
        )
    return registry, messages


def rebuild_lineage(snapshot: dict):
    """Rebuild the LineageState exactly as recorded at collection time."""
    from dck.lineage import LineageState

    lin = snapshot["lineage"]
    state = LineageState(
        question_entities=set(lin["question_entities"]),
        stop_entities=set(lin["stop_entities"]),
    )
    for key in sorted(lin["obs_entities"], key=int):
        t = int(key)
        state.add_observation(
            t,
            obs_entities=set(lin["obs_entities"][key]),
            query_entities=set(lin["query_entities"][key]),
        )
    return state


def iter_snapshots(snapshot_dir: str):
    for fname in sorted(os.listdir(snapshot_dir)):
        if fname.endswith(".json"):
            yield load_snapshot(os.path.join(snapshot_dir, fname))
