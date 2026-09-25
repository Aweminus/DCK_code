"""Observation archive and addressable-set registry (Spec 5.1, 5.2, 5.4).

Every complete <tool_response> message block is one atomic Observation. Batched
search/visit calls are never split. The archive stores the full raw tool return
outside the prompt; only visible_text (post window-safety wrapping) may enter
labels, features or protection.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import OBS_ATTR, STUB_LITERAL

_OBS_OPEN_RE = re.compile(r"<(tool_response|protected_evidence)\b([^>]*)>")
_OBS_CLOSE_RE = re.compile(r"</(tool_response|protected_evidence)>")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Observation:
    """One atomic tool return (Spec 5.1)."""

    obs_id: str                 # stable unique id, e.g. "q_0007_r2_o12"
    time_index: int             # 1-based observation counter within the rollout
    episode_id: str
    tool_type: str              # "search" | "visit"
    queries: List[str]          # natural-language query fields of the producing call
    urls: List[str]             # url or url list, in tool-return order
    raw_text: str               # full tool return, prompt-external archive only
    visible_text: str           # what actually entered the agent prompt
    role: str = "tool_response"  # or "protected_evidence" after a compaction
    token_length: Optional[int] = None   # agent tokenizer count of visible_text

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["raw_sha256"] = _sha256(self.raw_text)
        d["visible_sha256"] = _sha256(self.visible_text)
        return d

    # ------------------------------------------------------ serialization
    def serialize(self, text: Optional[str] = None) -> str:
        """Serialize the block as it appears in the working context."""
        tag = "tool_response" if self.role == "tool_response" else "protected_evidence"
        body = self.visible_text if text is None else text
        return f'<{tag} {OBS_ATTR}="{self.obs_id}">{body}</{tag}>'

    def stub(self) -> str:
        """Knockout replacement (Spec 5.4): identical ASCII literal, position,
        role and obs_id preserved; all semantic fields removed."""
        tag = "tool_response" if self.role == "tool_response" else "protected_evidence"
        return f'<{tag} {OBS_ATTR}="{self.obs_id}">{STUB_LITERAL}</{tag}>'


class ObservationRegistry:
    """Tracks which observations are currently independently addressable.

    A_s (Spec 5.4) = original <tool_response> blocks currently in the working
    context + <protected_evidence> entries surviving from the previous
    compaction. Content absorbed into an old summary is never re-expanded from
    the archive.
    """

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self._by_id: Dict[str, Observation] = {}
        self._addressable: Dict[str, bool] = {}   # obs_id -> addressable
        self._counter = 0
        # message index in the *current* working context containing each block
        self._block_message: Dict[str, int] = {}

    # ------------------------------------------------------------ lifecycle
    def register(self, tool_type: str, queries: List[str], urls: List[str],
                 raw_text: str, visible_text: str,
                 token_length: Optional[int] = None) -> Observation:
        """Register a fresh observation entering the working context."""
        self._counter += 1
        obs_id = f"{self.episode_id}_o{self._counter:04d}"
        obs = Observation(
            obs_id=obs_id,
            time_index=self._counter,
            episode_id=self.episode_id,
            tool_type=tool_type,
            queries=list(queries),
            urls=list(urls),
            raw_text=raw_text,
            visible_text=visible_text,
            role="tool_response",
            token_length=token_length,
        )
        self._by_id[obs_id] = obs
        self._addressable[obs_id] = True
        return obs

    def compaction_transition(self, protected_ids: Sequence[str],
                              block_message: Dict[str, int]) -> None:
        """After a compaction: only the protected set remains addressable, and
        its entries are re-serialized as <protected_evidence> blocks (single
        copy, re-ranked fresh at every event; no hysteresis, Spec 5.9)."""
        keep = set(protected_ids)
        for obs_id in self._addressable:
            self._addressable[obs_id] = obs_id in keep
        for obs_id in keep:
            self._by_id[obs_id].role = "protected_evidence"
        self._block_message = dict(block_message)

    def mark_addressable(self, ids: Sequence[str],
                         block_message: Dict[str, int]) -> None:
        """Set the addressable set explicitly without touching roles; used
        when rebuilding state from a collected snapshot."""
        keep = set(ids)
        for obs_id in self._addressable:
            self._addressable[obs_id] = obs_id in keep
        self._block_message = dict(block_message)

    # --------------------------------------------------------------- queries
    def get(self, obs_id: str) -> Observation:
        return self._by_id[obs_id]

    def addressable_ids(self) -> List[str]:
        """Addressable set in chronological (time_index) order."""
        ids = [oid for oid, ok in self._addressable.items() if ok]
        return sorted(ids, key=lambda oid: (self._by_id[oid].time_index, oid))

    def addressable_observations(self) -> List[Observation]:
        return [self._by_id[oid] for oid in self.addressable_ids()]

    def archive_dump_order(self) -> List[Observation]:
        """All observations ever registered (addressable or not), in
        chronological order - the prompt-external archive view."""
        return sorted(self._by_id.values(), key=lambda o: o.time_index)

    def candidates(self) -> List[Observation]:
        """C_s = {o_i : i in A_s} (Spec 5.4)."""
        return self.addressable_observations()

    def message_of(self, obs_id: str) -> Optional[int]:
        return self._block_message.get(obs_id)

    # ------------------------------------------------------------------- IO
    def archive_dump(self) -> List[dict]:
        """Full prompt-external archive for event snapshots (Spec 5.2, 7.1)."""
        return [obs.to_dict() for obs in
                sorted(self._by_id.values(), key=lambda o: o.time_index)]


def parse_obs_blocks(content: str) -> List[Tuple[str, str, str]]:
    """Parse serialized observation blocks from a message.

    Returns list of (obs_id, tag, body) in order of appearance. Used for
    round-trip tests and snapshot re-serialization.
    """
    blocks: List[Tuple[str, str, str]] = []
    pos = 0
    while True:
        m = _OBS_OPEN_RE.search(content, pos)
        if not m:
            break
        tag = m.group(1)
        attrs = m.group(2)
        attr_m = re.search(rf'{OBS_ATTR}="([^"]+)"', attrs)
        close = _OBS_CLOSE_RE.search(content, m.end())
        if attr_m and close and close.group(1) == tag:
            blocks.append((attr_m.group(1), tag, content[m.end():close.start()]))
        pos = m.end()
    return blocks
