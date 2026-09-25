"""Dynamic protection selection and budget-conserving composition (Spec 5.9, 5.10).

Payload rho(o_t) = metadata header (<= 192 agent tokens: fixed fields <= 32,
query <= 64, source <= 96) + evidence body (<= 512; head 384 + tail 128 when
over), so Tok_A(rho(o_t)) <= 704. Selection sorts by (-I_hat, time_index,
obs_id), keeps Top min(8, |C_s|), then re-orders the selected set
chronologically for presentation. Protected blocks are never truncated and K
is never temporarily reduced.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import DCKConfig
from .observations import Observation, parse_obs_blocks
from .serialization import Message

PROTECTED_HEADER = "[Protected evidence]\n"
SUMMARY_HEADER = "[Search history summary]\n"


# ------------------------------------------------------------------- scoring
def assert_finite_scores(scores: Dict[str, float]) -> None:
    """NaN / +/-inf predictions are implementation failures of the rollout,
    not sortable values (Spec 5.9)."""
    for obs_id, value in scores.items():
        if not math.isfinite(value):
            raise ValueError(
                f"SCORE_NOT_FINITE: obs_id={obs_id} score={value}"
            )


def select_protected(candidates: Sequence[Observation],
                     scores: Dict[str, float],
                     config: DCKConfig) -> Tuple[List[Observation], List[Observation]]:
    """B_s (selection order) and P_s (chronological presentation order).

    candidates: addressable observations C_s. scores: obs_id -> I_hat."""
    assert_finite_scores(scores)
    selection_order = sorted(
        candidates,
        key=lambda o: (-scores[o.obs_id], o.time_index, o.obs_id),
    )
    b_s = selection_order[: min(config.k_protect, len(candidates))]
    p_s = sorted(b_s, key=lambda o: (o.time_index, o.obs_id))
    return b_s, p_s


# ------------------------------------------------------------------- payloads
def _token_prefix(tokenizer, text: str, max_tokens: int) -> Tuple[str, int]:
    """Deterministic token-level truncation with the agent tokenizer."""
    if not text:
        return "", 0
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text, len(ids)
    return tokenizer.decode(ids[:max_tokens]), max_tokens


def _evidence_body(tokenizer, visible_text: str,
                   config: DCKConfig) -> Tuple[str, int]:
    """r(o_t): <= 512 tokens; head 384 + tail 128 when over budget."""
    ids = tokenizer.encode(visible_text, add_special_tokens=False)
    if len(ids) <= config.evidence_payload_tokens:
        return visible_text, len(ids)
    head = tokenizer.decode(ids[: config.evidence_head_tokens])
    tail = tokenizer.decode(ids[-config.evidence_tail_tokens:])
    return head + "\n[...]\n" + tail, config.evidence_payload_tokens


def _fixed_fields(obs: Observation, tokenizer,
                  config: DCKConfig) -> Tuple[str, int]:
    """Well-formed metadata header plus its fixed-field token count.

    The fixed fields are the semantic content (tag label, obs_id, tool_type,
    sha256 prefix) and must fit the frozen 32-token budget; the sha prefix
    shrinks (16 -> 8 -> 4 hex chars) before a configuration error is raised.
    The header itself is never truncated mid-attribute."""
    full_sha = hashlib.sha256(obs.raw_text.encode("utf-8")).hexdigest()
    for sha_len in (16, 8, 4):
        sha = full_sha[:sha_len]
        semantic = f"protected_evidence|{obs.obs_id}|{obs.tool_type}|{sha}"
        n = len(tokenizer.encode(semantic, add_special_tokens=False))
        if n <= config.fixed_fields_tokens:
            header = (
                f'<protected_evidence obs_id="{obs.obs_id}" '
                f'tool_type="{obs.tool_type}" sha256="{sha}">\n'
            )
            return header, n
    raise ValueError(
        "CONFIGURATION_ERROR: fixed fields (tag, obs_id, tool_type, sha256) "
        f"exceed {config.fixed_fields_tokens} agent tokens even with a 4-hex "
        f"sha prefix; obs_id scheme is too long for {obs.obs_id}"
    )


def build_payload(obs: Observation, tokenizer,
                  config: DCKConfig) -> Tuple[str, int]:
    """Serialize rho(o_t) as one <protected_evidence> block; returns
    (block_text, agent_token_count)."""
    fixed_text, fixed_n = _fixed_fields(obs, tokenizer, config)
    query_text, query_n = _token_prefix(
        tokenizer, "\n".join(obs.queries), config.query_tokens)
    source_json = json.dumps(obs.urls, ensure_ascii=False, separators=(",", ":"))
    source_text, source_n = _token_prefix(
        tokenizer, source_json, config.source_tokens)
    body_text, body_n = _evidence_body(tokenizer, obs.visible_text, config)
    block = (
        f"{fixed_text}[query] {query_text}\n[source] {source_text}\n"
        f"{body_text}\n</protected_evidence>"
    )
    total = fixed_n + query_n + source_n + body_n
    if total > config.per_observation_cap:
        raise ValueError(
            f"BUDGET_ERROR: payload for {obs.obs_id} is {total} > "
            f"{config.per_observation_cap} agent tokens"
        )
    return block, total


def build_payloads(p_s: Sequence[Observation], tokenizer,
                   config: DCKConfig) -> Tuple[str, int]:
    """Concatenation of all protected blocks plus their total token count."""
    parts: List[str] = []
    total = 0
    for obs in p_s:
        block, n = build_payload(obs, tokenizer, config)
        parts.append(block)
        total += n
    if total > config.protected_budget_tokens:
        raise ValueError(
            f"BUDGET_ERROR: protected blocks total {total} > "
            f"{config.protected_budget_tokens} agent tokens"
        )
    return "\n".join(parts), total


# ------------------------------------------------- summarizer-side operations
def replace_protected_with_ids(messages: Sequence[Message],
                               protected_ids: Sequence[str]) -> List[Message]:
    """M_H(H_s, B_s): in the summarizer input each selected observation keeps
    its original position but its body becomes a stable ID placeholder, so the
    summarizer does not duplicate content that survives verbatim."""
    keep = set(protected_ids)
    out: List[Message] = []
    for msg in messages:
        content = msg.get("content") or ""
        blocks = parse_obs_blocks(content)
        if not blocks or not any(oid in keep for oid, _, _ in blocks):
            out.append(dict(msg))
            continue
        new_content = content
        for oid, tag, body in blocks:
            if oid not in keep:
                continue
            original = f'<{tag} obs_id="{oid}">{body}</{tag}>'
            placeholder = f'<{tag} obs_id="{oid}">[protected_evidence:{oid}]</{tag}>'
            new_content = new_content.replace(original, placeholder, 1)
        out.append({**msg, "content": new_content})
    return out


def truncate_summary(tokenizer, summary_text: str,
                     l_sum: int) -> Tuple[str, int]:
    """Summaries are generated with the summarizer's own tokenizer but must be
    re-encoded with the agent tokenizer and cut to the first L_sum agent
    tokens (Spec 5.10)."""
    ids = tokenizer.encode(summary_text, add_special_tokens=False)
    if len(ids) <= l_sum:
        return summary_text, len(ids)
    return tokenizer.decode(ids[:l_sum]), l_sum


def compose(system_prompt: str, question: str, summary_text: str,
            protected_block: str) -> List[Message]:
    """Compose_H(q, S_s, payloads): the reset context."""
    messages: List[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "user", "content": SUMMARY_HEADER + summary_text},
    ]
    if protected_block:
        messages.append({"role": "user", "content": PROTECTED_HEADER + protected_block})
    return messages
