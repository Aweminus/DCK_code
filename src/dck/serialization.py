"""Deterministic message serialization (Spec 5.4, 5.10, 7.1).

Knockout and factual serializations differ only in the body of the blocks in
the knockout set Gamma; message position, role and obs_id are preserved
byte-for-byte. Teacher forcing renders with the agent chat template, appends
"Final answer: " + NFKC-normalized a_star, and scores only the tokens after
the longest common prefix of the with-answer / without-answer renderings.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .config import STUB_LITERAL
from .observations import parse_obs_blocks

Message = Dict[str, str]  # {"role": ..., "content": ...}

ANSWER_PREFIX = "Final answer: "


# ------------------------------------------------------------- knockout views
def knockout_messages(messages: Sequence[Message],
                      knockout_ids: Optional[Sequence[str]] = None) -> List[Message]:
    """Copy of `messages` where every observation block whose obs_id is in
    `knockout_ids` has its body replaced by the stub literal. All other
    bytes are preserved."""
    knockout_ids = set(knockout_ids or ())
    out: List[Message] = []
    for msg in messages:
        content = msg.get("content") or ""
        blocks = parse_obs_blocks(content)
        if not blocks or not any(oid in knockout_ids for oid, _, _ in blocks):
            out.append(dict(msg))
            continue
        # Rebuild the content with stubbed bodies. Walk the original string and
        # splice replacements for matched blocks only.
        new_content = content
        for oid, tag, body in blocks:
            if oid not in knockout_ids:
                continue
            original = f"<{tag} obs_id=\"{oid}\">{body}</{tag}>"
            stubbed = f"<{tag} obs_id=\"{oid}\">{STUB_LITERAL}</{tag}>"
            new_content = new_content.replace(original, stubbed, 1)
        out.append({**msg, "content": new_content})
    return out


# ---------------------------------------------------------- teacher forcing
def render_prompt(tokenizer, messages: Sequence[Message],
                  add_generation_prompt: bool = True) -> str:
    """Chat-template rendering of the working context."""
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
    )


def render_forced(tokenizer, messages: Sequence[Message],
                  answer_text: str) -> str:
    """Full teacher-forced rendering: context + ANSWER_PREFIX + answer."""
    # Render with an assistant turn carrying the forced answer so the template
    # contributes exactly one assistant opening.
    forced = [{**m, "content": m.get("content") or ""} for m in messages]
    forced.append({"role": "assistant", "content": ANSWER_PREFIX + answer_text})
    return tokenizer.apply_chat_template(forced, tokenize=False)


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def forced_target(tokenizer, messages: Sequence[Message],
                  answer_text: str) -> Tuple[List[int], int]:
    """Token ids of the full forced rendering plus the length of the masked
    longest-common-prefix (everything before it is prompt, not target).

    Returns (full_ids, prefix_len): score ids full_ids[prefix_len:].
    """
    prompt_ids = tokenizer.encode(
        render_prompt(tokenizer, messages, add_generation_prompt=True),
        add_special_tokens=False,
    )
    full_ids = tokenizer.encode(
        render_forced(tokenizer, messages, answer_text),
        add_special_tokens=False,
    )
    prefix = common_prefix_len(prompt_ids, full_ids)
    if prefix >= len(full_ids):
        raise ValueError("SERIALIZATION_ERROR: forced rendering adds no target tokens")
    return full_ids, prefix
