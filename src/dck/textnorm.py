"""Text and entity normalization (Spec 5.3, 5.6).

All normalization is deterministic and offline-frozen: exact checkpoints and
dependency versions are recorded in run manifests, never inferred at runtime.
"""
from __future__ import annotations

import re
import unicodedata

try:  # optional dependency; identity fallback is recorded in the manifest
    from opencc_python_reimplemented import OpenCC as _OpenCC

    _t2s = _OpenCC("t2s").convert
except Exception:  # pragma: no cover - environment dependent
    _t2s = None

_LATIN_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_COLLAPSE_WS = re.compile(r"\s+")
_HAS_CJK = re.compile(r"[一-鿿㐀-䶿]")
# Explicit punctuation class: ASCII + general punctuation + CJK punctuation.
_PUNCT_CHARS = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "‘’“”–—…·"
    "　、。，：；？！（）《》“”‘’"
)


def _strip_edge_punct(text: str) -> str:
    """Strip edge punctuation/whitespace (deterministic, no regex \\p support)."""
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or text[start] in _PUNCT_CHARS):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _PUNCT_CHARS):
        end -= 1
    return text[start:end]


def is_cjk_dominant(text: str) -> bool:
    """Route a text to the CJK pipeline when it is CJK dominant."""
    if not text:
        return False
    cjk = len(_HAS_CJK.findall(text))
    return cjk * 2 > len(text)


def _is_latin_only(text: str) -> bool:
    return bool(text) and all(
        ord(ch) < 0x2E80 for ch in text if not ch.isspace()
    )


def normalize_entity_text(text: str) -> str:
    """Entity-level normalization (Spec 5.3): Unicode NFKC, whitespace and
    punctuation normalization, Latin entities lowercased with leading articles
    removed, traditional Chinese mapped to simplified."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _strip_edge_punct(text)
    text = _COLLAPSE_WS.sub(" ", text).strip()
    if _t2s is not None and _HAS_CJK.search(text):
        text = _t2s(text)
    if _is_latin_only(text):
        text = _LATIN_ARTICLES.sub("", text, count=1)
        text = text.lower()
    return text


def norm_query(q: str) -> str:
    """Query normalization for the repeated-query flag r_t (Spec 5.6):
    collapse_space(lower_Latin(strip_edge_punct(NFKC(q)))). Exact string match
    only - no embeddings, edit distance, stemming or LLM judgement."""
    text = unicodedata.normalize("NFKC", q or "")
    text = _strip_edge_punct(text)
    text = _COLLAPSE_WS.sub(" ", text).strip()
    return text.lower()


def normalize_answer(a_star: str) -> str:
    """Gold answer normalization for teacher forcing (Spec 5.4 step 2):
    Unicode NFKC plus edge-whitespace cleanup. Must be non-empty upstream."""
    return unicodedata.normalize("NFKC", a_star or "").strip()
