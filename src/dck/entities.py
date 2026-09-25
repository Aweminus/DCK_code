"""Entity extraction and stop-entity tables (Spec 5.3).

English-dominant text uses a frozen spaCy `en_core_web_trf`; CJK-dominant text
uses a frozen Chinese NER checkpoint exposing the (start, end, type, surface)
interface. When neither asset is installed, a deterministic fallback extractor
is used and MUST be recorded in the run manifest (it changes the lineage
proxy's recall, not its definition).

Stop-entity tables are frozen on each agent's head-training split only:
    G_train^(A,l) = { e : df_train(e) / N_obs > 0.30 }
where document frequency counts an entity once per observation. Validation and
evaluation data never enter the statistics.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .textnorm import is_cjk_dominant, normalize_entity_text

# ------------------------------------------------------------------ backends

_CAP_LATIN = re.compile(r"\b[A-Z][A-Za-z0-9&'.\-]*(?:\s+[A-Z][A-Za-z0-9&'.\-]*)*\b")
_CJK_RUN = re.compile(r"[一-鿿㐀-䶿][一-鿿㐀-䶿·]{1,}")
_FALLBACK_MIN_LEN = 2


class EntityExtractor:
    """(start, end, type, surface) interface over frozen NER assets."""

    backend: str = "fallback_regex"

    def __init__(self, spacy_model: str = "en_core_web_trf",
                 zh_checkpoint: Optional[str] = None):
        self._spacy_nlp = None
        self._zh = None
        try:  # English pipeline
            import spacy  # noqa: WPS433 (runtime optional dep)

            self._spacy_nlp = spacy.load(spacy_model, disable=["parser", "lemmatizer"])
            self.backend = f"spacy:{spacy_model}"
        except Exception:
            self._spacy_nlp = None
        # Chinese NER: any checkpoint exposing .predict(text) ->
        # list[(start, end, type, surface)]; pluggable by path.
        if zh_checkpoint:
            self._zh = _load_zh_ner(zh_checkpoint)
            self.backend += f"|zh:{zh_checkpoint}"

    # ------------------------------------------------------------ extraction
    def extract(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Raw spans; normalization happens in entity_set()."""
        if not text:
            return []
        if is_cjk_dominant(text):
            if self._zh is not None:
                return self._zh(text)
            return [(m.start(), m.end(), "CJK", m.group()) for m in _CJK_RUN.finditer(text)]
        if self._spacy_nlp is not None:
            doc = self._spacy_nlp(text[:200000])
            return [(ent.start_char, ent.end_char, ent.label_, ent.text)
                    for ent in doc.ents]
        # Deterministic fallback: capitalized Latin spans (>=2 chars) and CJK runs.
        spans = [(m.start(), m.end(), "LATIN", m.group())
                 for m in _CAP_LATIN.finditer(text) if len(m.group()) >= _FALLBACK_MIN_LEN]
        if not spans:  # mixed/other scripts: fall back to CJK runs
            spans = [(m.start(), m.end(), "CJK", m.group())
                     for m in _CJK_RUN.finditer(text)]
        return spans

    def entity_set(self, text: str) -> Set[str]:
        """Normalized unique entity surfaces of a text."""
        return {normalize_entity_text(surface) for _, _, _, surface in self.extract(text)} - {""}


def _load_zh_ner(path: str):
    """Load a frozen Chinese NER checkpoint with the shared interface.

    The loader is deliberately conservative: it imports a module by path and
    expects predict(text) -> list[(start, end, type, surface)]."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("dck_zh_ner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise ValueError(f"zh NER checkpoint {path} does not expose predict(text)")
    return predict


# ------------------------------------------------------------- stop tables


def build_stop_table(obs_texts: Iterable[str], extractor: EntityExtractor,
                     threshold: float = 0.30) -> Set[str]:
    """G_train from one (agent, language) training split.

    obs_texts: visible_text of every observation in the split. An entity counts
    once per observation regardless of in-document frequency.
    """
    df: Counter = Counter()
    n_obs = 0
    for text in obs_texts:
        n_obs += 1
        for ent in extractor.entity_set(text):
            df[ent] += 1
    if n_obs == 0:
        return set()
    return {e for e, c in df.items() if c / n_obs > threshold}


@dataclass
class StopTables:
    """Per-(agent, language) frozen stop tables + manifest hash."""

    tables: Dict[str, Set[str]]           # key: f"{agent}|{lang}"
    extractor_backend: str
    threshold: float = 0.30

    def for_question(self, agent: str, lang: str) -> Set[str]:
        """Mixed-language questions use the union of en and zh tables; a
        language with no training observations yields an empty table."""
        primary = self.tables.get(f"{agent}|{lang}", set())
        if lang == "mixed":
            return (self.tables.get(f"{agent}|en", set())
                    | self.tables.get(f"{agent}|zh", set()))
        return primary

    def digest(self) -> str:
        payload = {
            "threshold": self.threshold,
            "extractor": self.extractor_backend,
            "tables": {k: sorted(v) for k, v in sorted(self.tables.items())},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------ IO
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "threshold": self.threshold,
                "extractor_backend": self.extractor_backend,
                "tables": {k: sorted(v) for k, v in self.tables.items()},
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "StopTables":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            tables={k: set(v) for k, v in raw["tables"].items()},
            extractor_backend=raw["extractor_backend"],
            threshold=raw["threshold"],
        )
