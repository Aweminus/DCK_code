"""Run manifests and per-event JSONL logging (Spec 5.2, 7.1).

Every rollout writes one manifest line (configuration hash, model revisions,
stop-table digest) followed by one line per compaction event: trigger state,
candidate features, predicted scores, selected set, payload budgets and
summary budget. The log is append-only and safe to replay offline.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Sequence


class EventLog:
    """Append-only JSONL event log."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("wall_time", time.time())
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_manifest(self, config, paradigm: str, dck_mode: str,
                       model_path: str, head_path: Optional[str],
                       stop_tables_digest: Optional[str],
                       extra: Optional[Dict[str, Any]] = None) -> None:
        rec: Dict[str, Any] = {
            "type": "manifest",
            "paradigm": paradigm,
            "dck_mode": dck_mode,
            "config_hash": config.config_hash(),
            "config": config.manifest(),
            "model_path": model_path,
            "head_path": head_path,
            "stop_tables_digest": stop_tables_digest,
        }
        if extra:
            rec.update(extra)
        self.write(rec)

    def write_event(self, question_id: str, rollout_seed: int,
                    compaction_index: int, trigger_tokens: int,
                    features: Dict[str, Sequence[float]],
                    scores: Dict[str, float],
                    selected_ids: Sequence[str],
                    payload_tokens: int, summary_budget: int,
                    status: str = "ok",
                    archive: Optional[Sequence[Dict]] = None,
                    summary_tokens: Optional[int] = None,
                    restart_tokens: Optional[int] = None,
                    overhead_tokens: Optional[int] = None,
                    working_context_after: Optional[int] = None) -> None:
        rec = {
            "type": "event",
            "question_id": question_id,
            "rollout_seed": rollout_seed,
            "compaction_index": compaction_index,
            "trigger_tokens": trigger_tokens,
            "features": features,
            "scores": scores,
            "selected_ids": list(selected_ids),
            "payload_tokens": payload_tokens,
            "summary_budget": summary_budget,
            "status": status,
            "archive": archive or [],
        }
        # realized-budget instrumentation (present when the arm reports it):
        # summary_tokens = post-truncation length, restart_tokens = measured
        # post-compose context, overhead = restart - payload - summary.
        for key, value in (("summary_tokens", summary_tokens),
                           ("restart_tokens", restart_tokens),
                           ("overhead_tokens", overhead_tokens),
                           ("working_context_after", working_context_after)):
            if value is not None:
                rec[key] = value
        self.write(rec)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_events(path: str):
    """Replay helper: yield parsed records from a JSONL log."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
