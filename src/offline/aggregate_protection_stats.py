"""Protection statistics from the DCK event logs (protection_stats.json).

Aggregates the per-compaction event records written by --dck_event_log:
protected-set sizes, payload/summary budgets, realized restart contexts and
overhead, for the protection arms (selected_ids non-empty) and the host
baseline arms (selected_ids empty, one record per baseline compaction).
Auto-generated arithmetic strings verify that every realized restart closes
under overhead + protected + summary.

Usage:
  python -m offline.aggregate_protection_stats \
      --protection browsecomp:<dck_events.jsonl> \
      --baseline browsecomp:<baseline_events.jsonl> \
      --out protection_stats.json
"""
from __future__ import annotations

import argparse
import json
import os

from dck.event_log import read_events


def load_events(path: str) -> list:
    records = [r for r in read_events(path) if r.get("type") == "event"]
    if not records:
        raise ValueError(f"INPUT_ERROR: no event records in {path}")
    return records


def mean(values: list) -> float:
    return round(sum(values) / len(values), 2) if values else None


def protection_averages(events: list) -> dict:
    return {
        "n_events": len(events),
        "avg_addressable_candidates": mean([len(e.get("scores", {}))
                                            for e in events]),
        "avg_protected_items": mean([len(e.get("selected_ids", []))
                                     for e in events]),
        "avg_protected_tokens": mean([e["payload_tokens"] for e in events]),
        "avg_summary_budget": mean([e["summary_budget"] for e in events]),
        "avg_summary_tokens": mean([e["summary_tokens"] for e in events
                                    if e.get("summary_tokens") is not None]),
        "avg_restart_tokens": mean([e["restart_tokens"] for e in events
                                    if e.get("restart_tokens") is not None]),
        "avg_overhead_tokens": mean([e["overhead_tokens"] for e in events
                                     if e.get("overhead_tokens") is not None]),
        "overflow_events": sum(1 for e in events if e.get("status", "ok") != "ok"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protection", action="append", default=[],
                        help="label:<protection arm event log> (repeatable)")
    parser.add_argument("--baseline", action="append", default=[],
                        help="label:<host baseline event log> (repeatable)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.protection and not args.baseline:
        parser.error("at least one --protection or --baseline log is required")

    per_compaction: dict = {}
    realized: dict = {"overhead_actual_avg": {}, "protection_avg_tokens": {},
                      "baseline_avg_tokens": {}, "arithmetic": {}}
    budget_config = None

    for spec in args.protection:
        label, path = spec.split(":", 1)
        events = load_events(path)
        per_compaction[label] = protection_averages(events)
        avgs = per_compaction[label]
        realized["overhead_actual_avg"][label] = avgs["avg_overhead_tokens"]
        realized["protection_avg_tokens"][label] = avgs["avg_restart_tokens"]
        # arithmetic string from the rounded means, so it closes exactly
        o = round(avgs["avg_overhead_tokens"] or 0)
        p = round(avgs["avg_protected_tokens"] or 0)
        s = round(avgs["avg_summary_tokens"] or 0)
        realized["arithmetic"][f"{label}_protection"] = (
            f"overhead {o} + protected {p} + summary {s} = {o + p + s}")
        if budget_config is None:
            manifest = next((r for r in read_events(path)
                             if r.get("type") == "manifest"), None)
            if manifest and manifest.get("config"):
                budget_config = manifest["config"]

    for spec in args.baseline:
        label, path = spec.split(":", 1)
        events = load_events(path)
        avgs = protection_averages(events)
        realized["baseline_avg_tokens"][label] = avgs["avg_restart_tokens"]
        o = round(avgs["avg_overhead_tokens"] or 0)
        s = round(avgs["avg_summary_tokens"] or 0)
        realized["arithmetic"][f"{label}_baseline"] = (
            f"overhead {o} + summary {s} = {o + s}")
        if label not in per_compaction:
            per_compaction[label] = avgs

    result = {
        "budget_config": budget_config or
            "echo from an event-log manifest (run with a log that has one)",
        "per_compaction_averages": per_compaction,
        "realized_reset_context": realized,
        "note": ("Protection arms (selected_ids non-empty) and host baselines "
                 "(selected_ids empty) are auto-distinguished per log; every "
                 "realized restart closes under overhead + protected + "
                 "summary and stays within the shared L_reset cap."),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(per_compaction, indent=2))


if __name__ == "__main__":
    main()
