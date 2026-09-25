"""Host trigger functions shared by every experimental arm (Spec 5.8).

DCK adds no scheduler of its own: the baseline, Random, Single and DCK arms of
the same host must call the exact same trigger code below. react_agent.py's
baseline loop is routed through resum_trigger_reached() so that the shared-code
requirement holds literally.
"""
from __future__ import annotations

# ------------------------------------------------------------- resum host


def resum_trigger_reached(token_count: int, l_host: int, calls_remaining: int,
                          fraction: float = 0.9) -> bool:
    """Trigger of the resum host:
    1[Tok(H_s) >= 0.9*L_host and calls_remaining > 0]."""
    return token_count >= fraction * l_host and calls_remaining > 0


# ------------------------------------------------------- recent_history host


def recent_history_threshold_tokens(l_host: int, keep_tokens: int = 22 * 1024) -> int:
    """recent_history trims to the most recent 22k tokens once the window is
    reached. Kept as a shared constant for all arms."""
    if keep_tokens > l_host:
        raise ValueError("CONFIGURATION_ERROR: keep window larger than host window")
    return keep_tokens


# ----------------------------------------------------- fixed_interval host


FIXED_INTERVAL_TOOL_CALLS = 10   # restart every 10 completed tool calls


def fixed_interval_trigger(tool_calls: int, calls_remaining: int,
                           interval: int = FIXED_INTERVAL_TOOL_CALLS) -> bool:
    """Trigger of the fixed_interval host: restart every `interval` completed
    tool calls while LLM calls remain."""
    return tool_calls > 0 and tool_calls % interval == 0 and calls_remaining > 0


# -------------------------------------------------------- SelfCompact host


class SelfCompactParams:
    """Frozen normalization rules for the SelfCompact-WS7B adapter (Spec 5.8.2).

    The original SelfCompact deployment agents have 128K-262K windows while
    WebSailor-7B has 32K, so thresholds are normalized to L_host fractions
    instead of claiming byte-level reproduction.
    """

    probe_start_round: int = 3        # rubric probe from round 3
    probe_min_interval: int = 2       # at least 2 rounds between probes
    allow_fraction: float = 0.20      # rubric may trigger above 0.20*L_host
    backstop_fraction: float = 0.30   # unconditional backstop at 0.30*L_host
    max_summaries_per_trajectory: int = 1

    @classmethod
    def allow_threshold(cls, l_host: int) -> float:
        return cls.allow_fraction * l_host

    @classmethod
    def backstop_threshold(cls, l_host: int) -> float:
        return cls.backstop_fraction * l_host
