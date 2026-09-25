"""Frozen budget configuration for DCK protected compaction (Spec 5.8, 5.10).

All token counts are Agent tokens, i.e. counted with the *search agent's*
tokenizer. L_host = MAX_CONTEXT*1024 - 1000.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------- stub literal
# Spec 5.4: every knocked-out observation is replaced by the identical ASCII
# literal below, preserving message position / role / obs_id.
STUB_LITERAL = "[result withheld]"

TOOL_RESPONSE_OPEN = "<tool_response"
TOOL_RESPONSE_CLOSE = "</tool_response>"
PROTECTED_OPEN = "<protected_evidence"
PROTECTED_CLOSE = "</protected_evidence>"

# Serialization of an addressable observation block (Spec 5.4):
#   <tool_response obs_id="...">[result withheld]</tool_response>
#   <protected_evidence obs_id="...">[result withheld]</protected_evidence>
OBS_ATTR = "obs_id"


@dataclass
class DCKConfig:
    """Frozen per-run budget configuration. Mutating it mid-experiment is a
    CONFIGURATION_ERROR."""

    # ---- host window (Spec 5.8.1) -----------------------------------------
    max_context_k: int = 32                 # MAX_CONTEXT in k tokens
    reserved_window_tokens: int = 1000      # L_host = 32*1024 - 1000
    trigger_fraction: float = 0.9           # resum host trigger fraction
    max_llm_calls_per_run: int = 60

    # ---- protection capacity (Spec 5.9, 5.10) ------------------------------
    k_protect: int = 8                      # Top-K capacity
    evidence_payload_tokens: int = 512      # max evidence body tokens
    evidence_head_tokens: int = 384         # head keep when over budget
    evidence_tail_tokens: int = 128         # tail keep when over budget
    fixed_fields_tokens: int = 32           # label/obs_id/tool_type/sha256
    query_tokens: int = 64                  # query prefix kept in payload
    source_tokens: int = 96                 # URL JSON prefix kept in payload

    # ---- reset budget (Spec 5.10) ------------------------------------------
    l_overhead_max: int = 2560              # system prompt + question + headers
    summary_floor_tokens: int = 4096        # L_sum >= 4096 guaranteed by construction

    # ---- supervision (Spec 5.4, 5.5, 6.5) ----------------------------------
    label_clip: float = 5.0
    max_label_roots: int = 24               # m_s = min(24, |C_s|)
    stop_entity_df_threshold: float = 0.30

    # ---- deterministic seeds ------------------------------------------------
    seed: int = 42

    # ---------------------------------------------------------------- derived
    @property
    def l_host(self) -> int:
        """Host window in agent tokens: L_host = MAX_CONTEXT*1024 - 1000."""
        return self.max_context_k * 1024 - self.reserved_window_tokens

    @property
    def trigger_threshold(self) -> float:
        """Host compaction trigger: Tok >= 0.9 * L_host."""
        return self.trigger_fraction * self.l_host

    @property
    def l_turn_max(self) -> int:
        """Max serialization increment of one tool turn: floor(0.1 * L_host)."""
        return int(math.floor(0.1 * self.l_host))

    @property
    def per_observation_cap(self) -> int:
        """Tok_A(rho(o_t)) <= 704 (Spec 5.10)."""
        return (
            self.evidence_payload_tokens
            + self.fixed_fields_tokens
            + self.query_tokens
            + self.source_tokens
        )

    @property
    def protected_budget_tokens(self) -> int:
        """Sum over P_s of payloads <= 8 * 704 = 5632."""
        return self.k_protect * self.per_observation_cap

    @property
    def l_reset(self) -> int:
        """L_reset = 1024 * ceil((L_overhead_max + 8*704 + 4096)/1024) = 12288."""
        raw = self.l_overhead_max + self.protected_budget_tokens + self.summary_floor_tokens
        return 1024 * int(math.ceil(raw / 1024))

    # ---------------------------------------------------------------- checks
    def validate(self) -> None:
        """Window-safety check (Spec 5.10). Raises on any violation before any
        rollout is started."""
        if self.l_reset + self.l_turn_max > self.trigger_threshold:
            raise ValueError(
                "CONFIGURATION_ERROR: window safety violated: "
                f"L_reset({self.l_reset}) + L_turn_max({self.l_turn_max}) > "
                f"0.9*L_host({self.trigger_threshold})"
            )
        if self.evidence_head_tokens + self.evidence_tail_tokens != self.evidence_payload_tokens:
            raise ValueError("CONFIGURATION_ERROR: head+tail must equal payload budget")
        if self.fixed_fields_tokens > 32:
            # Spec 5.10: fixed fields over 32 tokens is a configuration error.
            raise ValueError("CONFIGURATION_ERROR: fixed fields exceed 32 tokens")

    def summary_budget(self, protected_tokens: int, overhead_actual: int) -> int:
        """L_sum^A = L_reset - Tok_A(payloads) - L_overhead_actual (Spec 5.10)."""
        l_sum = self.l_reset - protected_tokens - overhead_actual
        if l_sum < self.summary_floor_tokens:
            raise ValueError(
                "CONFIGURATION_ERROR: protected blocks + overhead violate the "
                f"L_sum >= {self.summary_floor_tokens} lower bound (got {l_sum})"
            )
        return l_sum

    # ------------------------------------------------------------ manifest
    def manifest(self) -> dict:
        d = asdict(self)
        d.update(
            l_host=self.l_host,
            trigger_threshold=self.trigger_threshold,
            l_turn_max=self.l_turn_max,
            per_observation_cap=self.per_observation_cap,
            protected_budget_tokens=self.protected_budget_tokens,
            l_reset=self.l_reset,
            stub_literal=STUB_LITERAL,
        )
        return d

    def config_hash(self) -> str:
        """Stable hash used for run manifests and comparator isomorphism tests."""
        payload = json.dumps(self.manifest(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_CONFIG = DCKConfig()
