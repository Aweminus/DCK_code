"""Local model backend for labels and hidden states (Spec 5.4, 5.6).

The sglang serving endpoint cannot expose hidden states or per-token
log-probabilities, so teacher forcing and feature pooling run on a local
transformers copy of the *same frozen checkpoint* as the search agent. All
calls are deterministic (no sampling).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .serialization import Message, forced_target, render_prompt


class PoolSpanError(ValueError):
    """POOL_SPAN_ERROR: the requested character span was not located in the
    rendered prompt (offset-mapping miss)."""


class AgentTokenizer:
    """Tokenizer-only handle for budget accounting (Tok_A). Loads no weights,
    so the serving rollout can enforce budgets without a local model copy."""

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def encode(self, text: str, add_special_tokens: bool = False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def tok_len(self, text: str) -> int:
        return len(self.encode(text))

    def tok_len_batch(self, texts):
        enc = self.tokenizer(list(texts), add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]


class LocalBackend:
    """Deterministic loglikelihood + hidden-state pooling on one checkpoint."""

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: str = "bfloat16", max_length: Optional[int] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.hidden_dim = self.model.config.hidden_size

    # ------------------------------------------------------------ token count
    def tok_len(self, text: str) -> int:
        """Agent-token count of a string (used for all budget accounting)."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def tok_len_batch(self, texts: Sequence[str]) -> List[int]:
        enc = self.tokenizer(list(texts), add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    # ------------------------------------------------------------ loglikelihood
    @torch.no_grad()
    def loglikelihood(self, messages: Sequence[Message],
                      answer_text: str) -> float:
        """log P(answer | messages) under teacher forcing with longest-common-
        prefix masking (Spec 5.4 step 2)."""
        full_ids, prefix_len = forced_target(self.tokenizer, messages, answer_text)
        if self.max_length is not None and len(full_ids) > self.max_length:
            raise ValueError("SCORE_CONTEXT_OVERFLOW")
        input_ids = torch.tensor([full_ids], device=self.device)
        logits = self.model(input_ids).logits[0]          # [T, V]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        # Token at position i is predicted by logits[i-1].
        total = 0.0
        for i in range(prefix_len, len(full_ids)):
            total += logprobs[i - 1, full_ids[i]].item()
        return total

    # ------------------------------------------------------------ hidden states
    @torch.no_grad()
    def hidden_state(self, messages: Sequence[Message],
                     span_text: str) -> torch.Tensor:
        """Last-layer mean-pool over the character span of `span_text` in the
        rendered prompt, located via tokenizer offset mapping (Spec 5.6)."""
        prompt = render_prompt(self.tokenizer, messages, add_generation_prompt=True)
        char_start = prompt.find(span_text)
        if char_start < 0:
            raise PoolSpanError(
                f"POOL_SPAN_ERROR: span not found (len={len(span_text)})"
            )
        char_end = char_start + len(span_text)
        enc = self.tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=False,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        token_idx = [
            i for i, (s, e) in enumerate(offsets)
            if e > char_start and s < char_end and e > s
        ]
        if not token_idx:
            raise PoolSpanError("POOL_SPAN_ERROR: no tokens overlap the span")
        input_ids = enc["input_ids"].to(self.device)
        hidden = self.model(input_ids, output_hidden_states=True).hidden_states[-1][0]
        return hidden[token_idx].mean(dim=0)          # [d_h]
