"""Six preflight test classes required before any main experiment (Spec 7.1).

1. prefix invariance      - appending future observations never changes
                            existing horizons' lineage or labels
2. serialization roundtrip- one snapshot produces byte-identical factual and
                            knockout prompts; answer tokens are precisely
                            located by the LCP mask
3. budget conservation    - restart states re-checked with the agent
                            tokenizer never exceed L_reset
4. sort determinism       - ties, fewer than 8 candidates and stable ids all
                            give a unique output; non-finite scores fail
5. comparator isomorphism - 7B closure vs single share every config except
                            labels; head ownership/dims never mix
6. window safety          - any legal tool-turn append, host trigger and side
                            forward stay within L_host

Run:  python -m unittest discover tests -v     (no GPU / weights required)
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest

from dck.config import DCKConfig
from dck.event_log import EventLog, read_events
from dck.host import fixed_interval_trigger, resum_trigger_reached
from dck.labels import equidistant_roots, score_event
from dck.lineage import LineageState
from dck.observations import ObservationRegistry
from dck.protection import (
    build_payload,
    build_payloads,
    compose,
    select_protected,
)
from dck.selector import RandomSelector
from dck.serialization import (
    forced_target,
    knockout_messages,
    render_prompt,
)


class SplitTokenizer:
    """Deterministic test tokenizer: whitespace tokens, ChatML-style template."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts) if tokenize is False else None

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def make_registry_and_messages(n_obs: int = 3):
    """A small deterministic scenario with a chained lineage 1 -> 2 -> 3."""
    reg = ObservationRegistry("q1")
    obs = []
    texts = [
        "Alpha Corp earned 5.2 billion USD in 2023 up from 4.1 billion",
        "Beta Inc competes with Alpha Corp in the cloud segment",
        "Gamma LLC partners with Beta Inc on distribution",
    ]
    queries = ["Alpha Corp", "USD figures", "Beta Inc revenue"]
    for i in range(n_obs):
        o = reg.register(
            tool_type="search" if i % 2 == 0 else "visit",
            queries=[queries[i]], urls=[f"http://x/{i}"],
            raw_text=texts[i], visible_text=texts[i],
            token_length=len(texts[i].split()),
        )
        obs.append(o)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is Alpha Corp revenue?"},
    ] + [{"role": "user", "content": o.serialize()} for o in obs]
    return reg, obs, messages


def make_lineage(horizon: int = 3):
    lin = LineageState(question_entities={"alpha corp"}, stop_entities=set())
    lin.add_observation(1, {"alpha corp", "usd"}, {"alpha corp"})
    lin.add_observation(2, {"beta inc"}, {"usd"})
    lin.add_observation(3, {"gamma llc"}, {"beta inc"})
    return lin


class StubCountBackend:
    """Deterministic teacher-forcing stand-in: logprob = -1 - 0.5 * stubs."""

    def __init__(self):
        self.calls = 0

    def loglikelihood(self, messages, answer):
        self.calls += 1
        stubs = sum("[result withheld]" in (m.get("content") or "")
                    for m in messages)
        return -1.0 - 0.5 * stubs


# ---------------------------------------------------------------- 1
class TestPrefixInvariance(unittest.TestCase):
    """Appending future observations must not change existing lineage or
    labels (Spec 7.1 test 1)."""

    def test_lineage_prefix_invariance(self):
        lin = make_lineage(3)
        before = {
            t: (lin.first_entities[t], lin.descendants(t, 3))
            for t in (1, 2, 3)
        }
        # append two future observations
        lin.add_observation(4, {"delta ag"}, {"gamma llc"})
        lin.add_observation(5, {"epsilon co"}, {"delta ag"})
        for t, (firsts, desc) in before.items():
            self.assertEqual(lin.first_entities[t], firsts)
            self.assertEqual(lin.descendants(t, 3), desc)
        # horizon-s closures at the new horizon only grow by later nodes
        self.assertEqual(lin.descendants(1, 5), {2, 3, 4, 5})
        self.assertEqual(lin.descendants(1, 3), before[1][1])

    def test_labels_prefix_invariance(self):
        # the same event scored before and after future observations exist
        # (the future observations are registered but not addressable at the
        # old horizon and are absent from the old snapshot context)
        reg_a, obs_a, messages = make_registry_and_messages(3)
        lin = make_lineage(3)
        labels = score_event(StubCountBackend(), reg_a, lin, messages,
                             "5.2 billion", DCKConfig(), "q1", 42, 1)

        reg_b, _, _ = make_registry_and_messages(3)
        reg_b.register("search", ["future"], ["http://x/f1"], "future text 1",
                       "future text 1", token_length=3)
        reg_b.register("visit", ["future"], ["http://x/f2"], "future text 2",
                       "future text 2", token_length=3)
        reg_b.mark_addressable([o.obs_id for o in obs_a],
                               {o.obs_id: 2 + i for i, o in enumerate(obs_a)})
        lin_b = make_lineage(3)
        lin_b.add_observation(4, {"delta ag"}, {"gamma llc"})
        lin_b.add_observation(5, {"epsilon co"}, {"delta ag"})
        labels2 = score_event(StubCountBackend(), reg_b, lin_b, messages,
                              "5.2 billion", DCKConfig(), "q1", 42, 1)
        self.assertEqual(labels.to_dict(), labels2.to_dict())


# ---------------------------------------------------------------- 2
class TestSerializationRoundtrip(unittest.TestCase):
    """One snapshot yields byte-identical factual/knockout prompts and a
    precisely located answer-token mask (Spec 7.1 test 2)."""

    def setUp(self):
        self.reg, self.obs, self.messages = make_registry_and_messages(3)
        self.tok = SplitTokenizer()

    def test_factual_and_knockout_byte_identity(self):
        gamma = {self.obs[0].obs_id, self.obs[1].obs_id}
        ko1 = knockout_messages(self.messages, gamma)
        ko2 = knockout_messages(self.messages, gamma)
        self.assertEqual(ko1, ko2)
        # factual rendering is deterministic
        self.assertEqual(
            render_prompt(self.tok, self.messages),
            render_prompt(self.tok, self.messages),
        )
        # only Gamma blocks change; all other messages byte-identical
        gamma_messages = {2, 3}   # message indices of the stubbed blocks
        for idx, (orig, mod) in enumerate(zip(self.messages, ko1)):
            if idx in gamma_messages:
                self.assertIn("[result withheld]", mod["content"])
                self.assertNotIn("Alpha Corp", mod["content"])
            else:
                self.assertEqual(orig, mod)
        # knockout is stable: repeated knockout of an already-stubbed context
        ko_again = knockout_messages(ko1, gamma)
        self.assertEqual(ko_again, ko1)

    def test_answer_token_mask(self):
        answer = "5.2 billion"
        full_ids, prefix = forced_target(self.tok, self.messages, answer)
        prompt_ids = self.tok.encode(
            render_prompt(self.tok, self.messages, add_generation_prompt=True))
        self.assertGreater(prefix, 0)
        self.assertEqual(full_ids[:prefix], prompt_ids[:prefix])
        target_text = " ".join(full_ids[prefix:])
        self.assertIn("Final answer:", target_text)
        self.assertIn(answer, target_text)


# ---------------------------------------------------------------- 3
class TestBudgetConservation(unittest.TestCase):
    """Restart states re-checked with the agent tokenizer stay within
    L_reset (Spec 7.1 test 3, Spec 5.10)."""

    def setUp(self):
        self.cfg = DCKConfig()
        self.cfg.validate()
        self.tok = SplitTokenizer()
        self.reg, self.obs, self.messages = make_registry_and_messages(3)

    def test_derived_budgets(self):
        self.assertEqual(self.cfg.l_host, 32 * 1024 - 1000)
        self.assertEqual(self.cfg.l_turn_max, int(math.floor(0.1 * self.cfg.l_host)))
        self.assertEqual(self.cfg.per_observation_cap, 704)
        self.assertEqual(self.cfg.protected_budget_tokens, 5632)
        self.assertEqual(self.cfg.l_reset, 12288)
        # worst case summary budget still respects the floor
        self.assertEqual(
            self.cfg.summary_budget(self.cfg.protected_budget_tokens, 2560), 4096)

    def test_single_payload_cap(self):
        long_body = "word " * 5000   # forces head 384 + tail 128 truncation
        o = self.reg.register("visit", ["q" * 500], ["http://x/9"],
                              long_body, long_body, token_length=5000)
        block, n = build_payload(o, self.tok, self.cfg)
        self.assertLessEqual(n, self.cfg.per_observation_cap)

    def test_protected_budget_sum(self):
        texts = [f"observation number {i} with unique content {i} " * 60
                 for i in range(20)]
        reg = ObservationRegistry("q2")
        for i, t in enumerate(texts):
            reg.register("search", [f"query {i}"], [f"http://x/{i}"], t, t,
                         token_length=len(t.split()))
        _, scores = None, {o.obs_id: float(i) for i, o in
                           enumerate(reg.candidates())}
        _, p_s = select_protected(reg.candidates(), scores, self.cfg)
        self.assertEqual(len(p_s), self.cfg.k_protect)
        block, total = build_payloads(p_s, self.tok, self.cfg)
        self.assertLessEqual(total, self.cfg.protected_budget_tokens)
        # the composed reset context is within L_reset when summaries obey L_sum
        ctx = compose("s" * 500, "q" * 500, "summary " * 400, block)
        reset_tokens = len(self.tok.encode(
            self.tok.apply_chat_template(ctx, tokenize=False,
                                         add_generation_prompt=False)))
        summary_budget = self.cfg.l_reset - total - 512
        self.assertGreater(summary_budget, self.cfg.summary_floor_tokens)
        self.assertLessEqual(reset_tokens + summary_budget, self.cfg.l_reset + 512)

    def test_event_log_realized_budget_fields(self):
        # realized-budget instrumentation: restart = payload + summary + overhead
        with tempfile.TemporaryDirectory() as d:
            log = EventLog(os.path.join(d, "events.jsonl"))
            log.write_event(
                question_id="q1", rollout_seed=42, compaction_index=1,
                trigger_tokens=29000, features={}, scores={},
                selected_ids=["o1"], payload_tokens=400, summary_budget=7000,
                summary_tokens=6000, restart_tokens=8000,
                overhead_tokens=1600, working_context_after=8000)
            log.close()
            rec = next(r for r in read_events(log.path) if r["type"] == "event")
            self.assertEqual(rec["summary_tokens"], 6000)
            self.assertEqual(rec["restart_tokens"], 8000)
            self.assertEqual(rec["overhead_tokens"],
                             rec["restart_tokens"] - rec["payload_tokens"]
                             - rec["summary_tokens"])
            self.assertEqual(rec["working_context_after"], rec["restart_tokens"])


# ---------------------------------------------------------------- 4
class TestSortDeterminism(unittest.TestCase):
    """Ties, candidate counts below 8 and stable ids give unique outputs;
    non-finite scores are implementation failures (Spec 7.1 test 4)."""

    def setUp(self):
        self.cfg = DCKConfig()
        self.reg, self.obs, _ = make_registry_and_messages(3)

    def test_tie_breaks_by_time_then_id(self):
        scores = {o.obs_id: 1.0 for o in self.obs}
        b_s, p_s = select_protected(self.reg.candidates(), scores, self.cfg)
        self.assertEqual([o.obs_id for o in b_s], [o.obs_id for o in self.obs])
        self.assertEqual([o.obs_id for o in p_s], [o.obs_id for o in self.obs])

    def test_fewer_than_eight_candidates(self):
        b_s, p_s = select_protected(self.reg.candidates(),
                                    {o.obs_id: 1.0 for o in self.obs}, self.cfg)
        self.assertEqual(len(b_s), 3)
        self.assertEqual(len(p_s), 3)

    def test_stable_output(self):
        scores = {o.obs_id: float(-i) for i, o in enumerate(self.obs)}
        r1 = select_protected(self.reg.candidates(), scores, self.cfg)
        r2 = select_protected(self.reg.candidates(), scores, self.cfg)
        self.assertEqual([o.obs_id for o in r1[0]], [o.obs_id for o in r2[0]])
        self.assertEqual([o.obs_id for o in r1[1]], [o.obs_id for o in r2[1]])

    def test_non_finite_fails(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            scores = {self.obs[0].obs_id: bad,
                      self.obs[1].obs_id: 1.0,
                      self.obs[2].obs_id: 2.0}
            with self.assertRaises(ValueError):
                select_protected(self.reg.candidates(), scores, self.cfg)

    def test_random_selector_reproducible(self):
        # >8 candidates so the seed actually matters (k = 8 < 12)
        reg = ObservationRegistry("q9")
        obs = []
        for i in range(12):
            obs.append(reg.register("search", [f"query {i}"], [f"http://x/{i}"],
                                    f"text {i}", f"text {i}", token_length=2))
        scores = {o.obs_id: 0.0 for o in obs}
        s1 = RandomSelector("q9", 42).select(reg, scores, self.cfg, 1)
        s2 = RandomSelector("q9", 42).select(reg, scores, self.cfg, 1)
        self.assertEqual([o.obs_id for o in s1], [o.obs_id for o in s2])
        self.assertEqual(len(s1), self.cfg.k_protect)
        s3 = RandomSelector("q9", 42).select(reg, scores, self.cfg, 2)
        # a different compaction index is a different (deterministic) draw
        self.assertNotEqual([o.obs_id for o in s1], [o.obs_id for o in s3])

    def test_random_selector_distinct_rollout_seeds(self):
        # paired rollouts use rollout_seed = base*1000 + rollout_id; the
        # random arm's draws must differ across those iterations
        reg = ObservationRegistry("q10")
        obs = []
        for i in range(12):
            obs.append(reg.register("search", [f"query {i}"], [f"http://x/{i}"],
                                    f"text {i}", f"text {i}", token_length=2))
        scores = {o.obs_id: 0.0 for o in obs}
        s1 = RandomSelector("q10", 42001).select(reg, scores, self.cfg, 1)
        s2 = RandomSelector("q10", 42002).select(reg, scores, self.cfg, 1)
        self.assertEqual(len(s1), self.cfg.k_protect)
        self.assertNotEqual([o.obs_id for o in s1], [o.obs_id for o in s2])


# ---------------------------------------------------------------- 5
class TestComparatorIsomorphism(unittest.TestCase):
    """7B closure and single heads share every configuration except labels;
    head ownership, input dims and checkpoints never mix (Spec 7.1 test 5)."""

    def test_config_hash_shared_across_arms(self):
        closure_cfg = DCKConfig()
        single_cfg = DCKConfig()
        random_cfg = DCKConfig()
        self.assertEqual(closure_cfg.config_hash(), single_cfg.config_hash())
        self.assertEqual(closure_cfg.config_hash(), random_cfg.config_hash())
        mutated = DCKConfig(k_protect=4)
        self.assertNotEqual(closure_cfg.config_hash(), mutated.config_hash())

    def test_head_dims_and_parameter_count(self):
        from dck.head import ClosureHead
        h7b = ClosureHead(3584)
        self.assertEqual(sum(p.numel() for p in h7b.parameters()), 1_839_617)
        h3b = ClosureHead(2048)   # illustrative 3B hidden size
        # loading a 7B checkpoint into a 3B head must fail
        state = h7b.state_dict()
        with self.assertRaises(RuntimeError):
            h3b.load_state_dict(state)

    def test_single_and_closure_share_everything_but_labels(self):
        reg, obs, messages = make_registry_and_messages(3)
        lin = make_lineage(3)
        backend = StubCountBackend()
        cfg = DCKConfig()
        both = score_event(backend, reg, lin, messages, "5.2 billion",
                           cfg, "q1", 42, 1, include_single=True)
        only_closure = score_event(StubCountBackend(), reg, lin, messages,
                                   "5.2 billion", cfg, "q1", 42, 1,
                                   include_single=False)
        # same roots, same gamma sizes; only the single_* fields differ
        for a, b in zip(both.to_dict()["labels"],
                        only_closure.to_dict()["labels"]):
            self.assertEqual(a["obs_id"], b["obs_id"])
            self.assertEqual(a["gamma_size"], b["gamma_size"])
            self.assertEqual(a["closure_delta"], b["closure_delta"])
            self.assertIsNone(b["single_delta"])
            self.assertIsNotNone(a["single_delta"])

    def test_single_selector_shares_top8_capacity(self):
        # Single Knockout differs from DCK only in the supervision signal,
        # never in the Top-8 capacity or presentation (Spec 7.1)
        from dck.selector import DCKSelector, SingleSelector
        reg = ObservationRegistry("q11")
        obs = []
        for i in range(12):
            obs.append(reg.register("search", [f"query {i}"], [f"http://x/{i}"],
                                    f"text {i}", f"text {i}", token_length=2))
        scores = {o.obs_id: float(i) for i, o in enumerate(obs)}
        cfg = DCKConfig()
        dck_sel = DCKSelector(None, None, None).select(reg, scores, cfg)
        single_sel = SingleSelector(None, None, None).select(reg, scores, cfg)
        self.assertEqual([o.obs_id for o in dck_sel], [o.obs_id for o in single_sel])
        self.assertEqual(len(single_sel), cfg.k_protect)

    def test_stub_literal_is_global_constant(self):
        from dck.config import STUB_LITERAL
        reg, obs, _ = make_registry_and_messages(3)
        for o in obs:
            self.assertIn(STUB_LITERAL, o.stub())


# ---------------------------------------------------------------- 6
class TestWindowSafety(unittest.TestCase):
    """Any legal tool-turn append, host trigger and side forward stay within
    L_host (Spec 7.1 test 6, Spec 5.10)."""

    def setUp(self):
        self.cfg = DCKConfig()
        self.cfg.validate()

    def test_reset_plus_turn_below_trigger(self):
        self.assertLessEqual(self.cfg.l_reset + self.cfg.l_turn_max,
                             self.cfg.trigger_threshold)

    def test_trigger_plus_turn_below_host_window(self):
        # a turn appended right at the trigger still fits the host window
        worst = self.cfg.trigger_threshold + self.cfg.l_turn_max
        self.assertLessEqual(worst, self.cfg.l_host)

    def test_host_trigger_semantics(self):
        l_host = self.cfg.l_host
        self.assertFalse(resum_trigger_reached(28591, l_host, 1))
        self.assertTrue(resum_trigger_reached(28592, l_host, 1))
        # no compaction when no calls remain (forces a normal final answer)
        self.assertFalse(resum_trigger_reached(30000, l_host, 0))

    def test_fixed_interval_trigger_semantics(self):
        # restart every 10 completed tool calls while LLM calls remain
        for t in range(0, 10):
            self.assertFalse(fixed_interval_trigger(t, 10))
        self.assertTrue(fixed_interval_trigger(10, 10))
        self.assertTrue(fixed_interval_trigger(20, 10))
        # no restart when no calls remain
        self.assertFalse(fixed_interval_trigger(10, 0))

    def test_summary_floor_never_violated(self):
        # even at the worst case overhead the floor holds by construction
        self.assertEqual(
            self.cfg.summary_budget(self.cfg.protected_budget_tokens,
                                    self.cfg.l_overhead_max),
            self.cfg.summary_floor_tokens)
        with self.assertRaises(ValueError):
            self.cfg.summary_budget(self.cfg.protected_budget_tokens + 1024,
                                    self.cfg.l_overhead_max)

    def test_equidistant_roots_spec_formula(self):
        # idx_j = floor(j*(n-1)/(m-1)), endpoints included (Spec 6.5)
        for n in (1, 2, 5, 24, 25, 30, 100):
            roots = equidistant_roots(n, 24)
            m = min(24, n)
            self.assertEqual(len(roots), m)
            self.assertEqual(sorted(set(roots)), roots)
            self.assertEqual(roots[0], 1)
            self.assertEqual(roots[-1], n)
            if m > 1:
                self.assertEqual(roots,
                                 [int(j * (n - 1) // (m - 1)) + 1
                                  for j in range(m)])


# ---------------------------------------------------------------- 7
class _WordTok(SplitTokenizer):
    """Whitespace tokenizer with the tok_len budget helper."""

    def tok_len(self, text):
        return len(self.encode(text))


class TestCompactSummarizerSymmetry(unittest.TestCase):
    """_dck_compact threads last_summary into the host summarizer: the DCK arm
    switches to QUERY_SUMMARY_PROMPT_LAST from its second compaction on, like
    the baseline arm of the same host."""

    class _Harness:
        """Minimal stand-in self for calling DCKReactAgent._dck_compact."""

        def __init__(self):
            self.dck_mode = "random"
            self.rollout_seed = 7
            self.config = DCKConfig()
            self.system_message = "sys"
            self.agent_tok = _WordTok()
            self.backend = None
            self.seen_last_summaries = []

        def _summarize_for_host(self, question, recent_messages, last_summary):
            self.seen_last_summaries.append(last_summary)
            return "<summary>Alpha Corp earned 5.2 billion USD.</summary>"

    def test_last_summary_threaded_to_host_summarizer(self):
        try:
            import dck_agent  # pulls the serving stack; kept local to this test
        except ModuleNotFoundError as exc:
            self.skipTest(
                f"serving stack (qwen_agent etc.) not installed; run on the "
                f"serving machine: {exc}"
            )

        question = "What is Alpha Corp revenue?"
        harness = self._Harness()
        for compaction_index, last_summary in ((1, None), (2, "prior summary")):
            reg, _, messages = make_registry_and_messages(3)
            dck_agent.DCKReactAgent._dck_compact(
                harness, question, messages, reg, make_lineage(3),
                {}, compaction_index, None, "q1", last_summary)
        self.assertEqual(harness.seen_last_summaries, [None, "prior summary"])


if __name__ == "__main__":
    unittest.main()
