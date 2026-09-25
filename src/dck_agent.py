"""DCK agent loop: host paradigms with protected-compaction mounting (Spec 5.8, 5.10, 7.1).

Hosts (frozen, identical across arms):
  - react:          no compaction (baseline host)
  - resum:          periodic summarization host, trigger Tok >= 0.9 * L_host
  - fixed_interval: restart every 10 completed tool calls (self-summary)
  - recent_history: trim to the most recent 22k tokens
  - selfcompact:    SelfCompact-WS7B adapter (Spec 5.8.2)

DCK arms (dck_mode): off (host baseline) | dck | single | random. All arms of
one host share the same trigger code (dck.host) and the same tool wrapper; the
summarizer is a host property (resum uses the external summary tool,
fixed_interval / selfcompact use the agent's own served model) and is shared
verbatim by all arms of that host.

Control order per round (Spec 7.1):
  generate tool call -> normalize tool return by L_turn_max -> append +
  length assert -> extract hidden states on the safe pre-compaction context ->
  compute host trigger -> DCK scoring + compaction.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import tiktoken
from qwen_agent.utils.utils import build_text_completion_prompt
from transformers import AutoTokenizer

from prompt import QUERY_SUMMARY_PROMPT, QUERY_SUMMARY_PROMPT_LAST
from react_agent import MultiTurnReactAgent
from summary_utils import summarize_conversation

from dck.config import DCKConfig, DEFAULT_CONFIG
from dck.entities import EntityExtractor, StopTables
from dck.event_log import EventLog
from dck.features import FeatureScaler
from dck.head import load_head
from dck.host import (
    SelfCompactParams,
    fixed_interval_trigger,
    resum_trigger_reached,
    recent_history_threshold_tokens,
)
from dck.lineage import LineageState
from dck.model_backend import AgentTokenizer, LocalBackend
from dck.observations import ObservationRegistry
from dck.protection import (
    PROTECTED_HEADER,
    SUMMARY_HEADER,
    build_payload,
    build_payloads,
    compose,
    replace_protected_with_ids,
    truncate_summary,
)
from dck.selector import DCKSelector, RandomSelector, SingleSelector

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 60))
MAX_CONTEXT = int(os.getenv('MAX_CONTEXT', 32))


class DCKReactAgent(MultiTurnReactAgent):
    """Search agent with DCK protected compaction mounted on a frozen host."""

    def __init__(self,
                 paradigm: str = 'resum',
                 dck_mode: str = 'off',
                 rollout_seed: int = 42,
                 dck_config: Optional[DCKConfig] = None,
                 local_model_path: Optional[str] = None,
                 head_path: Optional[str] = None,
                 single_head_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 stop_tables: Optional[StopTables] = None,
                 event_log_path: Optional[str] = None,
                 context_log_path: Optional[str] = None,
                 snapshot_dir: Optional[str] = None,
                 backend_device: str = "cuda",
                 backend_dtype: str = "bfloat16",
                 recent_keep_tokens: int = 22 * 1024,
                 **kwargs):
        super().__init__(**kwargs)
        if paradigm not in {'react', 'resum', 'fixed_interval',
                            'recent_history', 'selfcompact'}:
            raise ValueError(f"CONFIGURATION_ERROR: unknown paradigm {paradigm}")
        if dck_mode not in {'off', 'dck', 'single', 'random'}:
            raise ValueError(f"CONFIGURATION_ERROR: unknown dck_mode {dck_mode}")
        self.paradigm = paradigm
        self.dck_mode = dck_mode
        self.rollout_seed = rollout_seed
        self.config = dck_config or DEFAULT_CONFIG
        self.config.validate()
        self.recent_keep_tokens = recent_keep_tokens

        # agent tokenizer for budget accounting (Tok_A)
        self.agent_tok = AgentTokenizer(self.llm_local_path)

        # DCK machinery (only needed when protection is on). The single arm
        # loads the single-knockout head; the closure arm the closure head
        # (comparator isomorphism: only the supervision differs, Spec 7.1).
        self.selector = None
        self.backend = None
        self.head = None
        self.scaler = None
        self.head_path = head_path
        if dck_mode == 'dck':
            if not (local_model_path and head_path and scaler_path):
                raise ValueError(
                    "CONFIGURATION_ERROR: dck mode needs --dck_local_model, "
                    "--dck_head and --dck_scaler"
                )
            self.backend = LocalBackend(local_model_path, device=backend_device,
                                        dtype=backend_dtype)
            self.head, _ = load_head(head_path)
            self.scaler = FeatureScaler.load(scaler_path)
            self.selector = DCKSelector(self.backend, self.head, self.scaler)
        elif dck_mode == 'single':
            if not (local_model_path and single_head_path and scaler_path):
                raise ValueError(
                    "CONFIGURATION_ERROR: single mode needs --dck_local_model, "
                    "--dck_single_head and --dck_scaler"
                )
            self.backend = LocalBackend(local_model_path, device=backend_device,
                                        dtype=backend_dtype)
            self.head, _ = load_head(single_head_path)
            self.scaler = FeatureScaler.load(scaler_path)
            self.selector = SingleSelector(self.backend, self.head, self.scaler)
            self.head_path = single_head_path   # manifest: head actually used
        elif dck_mode == 'random':
            self.selector = 'random'   # per-question instantiation, see _run

        self.extractor = EntityExtractor()
        self.stop_tables = stop_tables
        self.event_log_path = event_log_path
        self.context_log_path = context_log_path
        self.snapshot_dir = snapshot_dir

    # ------------------------------------------------------------- utilities
    def _tok_count(self, messages) -> int:
        """Host token count, same convention as the baseline react agent."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
        except Exception:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        from qwen_agent.llm.schema import Message
        full_prompt = build_text_completion_prompt(
            [Message(**x) for x in messages], allow_special=True
        )
        return len(tokenizer.encode(full_prompt))

    def _normalize_tool_return(self, raw_result: str) -> Tuple[str, bool]:
        """Deterministic tail truncation to L_turn_max agent tokens (Spec 5.10).
        Returns (visible_text, truncated)."""
        ids = self.agent_tok.encode(raw_result)
        if len(ids) <= self.config.l_turn_max:
            return raw_result, False
        return self.agent_tok.decode(ids[: self.config.l_turn_max]), True

    @staticmethod
    def _parse_tool_call(content: str) -> Optional[Tuple[str, dict]]:
        if '<tool_call>' not in content or '</tool_call>' not in content:
            return None
        try:
            call = json.loads(content.split('<tool_call>')[1].split('</tool_call>')[0])
            return call.get('name', ''), call.get('arguments', {}) or {}
        except (json.JSONDecodeError, IndexError):
            return None

    @staticmethod
    def _call_fields(tool_name: str, args: dict) -> Tuple[List[str], List[str]]:
        """(queries, urls) of the producing call, in tool-return order."""
        if tool_name == 'search':
            queries = args.get('query') or []
            if isinstance(queries, str):
                queries = [queries]
            return list(queries), []
        if tool_name == 'visit':
            urls = args.get('url') or []
            if isinstance(urls, str):
                urls = [urls]
            return [args.get('goal', '') or ''], list(urls)
        return [], []

    # ------------------------------------------------------------ compaction
    def _dck_compact(self, question, messages, registry, lineage,
                     hidden_cache, compaction_index, event_log, question_id,
                     last_summary):
        """One protected compaction (Spec 5.9, 5.10). Returns new messages."""
        cfg = self.config
        candidates = registry.candidates()

        if self.dck_mode == 'random':
            selector = RandomSelector(question_id, self.rollout_seed)
            scores = {obs.obs_id: 0.0 for obs in candidates}
            p_s = selector.select(registry, scores, cfg, compaction_index)
        else:
            current_body = {
                obs.obs_id: (obs.visible_text
                             if obs.role == 'tool_response' else None)
                for obs in candidates
            }
            current_body = {k: v for k, v in current_body.items() if v is not None}
            scores = self.selector.score(registry, lineage, messages,
                                         hidden_cache=hidden_cache,
                                         current_body=current_body)
            p_s = self.selector.select(registry, scores, cfg, compaction_index)

        protected_block, payload_tokens = build_payloads(p_s, self.agent_tok, cfg)

        # summarizer input: protected observations keep position, body -> ID
        summarizer_input = replace_protected_with_ids(
            messages, [obs.obs_id for obs in p_s])

        # L_sum^A = L_reset - Tok_A(payloads) - L_overhead_actual
        overhead_actual = self.agent_tok.tok_len(
            self.system_message + question + SUMMARY_HEADER + PROTECTED_HEADER
        )
        l_sum = cfg.summary_budget(payload_tokens, overhead_actual)

        recent_messages = summarizer_input[2:]
        try:
            summary_response = self._summarize_for_host(
                question, recent_messages, last_summary)
        except Exception as exc:
            print(f"[Summary Tool] invocation failed: {exc}")
            summary_response = ""
        summary_response, summary_tokens = truncate_summary(
            self.agent_tok, summary_response, l_sum)

        new_messages = compose(self.system_message, question,
                               summary_response, protected_block)

        # cache hidden states of protected bodies in the new context
        if self.backend is not None:
            for obs in p_s:
                block, _ = build_payload(obs, self.agent_tok, cfg)
                try:
                    hidden_cache[obs.obs_id] = self.backend.hidden_state(
                        new_messages, block)
                except Exception as exc:
                    print(f"[DCK] hidden pooling failed for {obs.obs_id}: {exc}")

        block_message = {obs.obs_id: len(new_messages) - 1 for obs in p_s} \
            if protected_block else {}
        registry.compaction_transition([obs.obs_id for obs in p_s], block_message)

        if event_log is not None:
            from dck.selector import observation_features
            horizon = max((o.time_index for o in candidates), default=0)
            restart_tokens = self._tok_count(new_messages)
            event_log.write_event(
                question_id=question_id, rollout_seed=self.rollout_seed,
                compaction_index=compaction_index,
                trigger_tokens=self._tok_count(messages),
                features=observation_features(registry, lineage, horizon),
                scores=scores,
                selected_ids=[obs.obs_id for obs in p_s],
                payload_tokens=payload_tokens, summary_budget=l_sum,
                summary_tokens=summary_tokens,
                restart_tokens=restart_tokens,
                overhead_tokens=restart_tokens - payload_tokens - summary_tokens,
                working_context_after=restart_tokens,
            )
        return new_messages, summary_response

    # ------------------------------------------------------- host summarizers
    def _self_summary(self, question, recent_messages, last_summary):
        """WebSailor self-summary: the agent's own served model compresses the
        conversation (fixed_interval and selfcompact hosts)."""
        recent_history_str = "\n".join([str(msg) for msg in recent_messages])
        if not last_summary:
            prompt = QUERY_SUMMARY_PROMPT.replace(
                "{{{question}}}", question).replace(
                "{{{recent_history_messages}}}", recent_history_str)
        else:
            prompt = QUERY_SUMMARY_PROMPT_LAST.replace(
                "{{{question}}}", question).replace(
                "{{{recent_history_messages}}}", recent_history_str).replace(
                "{{{last_summary}}}", last_summary)
        content = self.call_server([{"role": "user", "content": prompt}])
        content = re.sub(r'<think>.*?</think>', '', content,
                         flags=re.DOTALL).strip()
        try:
            content = content.split("<summary>")[1].split("</summary>")[0]
        except IndexError:
            pass
        return "<summary>" + content + "</summary>" if content else ""

    def _summarize_for_host(self, question, recent_messages, last_summary):
        """Host summarizer, shared verbatim by all arms of the host: resum
        calls the external summary tool; fixed_interval / selfcompact use the
        agent's own served model."""
        if self.paradigm == 'resum':
            return summarize_conversation(question, recent_messages,
                                          last_summary)
        return self._self_summary(question, recent_messages, last_summary)

    def _host_baseline_compact(self, question, messages, last_summary,
                               event_log, question_id, compaction_index,
                               trigger_tokens):
        """Baseline host compaction (dck_mode == off): summarize with the host
        summarizer and reset the context to (system, question + summary)."""
        recent_messages = messages[2:].copy()
        try:
            summary_response = self._summarize_for_host(
                question, recent_messages, last_summary)
        except Exception as exc:
            print(f"[Summary Tool] host summarizer invocation failed: {exc}")
            summary_response = ""
        if summary_response:
            last_summary = summary_response
            new_observation = (
                "Question: " + question
                + "\nBelow is a summary of the previous conversation. This summary "
                  "condenses key information from earlier steps, so please consider "
                  "it carefully. Assess whether the summary provides enough "
                  "information to answer the question and use it as the basis for "
                  "further reasoning and information gathering to answer the "
                  "question.\n"
                + "Summary: " + summary_response + "\n"
            )
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": new_observation},
            ]
        if event_log is not None:
            summary_tokens = self.agent_tok.tok_len(summary_response)
            restart_tokens = self._tok_count(messages)
            event_log.write_event(
                question_id=question_id, rollout_seed=self.rollout_seed,
                compaction_index=compaction_index,
                trigger_tokens=trigger_tokens,
                features={}, scores={},
                selected_ids=[],
                payload_tokens=0, summary_budget=0,
                summary_tokens=summary_tokens,
                restart_tokens=restart_tokens,
                overhead_tokens=restart_tokens - summary_tokens,
                working_context_after=restart_tokens,
            )
        return messages, last_summary

    def _recent_history_trim(self, messages) -> List[dict]:
        """Trim to the most recent `recent_keep_tokens` host tokens, keeping the
        system prompt and question."""
        threshold = recent_history_threshold_tokens(
            self.config.l_host, self.recent_keep_tokens)
        while len(messages) > 2 and self._tok_count(messages) > threshold:
            messages = messages[:2] + messages[3:]
        return messages

    def _selfcompact_compress(self, round_idx: int, token_count: int,
                              summaries_so_far: int) -> bool:
        """SelfCompact-WS7B adapter decision (Spec 5.8.2): rubric probes from
        round 3 with a >=2-round interval may compress above 0.20*L_host; an
        unconditional backstop fires at 0.30*L_host; at most one summary per
        trajectory."""
        p = SelfCompactParams
        l_host = self.config.l_host
        if summaries_so_far >= p.max_summaries_per_trajectory:
            return False
        if token_count >= p.backstop_threshold(l_host):
            return True
        if round_idx < p.probe_start_round:
            return False
        if round_idx % p.probe_min_interval != (p.probe_start_round
                                                % p.probe_min_interval):
            return False
        return token_count >= p.allow_threshold(l_host)

    # -------------------------------------------------------------- snapshots
    def _write_snapshot(self, question, answer, question_id, messages,
                        registry, lineage, compaction_index, kind: str,
                        token_count: int) -> None:
        """Dump one training-data snapshot: pre-compaction context, the full
        prompt-external archive and the lineage state (Spec 6.2, 6.3)."""
        import os as _os
        _os.makedirs(self.snapshot_dir, exist_ok=True)
        snapshot = {
            "kind": kind,                      # "event" | "terminal"
            "question": question,
            "answer": answer,
            "question_id": str(question_id),
            "rollout_seed": self.rollout_seed,
            "compaction_index": compaction_index,
            "token_count": token_count,
            "messages": [dict(m) for m in messages],
            "archive": registry.archive_dump(),
            "lineage": {
                "question_entities": sorted(lineage.question_entities),
                "stop_entities": sorted(lineage.stop_entities),
                "obs_entities": {str(k): sorted(v) for k, v in lineage.obs_entities.items()},
                "first_entities": {str(k): sorted(v) for k, v in lineage.first_entities.items()},
                "query_entities": {str(k): sorted(v) for k, v in lineage.query_entities.items()},
            },
        }
        fname = f"{question_id}_r{self.rollout_seed}_{kind}{compaction_index}.json"
        with open(_os.path.join(self.snapshot_dir, fname), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)

    def _write_context_record(self, question_id, turn: int, token_count: int,
                              compaction_index: int, running: bool,
                              termination: Optional[str] = None) -> None:
        """One per-tool-turn working-context record (context curves): the
        ACTIVE context size after append + compaction + trim, plus a final
        running=False record marking the end of the trajectory."""
        rec = {
            "question_id": str(question_id),
            "rollout_seed": self.rollout_seed,
            "turn": turn,
            "token_count": token_count,
            "compaction_index": compaction_index,
            "running": running,
        }
        if termination is not None:
            rec["termination"] = termination
        os.makedirs(os.path.dirname(self.context_log_path) or ".", exist_ok=True)
        with open(self.context_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ loop
    def _run(self, data, model: str, summary_iteration: int, **kwargs):
        self.model = model
        question = data['item']['question']
        answer = data['item']['answer']
        question_id = data['item'].get('question_id', data['rollout_id'])

        cfg = self.config
        l_host = cfg.l_host
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question},
        ]
        registry = ObservationRegistry(str(question_id))
        stop_entities = set()
        if self.stop_tables is not None:
            from dck.textnorm import is_cjk_dominant
            lang = 'zh' if is_cjk_dominant(question) else 'en'
            stop_entities = self.stop_tables.for_question(
                os.path.basename(str(self.llm_local_path).rstrip('/')), lang)
        lineage = LineageState(
            question_entities=self.extractor.entity_set(question),
            stop_entities=stop_entities,
        )
        hidden_cache: Dict[str, "object"] = {}

        event_log = None
        if self.event_log_path:
            event_log = EventLog(self.event_log_path)
            event_log.write_manifest(
                cfg, self.paradigm, self.dck_mode, self.llm_local_path,
                self.head_path,
                self.stop_tables.digest() if self.stop_tables is not None else None,
                extra={"question_id": question_id,
                       "rollout_seed": self.rollout_seed,
                       "extractor_backend": self.extractor.backend},
            )

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        tool_calls = 0
        compaction_index = 0
        summaries_so_far = 0
        last_summary = None
        full_trajectory = messages.copy()
        prediction, termination = 'No answer found.', 'answer not found'

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages)
            print(f'round {round}: {content}')

            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})
            full_trajectory.append({"role": "assistant", "content": content.strip()})

            parsed = self._parse_tool_call(content)
            had_tool_call = parsed is not None
            if parsed is not None:
                tool_name, tool_args = parsed
                try:
                    result = self._call_tool(tool_name, tool_args)
                    print(f"Tool call {tool_name} invocation success with length {len(result)}")
                except Exception as e:
                    print(f"Tool call error: {e}")
                    result = ('Error: Tool call is not a valid JSON. Tool call must '
                              'contain a valid "name" and "arguments" field.')

                # 1) normalize by L_turn_max (deterministic tail truncation)
                visible_text, truncated = self._normalize_tool_return(result)
                if truncated:
                    sha = hashlib.sha256(result.encode('utf-8')).hexdigest()
                    print(f"[DCK] tool return truncated to L_turn_max; raw sha256={sha}")

                queries, urls = self._call_fields(tool_name, tool_args)
                obs = registry.register(
                    tool_type=tool_name, queries=queries, urls=urls,
                    raw_text=result, visible_text=visible_text,
                    token_length=self.agent_tok.tok_len(visible_text),
                )
                block = obs.serialize()
                tool_calls += 1

                # 2) append + length assert (window safety)
                messages.append({"role": "user", "content": block})
                full_trajectory.append({"role": "user", "content": block})
                token_count = self._tok_count(messages)
                if token_count > l_host:
                    raise ValueError(
                        f"WINDOW_SAFETY_ERROR: context {token_count} > L_host "
                        f"{l_host} after append"
                    )

                # 3) lineage + hidden states on the safe pre-compaction context
                query_text = "\n".join(queries)
                lineage.add_observation(
                    obs.time_index,
                    obs_entities=self.extractor.entity_set(visible_text),
                    query_entities=self.extractor.entity_set(query_text),
                )
                if self.backend is not None:
                    try:
                        hidden_cache[obs.obs_id] = self.backend.hidden_state(
                            messages, visible_text)
                    except Exception as exc:
                        print(f"[DCK] hidden pooling failed for {obs.obs_id}: {exc}")

            elif '<answer>' in content and '</answer>' in content:
                answer_content = content.split('<answer>')[1].split('</answer>')[0].strip()
                if len(answer_content):
                    termination = 'answer'
                    prediction = answer_content
                    break

            # 4) host trigger (shared code, Spec 5.8)
            token_count = self._tok_count(messages)
            print(f"round: {round}, token count: {token_count}")

            if self.paradigm == 'resum':
                trigger = resum_trigger_reached(
                    token_count, l_host, num_llm_calls_available)
            elif self.paradigm == 'selfcompact':
                trigger = self._selfcompact_compress(
                    round, token_count, summaries_so_far)
            elif self.paradigm == 'fixed_interval':
                trigger = fixed_interval_trigger(
                    tool_calls, num_llm_calls_available)
            else:   # react / recent_history: never restart
                trigger = False

            if trigger:
                if self.snapshot_dir:
                    self._write_snapshot(
                        question, answer, question_id, messages, registry,
                        lineage, compaction_index + 1, "event", token_count)
                compaction_index += 1
                if self.dck_mode in {'dck', 'single', 'random'}:
                    messages, last_summary = self._dck_compact(
                        question, messages, registry, lineage, hidden_cache,
                        compaction_index, event_log, question_id, last_summary)
                    summaries_so_far += 1
                elif self.paradigm in {'resum', 'selfcompact', 'fixed_interval'}:
                    messages, last_summary = self._host_baseline_compact(
                        question, messages, last_summary, event_log,
                        question_id, compaction_index, token_count)
                    summaries_so_far += 1
                # record the reset context in the full trajectory (same as the
                # baseline, which appends its post-summary observation)
                full_trajectory.extend(messages[2:])
                token_count = self._tok_count(messages)
                print(f"round {round}, token count after compaction: {token_count}")

            if self.paradigm == 'recent_history':
                trimmed = self._recent_history_trim(messages)
                if len(trimmed) != len(messages):
                    messages = trimmed
                    token_count = self._tok_count(messages)

            # 5) per-tool-turn working-context instrumentation (context curves)
            if self.context_log_path and had_tool_call:
                self._write_context_record(
                    question_id, tool_calls, token_count, compaction_index,
                    running=True)

            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            if token_count > l_host:
                # window hit: force a final answer, judged normally
                print(f"Token count exceeds limit: {token_count} > {l_host}")
                messages[-1]['content'] = (
                    "You have now reached the maximum context length you can "
                    "handle. You should stop invoking tools and, based on all the "
                    "information above, think again and provide what you consider "
                    "to be the most likely answer in the following format: "
                    "<think> your final thinking </think>\n <answer> your answer "
                    "</answer>"
                )
                content = self.call_server(messages)
                messages.append({"role": "assistant", "content": content.strip()})
                full_trajectory.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = content.split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = content
                    termination = 'format error: generate an answer as token limit reached'
                break

        if self.snapshot_dir and compaction_index == 0:
            # trajectories without any compaction event contribute their
            # terminal horizon (Spec 6.4.5)
            self._write_snapshot(
                question, answer, question_id, messages, registry, lineage,
                compaction_index + 1, "terminal", self._tok_count(messages))

        if event_log is not None:
            event_log.close()

        if termination == 'answer not found':
            # loop exited without an <answer>: re-derive from the last message
            if '<answer>' in messages[-1]['content']:
                prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                termination = 'answer'
            elif num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        if self.context_log_path:
            self._write_context_record(
                question_id, tool_calls, self._tok_count(messages),
                compaction_index, running=False, termination=termination)

        return {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "messages": full_trajectory,
            "prediction": prediction,
            "termination": termination,
            "dck_meta": {
                "paradigm": self.paradigm,
                "dck_mode": self.dck_mode,
                "rollout_seed": self.rollout_seed,
                "compactions": compaction_index,
                "config_hash": self.config.config_hash(),
            },
        }
