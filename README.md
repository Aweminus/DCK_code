# DCK: Descendant Closure Knockout

🏅 **Introduction**

DCK is a retention-style context compaction method for long-horizon search agents. When the host triggers compaction, DCK ranks retrieval-layer observations by importance, preserves the Top-K critical observations verbatim in `<protected_evidence>` blocks, and lets the rest flow into the summary. The search agent, compaction trigger, and summarizer remain frozen throughout, making DCK plug-and-play with existing agents.

🔧 **Quick Start**

**Step 0 - Environment**

```bash
vllm / sglang
qwen-agent
torch + transformers
```

**Step 1 - Prepare models**

Inference requires three models: the search agent, the summarizer for the `visit` tool, and the compaction summary tool; their paths are configured at the top of `src/run_dck.sh`. The `dck` / `single` strategies additionally need a local HF copy of the same checkpoint, configured via `DCK_LOCAL_MODEL` / `DCK_HEAD` / `DCK_SCALER` / `DCK_STOP_TABLES` (the `single` strategy also requires `DCK_SINGLE_HEAD`). The summarizer follows the host choice: `resum` uses the external summary tool, while `fixed_interval` / `selfcompact` use the search agent itself for self-summary.

**Step 2 - Prepare datasets**

Save the official benchmarks as JSONL files (one `{"question": ..., "answer": ...}` per line) under `src/eval_data/`: `browsecomp_en.jsonl` (official 1,266 questions), `browsecomp_zh.jsonl` (all 289 BrowseComp-ZH questions), and `gaia.jsonl` (the 103-question text-only subset of the GAIA validation split; download from HF `gaia-benchmark/GAIA` after accepting the terms).

**Step 3 - Train the head offline**

```bash
cd src
python -m offline.prepare_train_questions --openseeker <pool.jsonl> \
    --eval_files eval_data/gaia.jsonl eval_data/browsecomp.jsonl eval_data/browsecomp_zh.jsonl \
    --out_dir data/dck_train
python -m offline.collect_trajectories --model <agent_path> \
    --questions data/dck_train/questions.jsonl --snapshot_dir data/dck_train/snapshots_7b
python -m offline.build_stop_tables --questions data/dck_train/questions.jsonl \
    --snapshot_dir data/dck_train/snapshots_7b --agent websailor7b --out data/dck_train/stop_tables_7b.json
python -m offline.materialize_labels --local_model <agent_path> \
    --questions data/dck_train/questions.jsonl --snapshot_dir data/dck_train/snapshots_7b \
    --out data/dck_train/labels_7b.jsonl --include_single
python -m offline.train_head --labels data/dck_train/labels_7b.jsonl \
    --hidden_dim 3584 --head_kind closure --out_dir checkpoints/closure_head_7b
```

**Step 4 - Preflight tests and inference**

```bash
cd src
python3 -m unittest discover tests -v        # preflight tests, no GPU required
bash run_dck.sh <model_path> browsecomp_en <output_path> resum dck
bash run_dck.sh <model_path> browsecomp_en <output_path> resum off   # baseline
```

**Step 5 - Evaluation**

Inference results are written to `<output_path>/<model_name>_sglang/<dataset>/iter*.jsonl`. Scoring uses an LLM judge (Qwen2.5-72B-Instruct by default; requires `DASHSCOPE_API_KEY`):

```bash
cd src && python3 evaluate.py --dataset browsecomp_en --input_folder <output_path>/<model_dir>
```

Once the scored `*_scored.jsonl` files are produced, use the scripts under `offline/` to reproduce the paper statistics (`<dataset_dir>` is the directory containing `iter*_scored.jsonl`):

```bash
python3 -m offline.conditional_efficiency --input_folder <dataset_dir> --out eff.json   # conditional efficiency
python3 -m offline.paired_stats --dataset browsecomp_en \
    --comparison dck_vs_resum:<a_dir>:<b_dir> --holm_family dck_vs_resum --out paired.json
python3 -m offline.aggregate_context_curves --arm react:<ctx.jsonl> --arm resum:<ctx.jsonl> --out curves.json
python3 -m offline.aggregate_compaction_slices --baseline_event_log <baseline_ev.jsonl> \
    --arm resum:<dataset_dir> --arm resum_dck:<dataset_dir> --out slices.json
python3 -m offline.aggregate_protection_stats --protection <dck_ev.jsonl> \
    --baseline <baseline_ev.jsonl> --out protection.json
python3 -m offline.head_diagnostics --labels data/dck_train/labels_7b.jsonl \
    --closure_head checkpoints/closure_head_7b --out head_diag.json
```
