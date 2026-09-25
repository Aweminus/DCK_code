import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from react_agent import MultiTurnReactAgent
from prompt import SYSTEM_PROMPT
import tool_search  # noqa: F401  (registers the "search" tool)
import tool_visit  # noqa: F401  (registers the "visit" tool)
from dck_agent import DCKReactAgent
from dck.entities import StopTables

# The agent is constructed lazily inside each worker process: submitting a
# closure-captured agent to a ProcessPoolExecutor cannot work (the callable
# must be picklable), and the dck/single arms load a local CUDA model that
# must never be initialized in the parent of a forked pool.
_AGENT = None


def _get_agent(agent_cfg, legacy):
    global _AGENT
    if _AGENT is None:
        if legacy:
            _AGENT = MultiTurnReactAgent(**agent_cfg)
        else:
            _AGENT = DCKReactAgent(**agent_cfg)
    return _AGENT


def _run_task(task, agent_cfg, model, summary_iteration, legacy):
    agent = _get_agent(agent_cfg, legacy)
    if not legacy:
        # per-rollout seed: the Random arm's draws and the sampling requests
        # stay distinct across the paired rollout iterations
        agent.rollout_seed = task["rollout_seed"]
        agent.llm_generate_cfg["seed"] = task["rollout_seed"]
    return agent._run(task, model, summary_iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--summary_iteration", type=int, default=1000)
    parser.add_argument("--legacy", action="store_true",
                        help="run the original MultiTurnReactAgent unchanged "
                             "(regression reference; requires react + off)")
    # ---- DCK extensions (all arms share the tool wrapper and trigger code) ----
    parser.add_argument("--paradigm", type=str, default="react",
                        choices=["react", "resum", "fixed_interval",
                                 "recent_history", "selfcompact"],
                        help="host paradigm; every arm (ReAct included) runs "
                             "the shared tool wrapper, unlike --legacy")
    parser.add_argument("--dck_mode", type=str, default="off",
                        choices=["off", "dck", "single", "random"],
                        help="protection arm mounted on the host")
    parser.add_argument("--rollout_seed", type=int, default=42,
                        help="base seed; the effective seed per rollout is "
                             "base*1000 + rollout_id")
    parser.add_argument("--dck_local_model", type=str, default="",
                        help="local HF checkpoint for labels/hidden states (dck/single)")
    parser.add_argument("--dck_backend_device", type=str, default="cuda")
    parser.add_argument("--dck_backend_dtype", type=str, default="bfloat16")
    parser.add_argument("--dck_head", type=str, default="", help="closure head checkpoint dir")
    parser.add_argument("--dck_single_head", type=str, default="",
                        help="single-knockout head checkpoint dir (required for --dck_mode single)")
    parser.add_argument("--dck_scaler", type=str, default="", help="feature scaler json")
    parser.add_argument("--dck_stop_tables", type=str, default="", help="frozen stop tables json")
    parser.add_argument("--dck_event_log", type=str, default="", help="JSONL event log path")
    parser.add_argument("--dck_context_log", type=str, default="",
                        help="JSONL per-tool-turn working-context log path")
    parser.add_argument("--run_tag", type=str, default="", help="run tag recorded in output dir name")
    args = parser.parse_args()

    if args.legacy and (args.paradigm != "react" or args.dck_mode != "off"):
        parser.error("--legacy requires --paradigm react and --dck_mode off")

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count

    # Parse model name (the part after the last / in the path)
    model_name = os.path.basename(model.rstrip('/'))

    # Create output directory structure: output_base/model_name_sglang/dataset_name/
    # --legacy keeps the original output dir; every experiment arm (ReAct
    # included) runs the shared tool wrapper and gets a paradigm-tagged dir.
    run_tag = f"_{args.run_tag}" if args.run_tag else ""
    if args.legacy:
        model_dir = os.path.join(output_base, f"{model_name}_sglang")
    else:
        dck_tag = f"_{args.dck_mode}" if args.dck_mode != "off" else ""
        model_dir = os.path.join(
            output_base, f"{model_name}_sglang{run_tag}_{args.paradigm}{dck_tag}")
    dataset_dir = os.path.join(model_dir, args.dataset)
    
    # Create directories
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Model name: {model_name}")
    print(f"Dataset name: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Rollout count: {roll_out_count}")

    data_filepath = f"eval_data/{args.dataset}.jsonl"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)
    
    tasks_to_run = []
    for rollout_idx in range(1, roll_out_count + 1):
        output_file = os.path.join(dataset_dir, f"iter{rollout_idx}.jsonl")
        print(f"\nBegin rollout {rollout_idx}/{roll_out_count}")
        base_num = len(tasks_to_run)

        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # Check for successful completion based on absence of top-level error key
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass

        for item in items:
            question = item.get("question", "").strip()
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                tasks_to_run.append({
                    "item": item.copy(),
                    "rollout_id": rollout_idx,
                    "rollout_seed": args.rollout_seed * 1000 + rollout_idx,
                })
            else:
                print(f"Skipping already processed question: {question}")

        print(f"Total questions in input: {len(items)}")
        print(f"Already successfully processed: {len(processed_queries)}")
        print(f"Total tasks to run for rollout {rollout_idx}: {len(tasks_to_run) - base_num}")

    if not tasks_to_run:
        print(f"All rollouts completed, no need to execute")
    else:
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p
            },
            'model_type': 'qwen_dashscope'
        }

        if args.legacy:
            agent_cfg = {
                "llm": llm_cfg,
                "function_list": ["search", "visit"],
                "system_message": SYSTEM_PROMPT,
            }
        else:
            stop_tables = None
            if args.dck_stop_tables:
                stop_tables = StopTables.load(args.dck_stop_tables)
            agent_cfg = {
                "llm": llm_cfg,
                "function_list": ["search", "visit"],
                "system_message": SYSTEM_PROMPT,
                "paradigm": args.paradigm,
                "dck_mode": args.dck_mode,
                "rollout_seed": args.rollout_seed,   # overridden per task
                "local_model_path": args.dck_local_model or None,
                "head_path": args.dck_head or None,
                "single_head_path": args.dck_single_head or None,
                "scaler_path": args.dck_scaler or None,
                "stop_tables": stop_tables,
                "event_log_path": args.dck_event_log or None,
                "context_log_path": args.dck_context_log or None,
                "backend_device": args.dck_backend_device,
                "backend_dtype": args.dck_backend_dtype,
            }
            if args.dck_mode in ("dck", "single") and args.max_workers > 2:
                print(f"WARNING: dck_mode={args.dck_mode} loads one local model "
                      f"copy per worker process; max_workers={args.max_workers} "
                      "may exceed GPU memory. Consider --max_workers <= 2.")

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit tasks (top-level callable + plain-dict args: picklable)
            future_to_task = {
                executor.submit(
                    _run_task,
                    task,
                    agent_cfg,
                    model,
                    args.summary_iteration,
                    args.legacy,
                ): task
                for task in tasks_to_run
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc=f"Processing Rollout {rollout_idx}"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_id"]
                output_file = os.path.join(dataset_dir, f"iter{rollout_idx}.jsonl")
                try:
                    result = future.result(timeout=3600)
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    print(f'Task for question "{task_info["item"]["question"]}" (Rollout {task_info["rollout_id"]}) generated an exception: {exc}')
                    # Log error to the output file
                    error_result = {
                        "question": task_info["item"]["question"],
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_id": task_info["rollout_id"],
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "prediction": "[Failed]",
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")

                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

    print(f"\nAll {roll_out_count} rollouts completed!")
