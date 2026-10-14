"""InfoBench - requirement checking on decomposed instruction requirements.

InfoBench (Qin et al., 2024) decomposes each instruction into yes/no
requirement questions (e.g. "Is the generated text a list of places?"). Each
(instruction, response, question) triple becomes one row; the judge answers
whether the response satisfies the question. Two sets of reference labels:

- ``infobench_gpt4``: GPT-4-0314 annotations of all 500 instructions answered
  by 6 LLMs (~13.5k rows).
- ``infobench_human``: expert annotations of 50 instructions answered by 5
  LLMs (~1.1k rows).

Metrics are computed per answering LLM; the aggregate is their mean. The
annotation files come from the InfoBench release; see data/README.md to place
them under GG_EVALS_DATA_ROOT/infobench.
"""

import json
import os
import re

import pandas as pd

from benchmarks._common import (
    require_data_root, format_all, generate_and_parse,
    compute_metrics, save_metrics, save_aggregate,
)

HUMAN_CSVS = ("InFoBench Expert Annotation - Easy_subset.csv",
              "InFoBench Expert Annotation - Hard_subset.csv")
HUMAN_MODELS = ("gpt-3.5-turbo", "gpt-4", "claude-v1", "alpaca-7b", "vicuna-13b")
GPT4_DIR = os.path.join("gpt4_annotation", "0314")


def _row(instruction, inp, response, question, label):
    inp = (inp or "").strip()
    return {"prompt": f"{instruction}\n\n{inp}" if inp else instruction,
            "response": response, "constraints": question, "labels": label}


def _load_gpt4(root):
    """Answering LLM -> rows, from the GPT-4-0314 ``*_eval.json`` (JSON lines) files."""
    by_model = {}
    gpt4_dir = os.path.join(root, GPT4_DIR)
    for fname in sorted(os.listdir(gpt4_dir)):
        if not fname.endswith("_eval.json"):
            continue
        rows = []
        with open(os.path.join(gpt4_dir, fname)) as f:
            for line in f:
                e = json.loads(line)
                questions, evals = e["decomposed_questions"], e["eval"]
                if e.get("output") is None or len(questions) != len(evals):
                    continue
                rows += [_row(e["instruction"], e.get("input"), e["output"], q, int(ev))
                         for q, ev in zip(questions, evals) if ev is not None]
        by_model[fname[:-len("_eval.json")]] = rows
    return by_model


def _parse_questions(text):
    """Split the CSV's "1. ... (type) \\n2. ..." list, dropping numbers and type tags."""
    parts = re.split(r"\n\s*\d+\.", "\n" + str(text))
    return [re.sub(r"\s*\([^)]+\)\s*$", "", p.strip()) for p in parts if p.strip()]


def _load_human(root):
    """Answering LLM -> rows, from the expert-annotation CSVs (Easy + Hard subsets)."""
    df = pd.concat([pd.read_csv(os.path.join(root, f)) for f in HUMAN_CSVS], ignore_index=True)
    by_model = {m: [] for m in HUMAN_MODELS}
    for _, r in df.iterrows():
        questions = _parse_questions(r["decomposed_questions"])
        inp = r["input"] if pd.notna(r["input"]) else ""
        for model in HUMAN_MODELS:
            ann = r.get(f"{model}-annotation")
            if pd.isna(ann) or pd.isna(r.get(model)):
                continue
            # Dot-separated per-question bits, e.g. "1.0.1".
            bits = [int(b) for b in str(ann).split(".") if b in ("0", "1")]
            if len(bits) != len(questions):
                continue
            by_model[model] += [_row(r["instruction"], inp, str(r[model]), q, b)
                                for q, b in zip(questions, bits)]
    return by_model


def _run(by_model, ensure_model, args, guard_fmt, guard_parse, out_base, bench_name):
    llm, tokenizer, sp = ensure_model()
    print(f"\n  {bench_name}: {sum(map(len, by_model.values()))} rows "
          f"across {len(by_model)} answering models")
    all_metrics = []
    for model_name, rows in sorted(by_model.items()):
        print(f"\n--- {model_name} ({len(rows)} rows) ---")
        prompts = format_all(rows, guard_fmt, {}, tokenizer, think=args.think)
        labels, probs, resp = generate_and_parse(
            prompts, llm, sp, guard_parse, tokenizer,
            batch_size=args.batch_size, nlogprobs=args.nlogprobs)

        df = pd.DataFrame(rows).rename(columns={"labels": "true_labels"})
        df["pred_labels"] = labels
        df["positive_label_probability"] = probs

        metrics = compute_metrics(
            pd.Series(df["pred_labels"]), pd.Series(df["true_labels"]),
            pd.Series(df["positive_label_probability"]) if probs[0] is not None else None)
        metrics.update(model_name=args.model_path, dataset=model_name, benchmark=bench_name)
        print(f"  F1={metrics['f1 Score']:.4f}  BAcc={metrics['balanced_accuracy']:.4f}")

        save_metrics(metrics, out_base, model_name)
        if args.save_predictions:
            df["full_responses"] = resp
            df.to_json(f"{out_base}/predictions_{model_name}.jsonl", orient="records", lines=True)
        all_metrics.append(metrics)

    save_aggregate(all_metrics, out_base,
                   f"{'think' if args.think else 'no_think'}_{args.model_path}", bench_name)


def run_gpt4(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    root = os.path.join(require_data_root(), "infobench")
    _run(_load_gpt4(root), ensure_model, args, guard_fmt, guard_parse, out_base, "infobench_gpt4")


def run_human(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    root = os.path.join(require_data_root(), "infobench")
    _run(_load_human(root), ensure_model, args, guard_fmt, guard_parse, out_base, "infobench_human")


run_gpt4.TASK = run_human.TASK = "requirement_check"
