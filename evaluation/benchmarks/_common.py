"""Shared utilities for all benchmarks: data loading, metrics, generation."""

import json
import os

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, auc,
)
from tqdm import tqdm

# Root directory for locally-prepared benchmark data (ood_safety, TRUE).
# There is no default - set GG_EVALS_DATA_ROOT to the directory you prepared
# following data/README.md. HF-native benchmarks (LM-AggreFact, FC-Reward-Bench)
# do not need this.
DATA_ROOT = os.environ.get("GG_EVALS_DATA_ROOT")

# Column-name fallbacks for auto-detection.
USER_COLS = ("prompt", "text", "Question", "instruction")
ASST_COLS = ("response", "Answer", "claim")
CTX_COLS = ("context", "document", "doc")


def require_data_root():
    """Return DATA_ROOT or raise a clear, actionable error if it is unset."""
    if not DATA_ROOT:
        raise RuntimeError(
            "This benchmark reads locally-prepared data, but GG_EVALS_DATA_ROOT "
            "is not set. Prepare the datasets as described in data/README.md and "
            "export GG_EVALS_DATA_ROOT=/path/to/your/data before running."
        )
    return DATA_ROOT


def resolve_col(sample, configured_col, fallbacks):
    """Return a column value, trying the configured name then the fallbacks."""
    if configured_col and configured_col in sample:
        return sample[configured_col]
    for alt in fallbacks:
        if alt in sample:
            return sample[alt]
    return None


def load_data(path, partition="test"):
    """Load a dataset from disk (Arrow), select a partition, normalize labels."""
    data = load_from_disk(path)
    if hasattr(data, "keys"):
        data = data[partition] if partition in data else data[next(iter(data))]

    for col in ("input_ids", "token_type_ids", "attention_mask"):
        if col in data.column_names:
            data = data.remove_columns([col])

    if "label" in data.column_names and "labels" not in data.column_names:
        data = data.rename_column("label", "labels")

    def _resolve(ex):
        lab = ex["labels"]
        if lab in ("safe", "unsafe"):
            ex["labels"] = {"unsafe": 1, "safe": 0}[lab]
        elif lab in ("1", "0"):
            ex["labels"] = int(lab)
        return ex

    return data.map(_resolve, keep_in_memory=True)


def format_all(data, guard_fmt, ds_config, tokenizer, **kwargs):
    """Format prompts for all samples."""
    return [guard_fmt(s, ds_config, tokenizer, **kwargs)
            for s in tqdm(data, desc="  Formatting")]


def generate_and_parse(prompts, llm, sp, guard_parse, tokenizer,
                       batch_size=512, nlogprobs=20):
    """Run vLLM generation + parse outputs. Returns (labels, probs, responses)."""
    labels, probs, responses = [], [], []
    for i in tqdm(range(0, len(prompts), batch_size), desc="  Generating"):
        outputs = llm.generate(prompts[i:i + batch_size], sp)
        for out in outputs:
            text = out.outputs[0].text.strip()
            responses.append(text)
            lab, prob = guard_parse(out, tokenizer, nlogprobs)
            labels.append(lab)
            probs.append(prob)
    return labels, probs, responses


def compute_metrics(y_pred, y_true, y_prob=None, num_failed=None):
    """Standard classification metrics (F1, AUC, balanced accuracy, etc.).

    Parse failures (NA predictions) are filled with the wrong label so they
    count against the model rather than being silently dropped.
    """
    metrics = {"Failed": num_failed if num_failed else int(y_pred.isna().sum())}

    valid = y_true.notna()
    y_true, y_pred = y_true[valid].copy(), y_pred[valid].copy()
    if y_prob is not None:
        y_prob = y_prob[valid].copy()

    na_preds = y_pred.isna()
    if na_preds.sum():
        y_pred[na_preds] = 1 - y_true[na_preds]
    if y_prob is not None:
        na_probs = y_prob.isna()
        if na_probs.sum():
            y_prob[na_probs] = np.random.uniform(0, 1, size=na_probs.sum())

    metrics["auc"] = metrics["ap"] = metrics["auprc"] = None
    if y_prob is not None and y_true.nunique() > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["ap"] = average_precision_score(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        metrics["auprc"] = auc(rec, prec)

    yt, yp = y_true.astype(int), y_pred.astype(int).clip(0, 1)
    metrics["accuracy"] = accuracy_score(yt, yp)
    metrics["balanced_accuracy"] = balanced_accuracy_score(yt, yp)
    metrics["recall"] = recall_score(yt, yp, zero_division=0) if yt.sum() else 0
    metrics["Precision"] = precision_score(yt, yp, zero_division=0)
    metrics["f1 Score"] = f1_score(yt, yp, zero_division=0)
    metrics["f1 Score Macro"] = f1_score(yt, yp, average="macro", zero_division=0)

    cm = confusion_matrix(yt, yp)
    if cm.size > 1:
        TN, FP, FN, TP = cm.ravel()
        metrics["fpr"] = FP / (FP + TN) if (FP + TN) else 0
        metrics["fnr"] = FN / (FN + TP) if (FN + TP) else 0
        metrics["avg_err_rate"] = (metrics["fpr"] + metrics["fnr"]) / 2.0
    else:
        metrics["fpr"] = metrics["fnr"] = metrics["avg_err_rate"] = 0
    return metrics


def save_metrics(metrics, out_base, ds_name):
    with open(os.path.join(out_base, f"results_{ds_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def save_aggregate(metrics_list, out_base, model_id, bench_name):
    """Write Aggregate.json as the mean of the per-dataset metrics.

    Reports both mean F1 (harm) and mean AUC (TRUE groundedness) so the right
    column can be read per model generation.
    """
    agg = {"model": model_id, "benchmark": bench_name}
    for key in ("auc", "ap", "auprc", "accuracy", "balanced_accuracy",
                "recall", "Precision", "f1 Score", "f1 Score Macro",
                "fpr", "fnr", "avg_err_rate"):
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        agg[key] = float(np.mean(vals)) if vals else None
    with open(os.path.join(out_base, "Aggregate.json"), "w") as f:
        json.dump(agg, f, indent=4)
    f1, bacc, auc_ = agg.get("f1 Score"), agg.get("balanced_accuracy"), agg.get("auc")
    print(f"\n  Aggregate -- F1={f1:.4f}  BAcc={bacc:.4f}"
          + (f"  AUC={auc_:.4f}" if auc_ is not None else ""))
