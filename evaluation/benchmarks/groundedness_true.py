"""TRUE benchmark - groundedness with document chunking.

Nine faithfulness datasets from the TRUE benchmark. Older Granite Guardian
cards (3.0-3.2) report RAG groundedness as the mean **AUC** across these
datasets; 3.3 reports mean balanced accuracy. Both are written to
Aggregate.json - read the column for your model generation.

See data/README.md for how to obtain the TRUE datasets and lay them out
under GG_EVALS_DATA_ROOT.
"""

import os
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    if not nltk.download("punkt_tab", quiet=True):
        raise RuntimeError(
            "The NLTK 'punkt_tab' tokenizer is required for the TRUE benchmark "
            "but is not installed and could not be downloaded (no network?). "
            "Install it once with: python -c \"import nltk; nltk.download('punkt_tab')\""
        )

from benchmarks._common import (
    require_data_root, load_data, format_all, generate_and_parse,
    compute_metrics, save_metrics, save_aggregate,
    CTX_COLS, ASST_COLS,
)

CRITERIA_ID = "groundedness"
PARTITION = "train"
# Upper bound on document-chunk size (tokens). Capped at runtime to the model's
# context window minus HEADROOM (for the prompt template, the claim, and the
# generated verdict), so the same code works for the 8K 3.0-3.2 family and the
# 128K 3.3 / 4.1 models.
MAX_CHUNK_SIZE = 32000
HEADROOM = 2048

DATASETS = {
    "TRUE_bench_begin":      "begin",
    "TRUE_bench_dialfact":   "dialfact",
    "TRUE_bench_frank":      "frank",
    "TRUE_bench_mnbm":       "mnbm",
    "TRUE_bench_paws":       "paws",
    "TRUE_bench_q2":         "q2",
    "TRUE_bench_qags_cnndm": "qags_cnndm",
    "TRUE_bench_qags_xsum":  "qags_xsum",
    "TRUE_bench_summeval":   "summeval",
}

DS_CONFIG = {"assistant_col": "response", "context_col": "context"}


def _sent_tokenize_with_newlines(text):
    blocks = text.split("\n")
    result = []
    for block in blocks:
        result.extend(sent_tokenize(block))
        result.append("\n")
    return result[:-1]


def _chunk_doc(sentences, chunk_size, tokenizer):
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        n_tokens = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        if current_len + n_tokens > chunk_size and current:
            text = " ".join(current).replace(" \n ", "\n").strip()
            chunks.append(text[:chunk_size - 100] if current_len > chunk_size else text)
            current, current_len = [sentence], n_tokens
        else:
            current.append(sentence)
            current_len += n_tokens
    if current:
        text = " ".join(current).replace(" \n ", "\n").strip()
        chunks.append(text[:chunk_size - 100] if current_len > chunk_size else text)
    return chunks


def chunk_for_groundedness(dataset, chunk_size, tokenizer):
    cols = dataset.column_names
    doc_key = next((k for k in CTX_COLS if k in cols), None)
    claim_key = next((k for k in ASST_COLS if k in cols), None)
    chunked = []
    for idx, row in enumerate(tqdm(dataset, desc="Chunking")):
        doc_chunks = [c for c in _chunk_doc(
            _sent_tokenize_with_newlines(row[doc_key]), chunk_size, tokenizer) if c]
        claim_sentences = sent_tokenize(row[claim_key])
        for chunk_id, doc_chunk in enumerate(doc_chunks):
            for claim_id, claim in enumerate(claim_sentences):
                chunked.append({"context": doc_chunk, "response": claim,
                                "chunk_id": chunk_id, "claim_id": claim_id,
                                "labels": row["labels"], "doc_id": idx})
    return Dataset.from_list(chunked)


def group_preds_by_doc_ids(df, threshold=0.5):
    num_failed = int(df["pred_labels"].isna().sum())
    df = df[df["pred_labels"].notna()]
    col = "positive_label_probability"
    claim_probs = df.groupby(["doc_id", "claim_id"])[[col]].agg("min")
    doc_probs = claim_probs.groupby("doc_id")[col].agg("max")
    y_true = df.groupby("doc_id")["true_labels"].agg(lambda x: x.mode().iloc[0])
    result = pd.DataFrame({
        "preds": [1 if p > threshold else 0 for p in doc_probs],
        "y_probs": doc_probs.values, "label": y_true.values})
    return result, num_failed


def _model_context_len(llm, tokenizer):
    """Best-effort read of the model's usable context length (in tokens)."""
    for obj, attr in ((getattr(llm, "model_config", None), "max_model_len"),
                      (tokenizer, "model_max_length")):
        val = getattr(obj, attr, None)
        # tokenizers sometimes report a sentinel like 1e30 when unset
        if isinstance(val, int) and 0 < val < 10_000_000:
            return val
    return None


def run(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    data_root = require_data_root()
    llm, tokenizer, sp = ensure_model()
    fmt_kw = {"think": args.think, "criteria_id": CRITERIA_ID}
    all_metrics = []

    # Cap the chunk size at the model's context window (minus headroom for the
    # template, claim, and generated verdict). The 3.0-3.2 family is 8K.
    ctx = _model_context_len(llm, tokenizer)
    chunk_size = MAX_CHUNK_SIZE
    if ctx:
        chunk_size = min(MAX_CHUNK_SIZE, max(512, ctx - HEADROOM))
    print(f"  Chunk size: {chunk_size} tokens (model context: {ctx})")

    for ds_name, folder in DATASETS.items():
        print(f"\n--- {ds_name} ---")
        data = load_data(os.path.join(data_root, folder), partition=PARTITION)
        print(f"  {len(data)} samples")

        data = chunk_for_groundedness(data, chunk_size, tokenizer)
        print(f"  Chunked -> {len(data)} chunk-claim pairs")

        prompts = format_all(data, guard_fmt, DS_CONFIG, tokenizer, **fmt_kw)
        labels, probs, resp = generate_and_parse(
            prompts, llm, sp, guard_parse, tokenizer,
            batch_size=args.batch_size, nlogprobs=args.nlogprobs)

        df = data.to_pandas().rename(columns={"labels": "true_labels"})
        df["pred_labels"] = labels
        df["positive_label_probability"] = probs

        cons, num_failed = group_preds_by_doc_ids(df)
        print(f"  Aggregated -> {len(cons)} docs (failed: {num_failed})")

        metrics = compute_metrics(
            pd.Series(cons["preds"]), pd.Series(cons["label"]),
            pd.Series(cons["y_probs"]), num_failed)
        metrics.update(model_name=args.base_model, dataset=ds_name, benchmark="groundedness_true")
        print(f"  AUC={metrics['auc']}  BAcc={metrics['balanced_accuracy']:.4f}")

        save_metrics(metrics, out_base, ds_name)
        all_metrics.append(metrics)

    if all_metrics:
        save_aggregate(all_metrics, out_base,
                       f"{'think' if args.think else 'no_think'}_{args.base_model}", "groundedness_true")
