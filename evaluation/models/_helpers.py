"""Shared helpers for Granite Guardian model modules."""

import json
import math

from benchmarks._common import resolve_col, USER_COLS, ASST_COLS, CTX_COLS


def softmax_yes_no(logprobs):
    """Softmax probability of the 'Yes' token from vLLM step logprobs.

    Aggregates the exp-logprobs of every candidate token that decodes to
    "yes"/"no" (case-insensitive) across generation steps, then returns the
    normalized probability mass on "yes" - i.e. P(risk detected).
    """
    yes_prob, no_prob = 1e-50, 1e-50
    for step in logprobs:
        for lp in step.values():
            tok = lp.decoded_token.strip().lower()
            if tok == "yes":
                yes_prob += math.exp(lp.logprob)
            elif tok == "no":
                no_prob += math.exp(lp.logprob)
    return yes_prob / (yes_prob + no_prob)


def softmax_safe_unsafe(logprobs):
    """Softmax probability of the 'unsafe' token from vLLM step logprobs.

    Same aggregation as :func:`softmax_yes_no` but over "safe"/"unsafe" tokens,
    for models that emit a safe/unsafe verdict instead of yes/no.
    """
    safe_prob, unsafe_prob = 1e-50, 1e-50
    for step in logprobs:
        for lp in step.values():
            tok = lp.decoded_token.strip().lower()
            if tok == "safe":
                safe_prob += math.exp(lp.logprob)
            elif tok == "unsafe":
                unsafe_prob += math.exp(lp.logprob)
    return unsafe_prob / (safe_prob + unsafe_prob)


def build_messages(sample, ds_config):
    """Build (messages, documents, tools) from a dataset row.

    - messages: chat turns (user, optional assistant) resolved from the
      configured / fallback column names.
    - documents: a single-doc list for groundedness benchmarks (or None).
    - tools: parsed tool/function definitions for function-calling (or None).
    """
    messages = []
    user_text = resolve_col(sample, ds_config.get("user_col"), USER_COLS)
    if user_text is not None:
        messages.append({"role": "user", "content": user_text})

    asst_text = resolve_col(sample, ds_config.get("assistant_col"), ASST_COLS)
    if asst_text is not None:
        if ds_config.get("tools_col"):
            try:
                asst_text = str(json.loads(asst_text))
            except (json.JSONDecodeError, TypeError):
                pass
        messages.append({"role": "assistant", "content": asst_text})

    docs = None
    ctx_text = resolve_col(sample, ds_config.get("context_col"), CTX_COLS)
    if ctx_text is not None:
        docs = [{"doc_id": "0", "text": ctx_text}]

    tools = None
    if ds_config.get("tools_col"):
        raw = sample.get(ds_config["tools_col"])
        if raw is not None:
            try:
                tools = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                tools = raw

    return messages, docs, tools
