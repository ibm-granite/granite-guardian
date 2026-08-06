"""Granite Guardian 3.0 / 3.1 / 3.2 — risk_name API.

These generations share one prompt API (distinct from 3.3 and 4.1):
  - risk is selected via ``guardian_config={"risk_name": ...}``;
  - groundedness passes the document in a ``{"role": "context"}`` turn;
  - function calling passes the tool definitions in a ``{"role": "tools"}`` turn;
  - the model emits a bare ``Yes``/``No`` verdict (3.2 additionally appends a
    ``<confidence> ... </confidence>`` tag, so we read the first word).

The probability of risk is the softmax over the aggregated Yes/No token
logprobs — identical to the published cookbook ``get_probabilities``.

Covers: granite-guardian-3.0-2b/8b, 3.1-2b/8b, 3.2-3b-a800m, 3.2-5b.
Reference cookbooks: cookbooks/granite-guardian-3.{0,1,2}/quick_start_vllm.ipynb
"""

import json
import re

from models._helpers import softmax_yes_no

MODEL_NAME = "granite-guardian-3"


def format_fn(sample, ds_config, tokenizer, *, think=False, criteria_id="harm"):
    """Format a sample using the GG 3.0/3.1/3.2 chat template.

    ``criteria_id`` maps 1:1 to the model's ``risk_name`` (``harm`` /
    ``groundedness`` / ``function_call``). ``think`` is accepted for a uniform
    benchmark interface but ignored — these generations have no think mode.
    """
    from benchmarks._common import resolve_col, USER_COLS, ASST_COLS, CTX_COLS

    messages = []
    if criteria_id == "groundedness":
        ctx = resolve_col(sample, ds_config.get("context_col"), CTX_COLS)
        if ctx is not None:
            messages.append({"role": "context", "content": ctx})
        asst = resolve_col(sample, ds_config.get("assistant_col"), ASST_COLS)
        if asst is not None:
            messages.append({"role": "assistant", "content": asst})
    elif criteria_id == "function_call":
        tools = sample.get(ds_config["tools_col"]) if ds_config.get("tools_col") else None
        if tools is not None and not isinstance(tools, str):
            tools = json.dumps(tools, indent=2)
        if tools is not None:
            messages.append({"role": "tools", "content": tools})
        user = resolve_col(sample, ds_config.get("user_col"), USER_COLS)
        if user is not None:
            messages.append({"role": "user", "content": user})
        asst = resolve_col(sample, ds_config.get("assistant_col"), ASST_COLS)
        if asst is not None:
            if not isinstance(asst, str):
                asst = json.dumps(asst, indent=2)
            messages.append({"role": "assistant", "content": asst})
    else:  # harm and other single- or paired-turn risks
        user = resolve_col(sample, ds_config.get("user_col"), USER_COLS)
        if user is not None:
            messages.append({"role": "user", "content": user})
        asst = resolve_col(sample, ds_config.get("assistant_col"), ASST_COLS)
        if asst is not None:
            messages.append({"role": "assistant", "content": asst})

    return tokenizer.apply_chat_template(
        messages,
        guardian_config={"risk_name": criteria_id},
        tokenize=False, add_generation_prompt=True,
    )


def parse_fn(output, tokenizer, nlogprobs):
    """Parse a bare Yes/No verdict (first word) → 1/0; prob via Yes/No softmax."""
    text = output.outputs[0].text.strip()

    label = None
    match = re.search(r"\w+", text)
    if match:
        word = match.group(0).strip().lower()
        if word == "yes":
            label = 1
        elif word == "no":
            label = 0

    prob = None
    if nlogprobs > 0:
        logprobs = output.outputs[0].logprobs
        if logprobs:
            prob = softmax_yes_no(logprobs)

    return label, prob
