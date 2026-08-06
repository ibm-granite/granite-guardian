"""Granite Guardian 3.3 — criteria_id API with think / no-think.

Risk is selected via ``guardian_config={"criteria_id": ...}``; documents are
passed with the ``documents=`` kwarg and tools with ``available_tools=``.
Output format is ``<score> yes/no </score>`` (preceded by a ``<think>`` block
in think mode).

Reference cookbook: cookbooks/granite-guardian-3.3/*.ipynb
"""

import re

from models._helpers import build_messages, softmax_yes_no

MODEL_NAME = "granite-guardian-3.3"


def format_fn(sample, ds_config, tokenizer, *, think=False, criteria_id="harm"):
    """Format a sample using the GG 3.3 chat template."""
    messages, docs, tools = build_messages(sample, ds_config)
    return tokenizer.apply_chat_template(
        messages,
        guardian_config={"criteria_id": criteria_id},
        documents=docs, available_tools=tools,
        think=think, tokenize=False, add_generation_prompt=True,
    )


def parse_fn(output, tokenizer, nlogprobs):
    """Parse ``<score> yes/no </score>`` → 1/0; prob via Yes/No softmax."""
    text = output.outputs[0].text.strip()

    label = None
    match = re.findall(r"<score>\s*(.*?)\s*</score>", text, re.DOTALL)
    if match:
        score = match[0].strip().lower()
        if "yes" in score:
            label = 1
        elif "no" in score:
            label = 0

    prob = None
    if nlogprobs > 0:
        logprobs = output.outputs[0].logprobs
        if logprobs:
            prob = softmax_yes_no(logprobs)

    return label, prob
