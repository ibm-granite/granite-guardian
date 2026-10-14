"""Requirement Check - LoRA / aLoRA adapters on Granite 4.1 base models.

Adapters from ``ibm-granite/granitelib-core-r1.0`` (``requirement-check/
granite-4.1-{3b,8b,30b}/{lora,alora}``). Judges whether an assistant response
satisfies user-specified requirements: the conversation is followed by a user
turn ``<requirements> {requirements}\\n{instruction}``, and the adapter answers
``{"score": "yes"}`` (satisfied) or ``{"score": "no"}``.

The prompt matches the model card's usage example
(https://huggingface.co/ibm-granite/granitelib-core-r1.0/blob/main/requirement-check/README.md).
The LoRA ``io.yaml`` writes ``<requirements>:`` with a colon, but that tokenizes
``>:`` as one token and breaks the aLoRA invocation sequence ``<requirements>``,
so the colon-free form is used for both adapter types. No think mode.

Only the requirement-checking benchmarks (``ifeval_multiconstraint``,
``infobench_gpt4``, ``infobench_human``) apply; each row carries a
``constraints`` field.
"""

from models._helpers import parse_json_score

MODEL_NAME = "requirement-check"
TASK = "requirement_check"
SUPPORTS_THINK = False
# The verdict is ~6 tokens; the cap only guards against an adapter that runs on.
MAX_TOKENS = 20

_INSTRUCTION = (
    "Please verify if the assistant's generation satisfies the user's "
    "requirements or not and reply with a binary label accordingly. "
    'Respond with a json {"score": "yes"} if the constraints are '
    'satisfied or respond with {"score": "no"} if the constraints are not '
    "satisfied."
)


def format_fn(sample, ds_config, tokenizer, **kwargs):
    """Format a (prompt, response, constraints) row as a requirement check."""
    messages = [
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["response"]},
        {"role": "user", "content": f"<requirements> {sample['constraints']}\n{_INSTRUCTION}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_fn(output, tokenizer, nlogprobs):
    """Parse ``{"score": "yes"/"no"}`` → 1 (satisfied) / 0; prob = P(yes)."""
    return parse_json_score(output, nlogprobs)
