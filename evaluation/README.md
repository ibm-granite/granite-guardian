# Granite Guardian evaluation

Reproduce the benchmark numbers reported on the
[Granite Guardian](https://huggingface.co/collections/ibm-granite/granite-guardian)
model cards, for the whole family (3.0 → 4.1). The framework loads a Guardian
model with [vLLM](https://github.com/vllm-project/vllm), runs one or more
benchmarks, and writes per-dataset metrics plus an aggregate.

It covers four benchmarks:

| Benchmark | What it measures | Metric | Data |
|---|---|---|---|
| `ood_safety` | Harm / out-of-distribution safety (10 datasets) | F1 | requires setup |
| `groundedness_true` | RAG hallucination on the [TRUE](https://github.com/google-research/true) benchmark (9 datasets) | AUC / BAcc | requires setup |
| `groundedness_aggrefact` | RAG hallucination on [LM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) | balanced accuracy | turnkey (HF) |
| `function_calling` | Function-call hallucination on [FC-Reward-Bench](https://huggingface.co/datasets/ibm-research/fc-reward-bench) | balanced accuracy | turnkey (HF) |

"Turnkey" benchmarks download their data from the Hugging Face Hub and need no
setup. The other two are evaluated on curated subsets of public datasets that we
do not redistribute; see [`data/README.md`](data/README.md) to assemble them.

## Install

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU (vLLM). An 8B model fits on a single 80 GB GPU; the 2B/5B
variants need less.

## Quickstart

Run the two turnkey benchmarks on Granite Guardian 4.1, no data setup needed:

```bash
python run_eval.py \
    --model-path ibm-granite/granite-guardian-4.1-8b \
    --benchmarks groundedness_aggrefact function_calling
```

The model type (which prompt template and output parser to use) is inferred from
the model path. Results are written under `results/`.

Add `--think` to evaluate in reasoning mode (Granite Guardian 3.3 and 4.1 only):

```bash
python run_eval.py \
    --model-path ibm-granite/granite-guardian-4.1-8b \
    --benchmarks groundedness_aggrefact function_calling --think
```

Run everything (after preparing the harm and TRUE data, see below):

```bash
export GG_EVALS_DATA_ROOT=/path/to/your/eval_data
python run_eval.py --model-path ibm-granite/granite-guardian-4.1-8b
```

## Models

Eight released models map onto three prompt APIs, each handled by one module in
`models/`. The `--model-type` is auto-inferred from the path; pass it explicitly
to override.

| Model | HF id | `--model-type` |
|---|---|---|
| Granite Guardian 3.0 2B | `ibm-granite/granite-guardian-3.0-2b` | `granite-guardian-3` |
| Granite Guardian 3.0 8B | `ibm-granite/granite-guardian-3.0-8b` | `granite-guardian-3` |
| Granite Guardian 3.1 2B | `ibm-granite/granite-guardian-3.1-2b` | `granite-guardian-3` |
| Granite Guardian 3.1 8B | `ibm-granite/granite-guardian-3.1-8b` | `granite-guardian-3` |
| Granite Guardian 3.2 3B-A800M | `ibm-granite/granite-guardian-3.2-3b-a800m` | `granite-guardian-3` |
| Granite Guardian 3.2 5B | `ibm-granite/granite-guardian-3.2-5b` | `granite-guardian-3` |
| Granite Guardian 3.3 8B | `ibm-granite/granite-guardian-3.3-8b` | `granite-guardian-3.3` |
| Granite Guardian 4.1 8B | `ibm-granite/granite-guardian-4.1-8b` | `granite-guardian-4.1` |

The three modules differ only in how the criterion and inputs are encoded:

- **`granite-guardian-3`** (3.0-3.2): `risk_name` config, document in a
  `context` turn, bare `Yes`/`No`. Harm and groundedness only; no think mode.
- **`granite-guardian-3.3`**: `criteria_id` config, `documents=`/`available_tools=`,
  `<score>yes/no</score>`; supports `--think`.
- **`granite-guardian-4.1`**: explicit `<guardian>` block,
  `<think>...</think><score>yes/no</score>`; supports `--think`.

## Reproducing the card benchmarks

Harm is F1; RAG on TRUE is AUC for 3.0-3.2 and balanced accuracy for 3.3; RAG on
LM-AggreFact and function calling are balanced accuracy. Function calling and
LM-AggreFact arrived with 3.3; 4.1 reports RAG on LM-AggreFact rather than TRUE.

```bash
# 3.0-3.2 (harm + TRUE; requires GG_EVALS_DATA_ROOT)
python run_eval.py \
    --model-path ibm-granite/granite-guardian-3.0-8b \
    --benchmarks ood_safety groundedness_true

# 3.3 (all four; add --think for reasoning mode)
python run_eval.py \
    --model-path ibm-granite/granite-guardian-3.3-8b

# 4.1 (harm + both hallucination benchmarks)
python run_eval.py \
    --model-path ibm-granite/granite-guardian-4.1-8b \
    --benchmarks ood_safety groundedness_aggrefact function_calling
```

> The TRUE **balanced-accuracy** operating point is fixed at 0.5, which reads
> slightly below the 3.3 card (≈0.76 vs 0.777); the ranking is unaffected
> (TRUE AUC ≈0.873), so use AUC for a threshold-free comparison.

## How scoring works

Each row becomes a Guardian prompt whose output gives a **label** (`1` = risk,
`0` = safe) and a **probability of risk** (softmax over the `Yes`/`No` token
log-probabilities, as in the Granite Guardian cookbooks). F1 and balanced
accuracy use the label; AUC uses the probability.

Two benchmarks preprocess: TRUE splits long documents into sentence chunks and
aggregates per-claim scores to a per-document verdict; FC-Reward-Bench expands
each correct/incorrect call pair into two examples (chosen = safe, rejected =
unsafe).

## Output layout

```
results/
  <mode>_<model-type>_<model-id>/
    <benchmark>/
      results_<dataset>.json      # per-dataset metrics
      Aggregate.json              # mean across datasets
```

`<mode>` is `think` or `no_think`. Pass `--save-predictions` to also dump the raw
model outputs per row.

## Data parallelism (sharding)

`ood_safety` and `groundedness_aggrefact` can be sharded across GPUs. `--num-shards N`
spawns N single-GPU processes, then merges and cleans up automatically:

```bash
python run_eval.py \
    --model-path ibm-granite/granite-guardian-4.1-8b \
    --benchmarks ood_safety --num-shards 8
```

Output is identical to a single-process run. `--keep-shards` retains the
per-shard directories for debugging. Non-shardable benchmarks in the same run
execute sequentially.

## Bring your own ...

The framework is three independent registries: models, benchmarks, and data.
Each extends by dropping in one file; nothing else has to change.

### Bring your own guard

Any guardrail can be evaluated against these benchmarks, not just Granite
Guardian. Drop a `models/<name>.py` exporting two functions and a name:

```python
MODEL_NAME = "my-guard"   # becomes the --model-type key

def format_fn(sample, ds_config, tokenizer, **kwargs):
    # Build the prompt for one row. Granite-specific kwargs (criteria_id,
    # think) arrive via **kwargs; ignore the ones your guard doesn't use.
    return prompt_string

def parse_fn(output, tokenizer, nlogprobs):
    # Read the verdict -> (label: 1=risk/0=safe, prob: probability of risk)
    return label, prob
```

The module is auto-discovered on startup and offered as `--model-type my-guard`
(pass it explicitly, since the path-based auto-inference only knows Granite
names). For safe/unsafe-style verdicts like Llama Guard or ShieldGemma,
`models/_helpers.py` provides `softmax_safe_unsafe`; for `Yes`/`No` guards use
`softmax_yes_no`.

### Bring your own benchmark

Drop a `benchmarks/<name>.py` with a `run` function and register it:

```python
# benchmarks/my_bench.py
def run(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    llm, tokenizer, sp = ensure_model()   # lazy; loaded once, shared
    ...                                    # format -> generate -> score -> save
run.SUPPORTS_SHARDING = True               # optional: enables --num-shards
```

Add it to `BENCHMARKS` in `benchmarks/__init__.py` and it becomes a valid
`--benchmarks` choice. `benchmarks/_common.py` has the shared helpers
(`format_all`, `generate_and_parse`, `compute_metrics`, `save_aggregate`).

### Bring your own dataset

The two turnkey benchmarks load from the Hugging Face Hub; the other two read
locally-prepared datasets from `GG_EVALS_DATA_ROOT`. To add your own data, save
it in the Arrow format the loader expects and point the env var at it. See
[`data/README.md`](data/README.md) for the directory layout, per-example schema,
and label conventions.

## Notes

- The `--nlogprobs` default (20) is enough to recover the `Yes`/`No` tokens for
  the probability computation. AUC metrics require it to be > 0.
- Some Hugging Face datasets are gated; accept their terms on the Hub and set
  `HF_TOKEN` in your environment before running.
