# Granite Guardian evaluation

Reproduce the benchmark numbers reported on the
[Granite Guardian](https://huggingface.co/collections/ibm-granite/granite-guardian)
model cards, for the whole family (3.0 → 4.1), and on the
[Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries)
cards for the Guardian Core and Requirement Check LoRA / aLoRA adapters. The
framework loads a model with [vLLM](https://github.com/vllm-project/vllm)
(aLoRA adapters with Hugging Face PEFT), runs one or more benchmarks, and writes
per-dataset metrics plus an aggregate.

It covers seven benchmarks. The first four are **guardian** benchmarks (risk
detection); the last three are **requirement-checking** benchmarks (does a
response satisfy given requirements?):

| Benchmark | What it measures | Metric | Data |
|---|---|---|---|
| `ood_safety` | Harm / out-of-distribution safety (10 datasets) | F1 | requires setup |
| `groundedness_true` | RAG hallucination on the [TRUE](https://github.com/google-research/true) benchmark (9 datasets) | AUC / BAcc | requires setup |
| `groundedness_aggrefact` | RAG hallucination on [LM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) | balanced accuracy | turnkey (HF) |
| `function_calling` | Function-call hallucination on [FC-Reward-Bench](https://huggingface.co/datasets/ibm-research/fc-reward-bench) | balanced accuracy | turnkey (HF) |
| `ifeval_multiconstraint` | Multi-constraint instruction following (IFEval-style, held-out set) | balanced accuracy | requires setup |
| `infobench_gpt4` | [InfoBench](https://arxiv.org/abs/2401.03601) decomposed requirements, GPT-4 labels | balanced accuracy | requires setup |
| `infobench_human` | InfoBench decomposed requirements, expert labels | balanced accuracy | requires setup |

"Turnkey" benchmarks download their data from the Hugging Face Hub and need no
setup. The others are evaluated on curated subsets of public datasets, or on
annotation files distributed outside the Hub, that we do not redistribute; see
[`data/README.md`](data/README.md) to assemble them.

## Install

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU (vLLM). An 8B model fits on a single 80 GB GPU; the 2B/5B
variants need less. We ran the 30B adapters on two 80 GB GPUs (`--ngpus 2`).

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

Eight released Guardian models map onto three prompt APIs, each handled by one
module in `models/`; the Granite Libraries adapters add two more (see below). The `--model-type` is auto-inferred from the path; pass it explicitly
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

The 4.1 and Guardian Core formatters pick the scoring schema from the
conversation: "If the last assistant's text meets the criteria..." when there is
an assistant turn, and "If the last user's text meets the criteria..." when there
is not (the prompt-only harm datasets). See
[Scoring schema for prompt-only harm](#scoring-schema-for-prompt-only-harm) for
how this relates to the published numbers.

### Granite Libraries adapters (LoRA / aLoRA)

The [Guardian Core](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/tree/main/guardian-core)
and [Requirement Check](https://huggingface.co/ibm-granite/granitelib-core-r1.0/tree/main/requirement-check)
adapters bring Guardian judging to the Granite 4.1 base models
(`ibm-granite/granite-4.1-{3b,8b,30b}`). Pass the adapter as `--model-path`,
either as `<hf-repo>/<subfolder>` (downloaded on first use) or as a local
directory holding `adapter_config.json`. The base model and adapter type are
read from `adapter_config.json`:

| Adapter | `--model-path` | `--model-type` | Benchmarks |
|---|---|---|---|
| Guardian Core | `ibm-granite/granitelib-guardian-r1.0/guardian-core/granite-4.1-<size>/<lora\|alora>` | `guardian-core` | `ood_safety`, `groundedness_aggrefact`, `function_calling` (any guardian benchmark) |
| Requirement Check | `ibm-granite/granitelib-core-r1.0/requirement-check/granite-4.1-<size>/<lora\|alora>` | `requirement-check` | `ifeval_multiconstraint`, `infobench_gpt4`, `infobench_human` |

- **`guardian-core`**: `<guardian>` judge block as the final user turn,
  `documents=`/`available_tools=`, `{"score": "yes"/"no"}` (yes = risk).
- **`requirement-check`**: a final `<requirements> ...` user turn,
  `{"score": "yes"/"no"}` (yes = requirements satisfied).

**LoRA** adapters run on vLLM (`LoRARequest`). **aLoRA** adapters run on Hugging
Face transformers + PEFT, because vLLM does not support aLoRA. The backend is
picked automatically. The HF path batches by padded length (`--hf-batch-tokens`,
default 32768; lower it if a long-context benchmark runs out of memory) and
returns the same step log-probabilities as vLLM, so AUC is reported for both.
Neither adapter family has a think mode, and generation is capped at 20 tokens
(the verdict is ~6). With no `--benchmarks`, a run covers every benchmark the
model type supports.

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

> The **3.3** LM-AggreFact card number (0.765) is the one reported by the official
> [MiniCheck](https://github.com/Liyan06/MiniCheck) project on the
> [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/), which runs 3.3 in
> think mode with a SentenceFusion protocol. Cite and run MiniCheck for that exact
> figure. This framework uses a single-pass groundedness path (the same one 4.1
> uses); on 3.3 it reproduces the card to within a point (≈0.758) and matches
> MiniCheck to <1 point on 10 of 11 datasets. The 4.1 LM-AggreFact card number is
> reproduced exactly by this path.

### Granite Libraries adapters

Each command runs every benchmark its model type supports (Guardian Core: harm,
LM-AggreFact, FC-Reward-Bench; Requirement Check: IFEval and both InfoBench
splits). Swap the size (`3b`, `8b`, `30b`) and type (`lora`, `alora`) in the
path for the other adapters:

```bash
export GG_EVALS_DATA_ROOT=/path/to/your/eval_data

# Guardian Core
python run_eval.py \
    --model-path ibm-granite/granitelib-guardian-r1.0/guardian-core/granite-4.1-8b/lora \
    --benchmarks ood_safety groundedness_aggrefact function_calling

# Requirement Check
python run_eval.py \
    --model-path ibm-granite/granitelib-core-r1.0/requirement-check/granite-4.1-8b/alora
```

Reproduced on all six 4.1 adapters of each family (H100, vLLM 0.31, PEFT 0.21):

**Guardian Core**: harm F1, LM-AggreFact mean per-dataset BAcc, FC-Reward-Bench
BAcc. The harm column is *this framework / card protocol / card*; the other two are
*reproduced / card*.

| Adapter | `ood_safety` F1 | `groundedness_aggrefact` BAcc | `function_calling` BAcc |
|---|---|---|---|
| 4.1-3B LoRA | 0.76 / 0.77 / 0.77 | 0.720 / 0.721 | 0.73 / 0.73 |
| 4.1-3B aLoRA | 0.78 / 0.78 / 0.78 | 0.744 / 0.745 | 0.75 / 0.75 |
| 4.1-8B LoRA | 0.80 / 0.79 / 0.79 | 0.727 / 0.728 | 0.78 / 0.78 |
| 4.1-8B aLoRA | 0.80 / 0.80 / 0.80 | 0.740 / 0.741 | 0.76 / 0.76 |
| 4.1-30B LoRA | 0.80 / 0.80 / 0.80 | 0.750 / 0.750 | 0.78 / 0.79 |
| 4.1-30B aLoRA | 0.81 / 0.81 / 0.81 | 0.757 / 0.757 | 0.79 / 0.78 |

**Requirement Check** (reproduced / card), balanced accuracy:

| Adapter | `ifeval_multiconstraint` | `infobench_gpt4` | `infobench_human` |
|---|---|---|---|
| 4.1-3B LoRA | 0.8147 / 0.8153 | 0.6970 / 0.6960 | 0.6808 / 0.6728 |
| 4.1-3B aLoRA | 0.8445 / 0.8425 | 0.7164 / 0.7150 | 0.6407 / 0.6475 |
| 4.1-8B LoRA | 0.8373 / 0.8360 | 0.7368 / 0.7373 | 0.6540 / 0.6546 |
| 4.1-8B aLoRA | 0.8514 / 0.8554 | 0.7124 / 0.7119 | 0.6794 / 0.6761 |
| 4.1-30B LoRA | 0.8420 / 0.8406 | 0.7500 / 0.7512 | 0.7048 / 0.7038 |
| 4.1-30B aLoRA | 0.8498 / 0.8492 | 0.7431 / 0.7432 | 0.7023 / 0.7052 |

Under the card protocol, Guardian Core matches the card to within one unit in its
last reported digit (per dataset, harm F1 and LM-AggreFact BAcc stay within
±0.01). Requirement Check
matches to within ±0.008 BAcc; the largest gaps are on the small
`infobench_human` set (~230 rows per answering model). LoRA results come from
vLLM and aLoRA results from the HF + PEFT path; both reproduce their card rows.

### Scoring schema for prompt-only harm

Five `ood_safety` datasets (AegisSafetyTest, HarmBench_Prompt, OAI_hf,
simpleSafetyTest, toxic_chat) are prompt-only: the conversation has no assistant
turn. This framework then asks the judge about "the last user's text", matching
the prompt-only example on the Guardian Core card.

**The Guardian Core card's harm numbers were produced with the assistant-text
schema ("If the last assistant's text meets the criteria...") on every row,
prompt-only rows included.** That is the "card protocol" column above: with the
assistant-text schema on every row, these adapters reproduce all six card
aggregates exactly at two decimals, and 24 of the 30 prompt-only per-dataset
cells. With the user-text schema they reproduce 4 of 6 aggregates and 12 of 30
cells.

The schema matters for prompt-only rows. Switching it moves per-dataset F1 by up
to ±0.02 (e.g. 4.1-8B LoRA: AegisSafetyTest 0.840 → 0.861, toxic_chat
0.771 → 0.761), with no consistent direction across adapters. Aggregates move by
at most 0.006. Prompts with an assistant turn are unchanged. Their scores can still
shift by up to ~0.008 between runs, because borderline rows (P(yes) ≈ 0.5) flip
under vLLM's batch-dependent numerics, and one flip on a few-hundred-row dataset
moves F1 by nearly that much.

For Granite Guardian 4.1 8B the aggregates barely move. No-think harm F1 is 0.794
under both schemas. Think mode is 0.779 (assistant-text) vs. 0.782 (user-text),
though think-mode toxic_chat moves from 0.705 to 0.753. Both modes still round to
the 4.1 card (0.79 / 0.78).

For an example comparison of these adapters against prompted Granite 4.1 models
and hosted decision models, see
[`result_files/decision-models-comparison.md`](result_files/decision-models-comparison.md).

## How scoring works

Each row becomes a Guardian prompt whose output gives a **label** (`1` = risk,
`0` = safe) and a **probability of risk** (softmax over the `Yes`/`No` token
log-probabilities, as in the Granite Guardian cookbooks). F1 and balanced
accuracy use the label; AUC uses the probability.

The requirement-checking benchmarks flip the polarity: `1` = the response
satisfies the requirement(s), and the probability is P(`yes`) = P(satisfied).

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
`--benchmarks` choice. Set `run.TASK = "requirement_check"` for a benchmark that
judges responses against requirements; it then runs only on model types whose
module sets the same `TASK` (e.g. `requirement-check`). `benchmarks/_common.py` has the shared helpers
(`format_all`, `generate_and_parse`, `compute_metrics`, `save_aggregate`).

### Bring your own dataset

The two turnkey benchmarks load from the Hugging Face Hub; the others read
locally-prepared datasets from `GG_EVALS_DATA_ROOT`. To add your own data, save
it in the Arrow format the loader expects and point the env var at it. See
[`data/README.md`](data/README.md) for the directory layout, per-example schema,
and label conventions.

## Notes

- The `--nlogprobs` default (20) is enough to recover the `Yes`/`No` tokens for
  the probability computation. AUC metrics require it to be > 0.
- Some Hugging Face datasets are gated; accept their terms on the Hub and set
  `HF_TOKEN` in your environment before running.
