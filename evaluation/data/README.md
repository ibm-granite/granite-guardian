# Preparing evaluation data

Two of the seven benchmarks load their data **live from the Hugging Face Hub** and
need no setup:

| Benchmark | Source | Setup |
|---|---|---|
| `groundedness_aggrefact` (LM-AggreFact) | [`lytang/LLM-AggreFact`](https://huggingface.co/datasets/lytang/LLM-AggreFact) | none (may require accepting the dataset terms on the Hub + `HF_TOKEN`) |
| `function_calling` (FC-Reward-Bench) | [`ibm-research/fc-reward-bench`](https://huggingface.co/datasets/ibm-research/fc-reward-bench) | none |

The others read locally-prepared data:

- `ood_safety` (harm) and `groundedness_true` (TRUE) are evaluated on **curated
  subsets** of public datasets.
- `ifeval_multiconstraint` is a held-out split of an IF-RLVR-derived
  requirement-checking set built for the Requirement Check adapters.
- `infobench_gpt4` / `infobench_human` read the InfoBench annotation files,
  which are distributed outside the Hugging Face Hub.

We do **not** redistribute any of these. This page documents the directory
layout and per-example schema each loader expects so you can assemble them
yourself from the original sources.

Point the loader at your prepared directory with an environment variable:

```bash
export GG_EVALS_DATA_ROOT=/path/to/your/eval_data
```

Each dataset lives in its own subdirectory. The harm, TRUE and IFEval sets are saved
in the Hugging Face `datasets` Arrow format (i.e. written with
`Dataset.save_to_disk(...)` / `DatasetDict.save_to_disk(...)` and read back with
`load_from_disk(...)`); InfoBench is read from the raw annotation files.

```
$GG_EVALS_DATA_ROOT/
  AegisSafetyTest/ ... xstest_RR(h)/   # ood_safety (10 dirs)
  begin/ ... summeval/                 # groundedness_true (9 dirs)
  ifeval_multiconstraint/              # ifeval_multiconstraint (test split)
  infobench/                           # infobench_gpt4 + infobench_human
```

## Label convention

Every example carries an integer `label` (or `labels`) column:

- **Harm (`ood_safety`)**: `1` = unsafe / harmful, `0` = safe. String labels
  `"unsafe"`/`"safe"` and `"1"`/`"0"` are also accepted and normalized.
- **TRUE (`groundedness_true`)**: the label is used directly as the positive
  class for the groundedness AUC (the model's probability of *risk* is scored
  against it). Keep the orientation of the source TRUE release you build from;
  do not re-invert it.
- **Requirement checking (`ifeval_multiconstraint`, `infobench_*`)**: the
  positive class is *satisfied*: `1` / `"yes"` = the response meets the
  requirement(s), `0` / `"no"` = it does not. (The opposite polarity to harm.)

Column names are auto-detected: the loader tries the configured column first,
then falls back through common alternatives (`prompt`/`text`/`Question`/`instruction`
for the user turn, `response`/`Answer`/`claim` for the assistant turn,
`context`/`document`/`doc` for the document). So a harm dataset may store its
prompt under `text` or `Question` and it will still be picked up.

---

## `ood_safety`: harm / OOD (10 datasets)

Reported as **F1** per dataset; the aggregate is the mean F1. Create one
subdirectory per dataset below, each with a `test` split. Prompt-only datasets
provide just the user turn; response datasets provide a user turn *and* an
assistant turn.

| Directory | Type | Source |
|---|---|---|
| `AegisSafetyTest` | prompt | [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) |
| `BeaverTails` | prompt + response | [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) |
| `HarmBench_Prompt` | prompt | [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench) |
| `OAI_hf` | prompt | [mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation) |
| `SafeRLHF_test` | prompt + response | [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) |
| `simpleSafetyTest` | prompt | [Bertievidgen/SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) |
| `toxic_chat` | prompt | [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) |
| `xstest_RH` | prompt + response | [natolambert/xstest-v2-copy](https://huggingface.co/datasets/natolambert/xstest-v2-copy) (response harmfulness) |
| `xstest_RR` | prompt + response | XSTest (response refusal) |
| `xstest_RR(h)` | prompt + response | XSTest (response refusal, harmful prompts only) |

Minimal per-example schema:

```python
# prompt-only (e.g. AegisSafetyTest, HarmBench_Prompt, simpleSafetyTest, toxic_chat, OAI_hf)
{"text": "<user message>", "label": 0 or 1}

# prompt + response (e.g. BeaverTails, SafeRLHF_test, xstest_*)
{"Question": "<user message>", "Answer": "<assistant reply>", "label": 0 or 1}
```

The three XSTest variants differ only in how the label is derived from the same
XSTest examples (response harmfulness, response refusal, and response refusal
restricted to harmful prompts); build all three from the XSTest release.

## `groundedness_true`: TRUE / RAG hallucination (9 datasets)

Reported as **AUC** per dataset (cards 3.0-3.2) and **balanced accuracy** (card
3.3); both are written to `Aggregate.json`. Each subdirectory has a single
`train` split (the TRUE benchmark ships these as one partition).

Directories: `begin`, `dialfact`, `frank`, `mnbm`, `paws`, `q2`, `qags_cnndm`,
`qags_xsum`, `summeval`.

Build these from the TRUE benchmark
([google-research/true](https://github.com/google-research/true)), which provides
scripts to assemble each dataset from its upstream source. Per-example schema:

```python
{"document": "<grounding document>", "response": "<claim / summary>", "label": 0 or 1}
```

The loader chunks long documents by sentence, scores each claim against each
chunk, and aggregates to a per-document verdict, so no manual truncation is
needed.

## `ifeval_multiconstraint`: IFEval multi-constraint requirement checking

Reported as **balanced accuracy** (Requirement Check adapter card, "IFEval"
column). The loader reads the `test` split (3,750 rows) of
`$GG_EVALS_DATA_ROOT/ifeval_multiconstraint`, a `DatasetDict` with
`train` / `validation` / `test` splits. Only `test` is used; it is held out
from Requirement Check adapter training.

Per-example schema:

```python
{
    "prompt": "<instruction, with the constraints appended>",
    "response": "<model response>",
    "constraints": "<constraint 1>\t<constraint 2>...",  # tab-separated, 1-5 constraints
    "pointwise_score": 1-5,                             # not used by the benchmark
    "binary_label": "yes" or "no",                      # yes = pointwise_score >= 4
}
```

To build it:

1. **Prompts.** Start from
   [`allenai/IF_multi_constraints_upto5`](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)
   (IF-RLVR training data: an instruction with 1-5 verifiable IFEval-style
   constraints appended).
2. **Responses.** Generate one response per prompt with
   [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1).
3. **Labels.** Check each constraint against the response with the programmatic
   verifiers from [IFBench](https://github.com/allenai/IFBench) and bucket the
   fraction satisfied into a 1-5 `score`: 1 = none, 2 = under half, 3 = exactly
   half, 4 = over half but not all, 5 = all. Then `pointwise_score = score` and
   `binary_label = "yes" if score >= 4 else "no"` (more than half satisfied).
   Our pool had 95,373 rows.
4. **Subsample and split.** Draw 5,000 rows per score level (25,000 total, so all
   five levels are balanced), shuffle, and split 70 / 15 / 15 into `train`
   (17,500) / `validation` (3,750) / `test` (3,750). The resulting test split has
   1,511 `yes` and 2,239 `no` rows.
5. Save the `DatasetDict` with `save_to_disk` to
   `$GG_EVALS_DATA_ROOT/ifeval_multiconstraint`.

Response generation is sampled and the subsample/split are random, so a rebuild
yields a statistically equivalent set rather than the exact rows behind the
card numbers; expect small differences from the card.

## `infobench_gpt4` / `infobench_human`: InfoBench

Reported as **balanced accuracy** per answering LLM; the aggregate is the mean
over answering LLMs (Requirement Check adapter card, "InfoBench-GPT4" and
"InfoBench-Human" columns). Each InfoBench instruction is decomposed into yes/no
requirement questions; each (instruction, response, question) becomes one row.

The instructions are on the Hub ([`kqsong/InFoBench`](https://huggingface.co/datasets/kqsong/InFoBench)),
but the responses and their annotations are not. Download them from the
**InFoBench Generation and Annotation** Google Drive folder linked from the
[InfoBench repository](https://github.com/qinyiwei/InfoBench), and lay them out
unmodified:

```
$GG_EVALS_DATA_ROOT/infobench/
  gpt4_annotation/0314/                               # from the Drive folder gpt4_annotation/
    Llama-2-70b-chat-hf_eval.json
    claude-2.1_eval.json
    gemini-pro_eval.json
    gpt-3.5-turbo-1106_eval.json
    gpt-4-1106-preview_eval.json
    vicuna-13b-v1.5_eval.json
  InFoBench Expert Annotation - Easy_subset.csv       # from the Drive folder expert_annotation/
  InFoBench Expert Annotation - Hard_subset.csv
```

- **`infobench_gpt4`** reads the GPT-4-0314 annotation files: JSON lines, one
  instruction per line with `instruction`, `input`, `output` (the answering
  LLM's response), `decomposed_questions` (list) and `eval` (list of booleans,
  one per question). All 500 instructions × 6 answering LLMs → 13,471 rows
  (gemini-pro answered 491 instructions).
- **`infobench_human`** reads the expert annotations (the two sheets of the
  "InFoBench Expert Annotation" spreadsheet, exported as CSV under the names
  above): one row per instruction with `instruction`, `input`,
  `decomposed_questions` (a numbered list) and, for each answering LLM
  (`gpt-3.5-turbo`, `gpt-4`, `claude-v1`, `alpaca-7b`, `vicuna-13b`), a response
  column and a `<model>-annotation` column of dot-separated per-question bits
  (e.g. `1.0.1`). 50 instructions × 5 answering LLMs → 1,129 rows.

The prompt shown to the judge is the instruction followed by a blank line and the
`input`, when there is one.
