# Preparing evaluation data

Two of the four benchmarks load their data **live from the Hugging Face Hub** and
need no setup:

| Benchmark | Source | Setup |
|---|---|---|
| `groundedness_aggrefact` (LM-AggreFact) | [`lytang/LLM-AggreFact`](https://huggingface.co/datasets/lytang/LLM-AggreFact) | none (may require accepting the dataset terms on the Hub + `HF_TOKEN`) |
| `function_calling` (FC-Reward-Bench) | [`ibm-research/fc-reward-bench`](https://huggingface.co/datasets/ibm-research/fc-reward-bench) | none |

The other two, `ood_safety` (harm) and `groundedness_true` (TRUE), are evaluated
on **curated subsets** of public datasets. We do **not** redistribute those subsets.
This page documents the directory layout and per-example schema the loader expects
so you can assemble them yourself from the original sources.

Point the loader at your prepared directory with an environment variable:

```bash
export GG_EVALS_DATA_ROOT=/path/to/your/eval_data
```

Each dataset lives in its own subdirectory saved in the Hugging Face `datasets`
Arrow format (i.e. written with `Dataset.save_to_disk(...)` / `DatasetDict.save_to_disk(...)`
and read back with `load_from_disk(...)`).

## Label convention

Every example carries an integer `label` (or `labels`) column:

- **Harm (`ood_safety`)**: `1` = unsafe / harmful, `0` = safe. String labels
  `"unsafe"`/`"safe"` and `"1"`/`"0"` are also accepted and normalized.
- **TRUE (`groundedness_true`)**: the label is used directly as the positive
  class for the groundedness AUC (the model's probability of *risk* is scored
  against it). Keep the orientation of the source TRUE release you build from;
  do not re-invert it.

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
