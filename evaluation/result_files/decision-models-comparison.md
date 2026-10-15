# Granite Libraries adapters vs. decision models

An example comparison of the Granite 4.1 Guardian Core and Requirement Check
adapters against prompted Granite 4.1 base models and four hosted decision
models. All numbers are transcribed from the model cards:

- Guardian Core: [`ibm-granite/granitelib-guardian-r1.0` / `guardian-core`](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/blob/main/guardian-core/README.md#performance-comparison-to-decision-models)
- Requirement Check: [`ibm-granite/granitelib-core-r1.0` / `requirement-check`](https://huggingface.co/ibm-granite/granitelib-core-r1.0/blob/main/requirement-check/README.md#performance-comparison-to-decision-models)

The adapter rows are reproduced by this framework; see
[Reproducing the card benchmarks](../README.md#granite-libraries-adapters).
The decision models are not covered by the framework.

The OOD Safety numbers follow the card protocol, which uses the assistant-text
scoring schema on every row, prompt-only datasets included. This framework uses
the user-text schema for prompt-only rows; see
[Scoring schema for prompt-only harm](../README.md#scoring-schema-for-prompt-only-harm).

"Best prompted Granite-4.1" is the best of the 3B / 8B / 30B base models prompted
with the same task, and "Best trained adapter" is the best of the 4.1 LoRA / aLoRA
adapters, each taken per benchmark.

## Guardian Core

| Benchmark | Metric | Best prompted Granite-4.1 | Best trained adapter | [jev-1.13](https://openrouter.ai/typesafe/jev-1.13) | [microsoft-decision-1](https://openrouter.ai/microsoft/microsoft-decision-1) | [gpt-6-luna-decisions](https://openrouter.ai/openai/gpt-6-luna-decisions) | [clef-flash](https://openrouter.ai/cloudflare/clef-flash) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| OOD Safety (`ood_safety`) | F1 | 0.690 | 0.810 | 0.779 | 0.787 | 0.769 | 0.761 |
| RAG Hallucination (`groundedness_aggrefact`) | BAcc | 0.496 | 0.757 | 0.783 | 0.778 | 0.764 | 0.672 |
| Function Calling (`function_calling`) | BAcc | 0.490 | 0.790 | 0.689 | 0.741 | 0.731 | 0.542 |

## Requirement Check

| Benchmark | Metric | Best prompted Granite-4.1 | Best trained adapter | [jev-1.13](https://openrouter.ai/typesafe/jev-1.13) | [microsoft-decision-1](https://openrouter.ai/microsoft/microsoft-decision-1) | [gpt-6-luna-decisions](https://openrouter.ai/openai/gpt-6-luna-decisions) | [clef-flash](https://openrouter.ai/cloudflare/clef-flash) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| IFEval (`ifeval_multiconstraint`) | BAcc | 0.624 | 0.855 | 0.685 | 0.678 | 0.633 | 0.663 |
| InfoBench-GPT4 (`infobench_gpt4`) | BAcc | 0.686 | 0.751 | 0.768 | 0.769 | 0.783 | 0.730 |
| InfoBench-Human (`infobench_human`) | BAcc | 0.624 | 0.705 | 0.699 | 0.698 | 0.713 | 0.693 |

## OOD Safety, per dataset

F1 on each `ood_safety` dataset (Guardian Core card, extended with the decision
models). Adapter rows are in bold.

| Model | AVG | AegisSafetyTest | BeaverTails | HarmBench | OAI_hf | SafeRLHF | simpleSafety | toxic_chat | xstest_RH | xstest_RR | xstest_RR(h) |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| granite-4.1-3b (prompting) | 0.01 | 0.00 | 0.00 | 0.05 | 0.03 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 |
| granite-4.1-8b (prompting) | 0.67 | 0.77 | 0.66 | 0.68 | 0.83 | 0.60 | 0.96 | 0.61 | 0.59 | 0.37 | 0.61 |
| granite-4.1-30b (prompting) | 0.69 | 0.72 | 0.65 | 0.69 | 0.83 | 0.67 | 0.97 | 0.67 | 0.68 | 0.37 | 0.66 |
| granite-guardian-3.3-8b (no_think) | 0.81 | 0.87 | 0.84 | 0.80 | 0.77 | 0.80 | 0.99 | 0.76 | 0.90 | 0.49 | 0.87 |
| **granite-guardian-4.0-micro (LoRA)** | **0.76** | 0.81 | 0.76 | 0.79 | 0.81 | 0.74 | 0.99 | 0.75 | 0.85 | 0.39 | 0.77 |
| **granite-guardian-4.1-3b (LoRA)** | **0.77** | 0.79 | 0.75 | 0.79 | 0.81 | 0.74 | 0.98 | 0.77 | 0.84 | 0.41 | 0.78 |
| **granite-guardian-4.1-3b (aLoRA)** | **0.78** | 0.82 | 0.81 | 0.79 | 0.78 | 0.78 | 0.99 | 0.73 | 0.89 | 0.42 | 0.81 |
| **granite-guardian-4.1-8b (LoRA)** | **0.79** | 0.85 | 0.80 | 0.79 | 0.81 | 0.79 | 0.98 | 0.77 | 0.91 | 0.42 | 0.82 |
| **granite-guardian-4.1-8b (aLoRA)** | **0.80** | 0.86 | 0.81 | 0.78 | 0.83 | 0.81 | 0.99 | 0.80 | 0.91 | 0.43 | 0.82 |
| **granite-guardian-4.1-30b (LoRA)** | **0.80** | 0.83 | 0.79 | 0.80 | 0.84 | 0.79 | 0.99 | 0.77 | 0.90 | 0.45 | 0.84 |
| **granite-guardian-4.1-30b (aLoRA)** | **0.81** | 0.87 | 0.82 | 0.80 | 0.82 | 0.82 | 0.99 | 0.76 | 0.92 | 0.45 | 0.85 |
| [jev-1.13](https://openrouter.ai/typesafe/jev-1.13) | 0.78 | 0.75 | 0.72 | 0.76 | 0.83 | 0.81 | 0.99 | 0.75 | 0.93 | 0.42 | 0.81 |
| [microsoft-decision-1](https://openrouter.ai/microsoft/microsoft-decision-1) | 0.79 | 0.76 | 0.77 | 0.77 | 0.83 | 0.81 | 0.98 | 0.72 | 0.91 | 0.47 | 0.86 |
| [gpt-6-luna-decisions](https://openrouter.ai/openai/gpt-6-luna-decisions) | 0.77 | 0.72 | 0.78 | 0.75 | 0.84 | 0.80 | 0.98 | 0.59 | 0.90 | 0.48 | 0.85 |
| [clef-flash](https://openrouter.ai/cloudflare/clef-flash) | 0.76 | 0.75 | 0.75 | 0.72 | 0.83 | 0.70 | 0.99 | 0.76 | 0.87 | 0.44 | 0.79 |
