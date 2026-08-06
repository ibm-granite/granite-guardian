"""LM-AggreFact benchmark — balanced accuracy per dataset + overall.

Data is loaded live from Hugging Face (``lytang/LLM-AggreFact``); nothing is
stored in this repo. You may need to accept the dataset terms on the Hub and
set ``HF_TOKEN`` in your environment.

Granite Guardian models are run with the ``groundedness`` criterion. A guard
verdict of "yes" means *ungrounded* (risk detected), whereas LM-AggreFact
label 1 means *supported*, so predictions are flipped before scoring.
"""

import json
import os

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score

HF_DATASET = "lytang/LLM-AggreFact"
SPLIT = "test"


def _per_dataset_bacc(df):
    rows = []
    for name in sorted(df.dataset.unique()):
        sub = df[df.dataset == name]
        rows.append({"Dataset": name,
                     "BAcc": balanced_accuracy_score(sub.label, sub.preds) * 100})
    result = pd.DataFrame(rows)
    result.loc[len(result)] = {"Dataset": "Average", "BAcc": result.BAcc.mean()}
    return result.round(1)


def run(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    shard_index = getattr(args, "shard_index", 0)
    num_shards = getattr(args, "num_shards", 1)

    print(f"\n  Loading {HF_DATASET} [{SPLIT}]")
    df = pd.DataFrame(load_dataset(HF_DATASET, split=SPLIT))
    print(f"  {len(df)} samples total")
    if num_shards > 1:
        df = df.iloc[shard_index::num_shards].reset_index(drop=True)
        print(f"  Shard {shard_index}/{num_shards}: {len(df)} samples")

    llm, tokenizer, sp = ensure_model()
    ds_config = {"context_col": "doc", "assistant_col": "claim"}
    fmt_kw = {"think": args.think, "criteria_id": "groundedness"}

    from benchmarks._common import format_all, generate_and_parse
    prompts = format_all(df.to_dict("records"), guard_fmt, ds_config, tokenizer, **fmt_kw)
    labels, probs, resp = generate_and_parse(
        prompts, llm, sp, guard_parse, tokenizer,
        batch_size=args.batch_size, nlogprobs=args.nlogprobs)

    preds = pd.Series(labels)
    na = preds.isna()
    # Guard "yes" = ungrounded = NOT supported, but label 1 = supported → flip.
    preds[~na] = 1 - preds[~na]
    if na.sum():
        preds[na] = 1 - df.label[na]
    df["preds"] = preds.astype(int).values

    # Save predictions (needed for shard merging).
    df.to_json(os.path.join(out_base, "predictions.jsonl"), orient="records", lines=True)

    if num_shards > 1:
        print(f"  Shard {shard_index} done — predictions saved; the orchestrator will merge.")
        return

    bacc_table = _per_dataset_bacc(df)
    print(f"\n{bacc_table.to_string(index=False)}")
    bacc_table.to_csv(os.path.join(out_base, "per_dataset_bacc.csv"), index=False)

    for _, row in bacc_table.iterrows():
        if row["Dataset"] == "Average":
            continue
        with open(os.path.join(out_base, f"results_{row['Dataset']}.json"), "w") as f:
            json.dump({"model_name": args.base_model, "dataset": row["Dataset"],
                       "benchmark": "groundedness_aggrefact",
                       "balanced_accuracy": row["BAcc"] / 100}, f, indent=4)

    overall = balanced_accuracy_score(df.label, df.preds)
    avg_per_ds = bacc_table[bacc_table["Dataset"] == "Average"]["BAcc"].values[0] / 100
    print(f"\n  Overall BAcc={overall:.4f}  Avg per-dataset={avg_per_ds:.3f}")

    agg = {"model_name": args.base_model, "benchmark": "groundedness_aggrefact",
           "balanced_accuracy": overall, "avg_bacc_per_dataset": avg_per_ds, "Failed": 0}
    with open(os.path.join(out_base, "Aggregate.json"), "w") as f:
        json.dump(agg, f, indent=4)

run.SUPPORTS_SHARDING = True
