"""Loading Granite Libraries LoRA / aLoRA adapters.

An adapter is addressed by ``--model-path``, either as a local directory holding
``adapter_config.json`` or as ``<hf-repo>/<subfolder>``, e.g.
``ibm-granite/granitelib-guardian-r1.0/guardian-core/granite-4.1-8b/lora``.
The base model and adapter type come from ``adapter_config.json``: an adapter
with ``alora_invocation_tokens`` is an aLoRA, anything else a plain LoRA.

- **LoRA** runs on vLLM with a ``LoRARequest``.
- **aLoRA** runs on Hugging Face transformers + PEFT, since vLLM has no aLoRA
  support. ``PeftLLM`` exposes the slice of the vLLM interface the benchmarks
  use (``generate`` / ``get_tokenizer``, step logprobs) so they run unchanged.
"""

import json
import os


def resolve_adapter(model_path):
    """Return the local adapter directory for ``model_path``, or None if it is a full model."""
    if os.path.isdir(model_path):
        return model_path if os.path.isfile(os.path.join(model_path, "adapter_config.json")) else None
    parts = model_path.strip("/").split("/")
    if len(parts) <= 2:
        return None  # a plain HF model id
    from huggingface_hub import snapshot_download

    repo_id, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    local = os.path.join(snapshot_download(repo_id, allow_patterns=f"{subfolder}/*"), subfolder)
    if not os.path.isfile(os.path.join(local, "adapter_config.json")):
        raise FileNotFoundError(f"No adapter_config.json under {repo_id}/{subfolder}")
    return local


def read_adapter_config(adapter_path):
    """Return (base_model, adapter_type, rank) from ``adapter_config.json``."""
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        cfg = json.load(f)
    adapter_type = "alora" if cfg.get("alora_invocation_tokens") else "lora"
    return cfg["base_model_name_or_path"], adapter_type, cfg["r"]


# Every Granite Libraries adapter answers with a single JSON object, and the r1.0
# adapters do not reliably emit EOS after it, so generation stops at the closing
# brace. A string stop is needed: the brace arrives fused as the `"}` token.
STOP_STRINGS = ["}"]


def load_adapter_model(args):
    """Load base model + adapter. Returns (llm, tokenizer, sampling_params)."""
    if args.adapter_type == "alora":
        llm = PeftLLM(args)
        return llm, llm.get_tokenizer(), None

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading base model: {args.base_model} (LoRA, vLLM)")
    llm_kwargs = dict(model=args.base_model, tensor_parallel_size=args.ngpus,
                      seed=args.seed, trust_remote_code=True,
                      enable_lora=True, max_loras=1, max_lora_rank=args.adapter_rank)
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0,
        repetition_penalty=args.repetition_penalty,
        logprobs=args.nlogprobs if args.nlogprobs > 0 else None,
        stop=STOP_STRINGS, include_stop_str_in_output=True,
        stop_token_ids=[tokenizer.eos_token_id])
    return LoRALLM(llm, LoRARequest("adapter", 1, args.adapter_path)), tokenizer, sp


class LoRALLM:
    """vLLM ``LLM`` that applies one LoRA adapter to every request."""

    def __init__(self, llm, lora_request):
        self.llm = llm
        self.lora_request = lora_request

    def get_tokenizer(self):
        return self.llm.get_tokenizer()

    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)


class PeftLLM:
    """Hugging Face + PEFT backend with a vLLM-shaped ``generate``.

    Greedy decoding over length-sorted, left-padded batches. Each batch holds at
    most ``args.hf_batch_tokens`` padded tokens. With ``nlogprobs > 0`` every
    output carries vLLM-style step logprobs (top-N candidates per generated
    token, from the raw logits) so the yes/no probability, and hence AUC, is
    computed exactly as on the vLLM path.
    """

    def __init__(self, args):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model: {args.base_model} (aLoRA, HF + PEFT)")
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        self.tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="auto")
        self.model = PeftModel.from_pretrained(model, args.adapter_path).eval()
        self.invocation = list(self.model.peft_config["default"].alora_invocation_tokens)
        self.max_tokens = args.max_tokens
        self.nlogprobs = args.nlogprobs
        self.batch_tokens = args.hf_batch_tokens
        self._decoded = {}  # token id -> decoded string

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params=None):
        encoded = [self.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        for i, ids in enumerate(encoded):
            if not self._contains(ids, self.invocation):
                # PEFT would silently fall back to the base model for this row.
                raise ValueError(
                    f"Prompt {i} does not contain the aLoRA invocation tokens "
                    f"{self.invocation}; the adapter would not be activated.")

        outputs = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
        batch = []
        for i in order:
            # Sorted ascending, so the newest prompt is the longest in the batch.
            if batch and len(encoded[i]) * (len(batch) + 1) > self.batch_tokens:
                self._generate_batch(batch, encoded, prompts, outputs)
                batch = []
            batch.append(i)
        if batch:
            self._generate_batch(batch, encoded, prompts, outputs)
        return outputs

    @staticmethod
    def _contains(ids, sub):
        n = len(sub)
        return any(ids[j:j + n] == sub for j in range(len(ids) - n + 1))

    def _generate_batch(self, batch, encoded, prompts, outputs):
        import torch

        inputs = self.tokenizer.pad(
            {"input_ids": [encoded[i] for i in batch]}, return_tensors="pt"
        ).to(self.model.device)
        gen_kwargs = dict(max_new_tokens=self.max_tokens, do_sample=False,
                          stop_strings=STOP_STRINGS, tokenizer=self.tokenizer,
                          pad_token_id=self.tokenizer.pad_token_id)
        if self.nlogprobs > 0:
            gen_kwargs.update(return_dict_in_generate=True, output_logits=True)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        sequences = out.sequences if self.nlogprobs > 0 else out
        new_tokens = sequences[:, inputs["input_ids"].shape[1]:].tolist()

        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        for row, i in enumerate(batch):
            tokens = new_tokens[row]
            # Rows that stopped early are padded out to the longest row.
            end = next((t for t, tok in enumerate(tokens) if tok in stop_ids), len(tokens))
            tokens = tokens[:end]
            logprobs = None
            if self.nlogprobs > 0:
                logprobs = self._step_logprobs([step[row] for step in out.logits[:end]], tokens)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            outputs[i] = _Output(prompts[i], text, logprobs)

    def _step_logprobs(self, step_logits, tokens):
        """One ``{token_id: Logprob}`` dict per generated token, as vLLM returns."""
        import torch

        steps = []
        for logits, chosen in zip(step_logits, tokens):
            lp = torch.log_softmax(logits.float(), dim=-1)
            top = torch.topk(lp, self.nlogprobs)
            cands = dict(zip(top.indices.tolist(), top.values.tolist()))
            cands.setdefault(chosen, float(lp[chosen]))  # vLLM also adds the sampled token
            step = {}
            for tid, logprob in cands.items():
                if tid not in self._decoded:
                    self._decoded[tid] = self.tokenizer.decode([tid])
                step[tid] = _Logprob(logprob, self._decoded[tid])
            steps.append(step)
        return steps


class _Logprob:
    """The fields of ``vllm.logprobs.Logprob`` that the parsers read."""

    def __init__(self, logprob, decoded_token):
        self.logprob = logprob
        self.decoded_token = decoded_token


class _Completion:
    def __init__(self, text, logprobs):
        self.text = text
        self.logprobs = logprobs


class _Output:
    """The fields of ``vllm.RequestOutput`` that the benchmarks read."""

    def __init__(self, prompt, text, logprobs):
        self.prompt = prompt
        self.outputs = [_Completion(text, logprobs)]
