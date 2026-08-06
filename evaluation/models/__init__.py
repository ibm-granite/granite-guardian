"""Model registry — auto-discovers all model modules in this package.

Each model module must export:
  format_fn(sample, ds_config, tokenizer, **kwargs) -> prompt string
  parse_fn(output, tokenizer, nlogprobs) -> (label, prob)

The registry key is the module's ``MODEL_NAME`` (with underscores replaced by
dashes), e.g. ``granite-guardian-3``. To add a new model family (or a baseline
guardrail model), drop a new module here that exports the two functions above.
"""

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

MODELS = {}  # key -> (format_fn, parse_fn)

for _, name, _ in pkgutil.iter_modules(__path__):
    if name.startswith("_"):
        continue
    try:
        mod = importlib.import_module(f".{name}", __package__)
    except Exception as e:
        logger.debug(f"Skipping model {name}: {e}")
        continue
    if hasattr(mod, "format_fn") and hasattr(mod, "parse_fn"):
        key = getattr(mod, "MODEL_NAME", name).replace("_", "-")
        MODELS[key] = (mod.format_fn, mod.parse_fn)
