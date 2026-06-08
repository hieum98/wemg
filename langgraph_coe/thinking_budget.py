"""Server-side thinking-token budget for Qwen-on-SGLang.

SGLang exposes **no** request parameter that caps the number of *reasoning*
tokens (unlike Anthropic's ``thinking.budget_tokens``). The only mechanism is a
**custom logit processor**: the server must be launched with
``--enable-custom-logit-processor``, and each request carries a processor that
forces the ``</think>`` token once ``thinking_budget`` thinking tokens have been
emitted. SGLang ships a built-in ``Qwen3ThinkingBudgetLogitProcessor``, but its
token ids are hardcoded for the *original* Qwen3 family (``<think>``=151667).
Qwen3.5 uses different ids (248068/248069), so the built-in silently no-ops.

How this module avoids the sharp edges
--------------------------------------
* **Param-driven, one blob for every model.** The processor reads the
  ``<think>`` / ``</think>`` / newline token ids out of ``custom_params`` at
  request time instead of hardcoding them, so a single serialized blob serves
  Qwen3, Qwen3.5, … — the caller just supplies the right ids (see
  ``THINK_TOKEN_IDS``).
* **Python-version pinning.** The processor is shipped to the server as a
  ``dill``-pickled class, and ``dill`` embeds version-specific bytecode. A blob
  pickled in the app venv (Py3.13) FAILS to unpickle on the server (Py3.12).
  So the blob is generated **with the server's Python** and committed verbatim
  next to this file (:data:`CLP_BLOB_PATH`). At runtime the app reads it as an
  opaque string and never imports ``dill`` or ``sglang``.

Regenerating the blob (only needed if ``_ThinkingBudgetLogitProcessor.__call__``
changes — token ids are data, not code):

    <server-sglang-env>/bin/python -m langgraph_coe.thinking_budget --regen

A malformed processor crashes the whole SGLang worker (``SIGQUIT``), so never
hand-edit the committed blob.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CLP_BLOB_PATH = Path(__file__).resolve().parent / "thinking_budget_clp.json"

# model-name substring -> (<think> id, </think> id, newline id).
# Match is case-insensitive against the model name with any provider prefix
# (e.g. "openai/") stripped. Order matters: most specific first.
# Verified live: Qwen3.5-4B -> (248068, 248069, 198).
# Qwen3 / Qwen3-Next share the original Qwen3 special-token ids (unverified for
# Qwen3-Next at time of writing — override per tier if it ever differs).
THINK_TOKEN_IDS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("qwen3.5", (248068, 248069, 198)),
    ("qwen3-next", (151667, 151668, 198)),
    ("qwen3", (151667, 151668, 198)),
)


def resolve_think_token_ids(model_name: str) -> Optional[Tuple[int, int, int]]:
    """Return ``(start, end, newline)`` think-token ids for *model_name*, or None.

    None means we don't know this model's ids — the caller should skip applying a
    budget (and warn) rather than send wrong ids that would silently no-op.
    """
    name = model_name.split("/")[-1].lower()
    for needle, ids in THINK_TOKEN_IDS:
        if needle in name:
            return ids
    return None


class _ThinkingBudgetLogitProcessor:
    """Force ``</think>`` once the thinking budget is spent.

    Mirrors SGLang's ``ThinkingBudgetLogitProcessor`` but reads token ids from
    ``custom_params`` so one pickled blob works for every model. Pure stdlib so
    ``dill`` can serialize it by value; the server injects ``__req__`` per row.
    """

    def __call__(
        self, logits, custom_param_list: Optional[List[Dict[str, Any]]] = None
    ):
        if not custom_param_list:
            return logits
        for i, p in enumerate(custom_param_list):
            if not p:
                continue
            tb = p.get("thinking_budget")
            if tb is None or not isinstance(tb, int) or tb < 0:
                continue
            start_id = p.get("think_start_id")
            end_id = p.get("think_end_id")
            newline_id = p.get("newline_id")
            if start_id is None or end_id is None or newline_id is None:
                continue
            req = p.get("__req__")
            if req is None:
                continue
            cur_ids = [*req.origin_input_ids, *req.output_ids]
            # Only act while inside the thinking span.
            if start_id not in cur_ids or end_id in cur_ids:
                continue
            start_index = cur_ids.index(start_id)
            num_after_start = len(cur_ids) - start_index - 1
            if num_after_start < tb:
                continue
            # Emit a newline before </think> if the last token isn't one.
            if not req.output_ids or req.output_ids[-1] != newline_id:
                logits[i, :] = -float("inf")
                logits[i, newline_id] = 0.0
                continue
            logits[i, :] = -float("inf")
            logits[i, end_id] = 0.0
        return logits


def load_clp_blob() -> str:
    """Read the committed, server-Python-pinned processor blob (opaque string)."""
    return CLP_BLOB_PATH.read_text(encoding="utf-8")


def build_request_kwargs(
    model_name: str, thinking_budget: int
) -> Optional[Dict[str, Any]]:
    """Build the ``model_kwargs`` fragment that enforces *thinking_budget*.

    Returns ``{"custom_logit_processor": <blob>, "custom_params": {...}}`` or
    ``None`` when the model's think-token ids are unknown (logs a warning so the
    fallback is observable, never silent).
    """
    if thinking_budget is None or thinking_budget < 0:
        return None
    ids = resolve_think_token_ids(model_name)
    if ids is None:
        logger.warning(
            "thinking_budget set for model %r but its <think>/</think> token ids "
            "are unknown (see THINK_TOKEN_IDS); skipping budget enforcement.",
            model_name,
        )
        return None
    start_id, end_id, newline_id = ids
    return {
        "custom_logit_processor": load_clp_blob(),
        "custom_params": {
            "thinking_budget": thinking_budget,
            "think_start_id": start_id,
            "think_end_id": end_id,
            "newline_id": newline_id,
        },
    }


def _regen() -> None:
    """Serialize the processor with the *current* interpreter into CLP_BLOB_PATH.

    Run with the SERVER's Python (the one running SGLang) so the dill bytecode
    matches what the server will unpickle.
    """
    import sys

    import dill  # only needed for regeneration

    blob = json.dumps({"callable": dill.dumps(_ThinkingBudgetLogitProcessor).hex()})
    CLP_BLOB_PATH.write_text(blob, encoding="utf-8")
    print(
        f"wrote {CLP_BLOB_PATH} ({len(blob)} bytes) using Python {sys.version.split()[0]}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen",
        action="store_true",
        help="Regenerate the committed processor blob with this interpreter.",
    )
    args = ap.parse_args()
    if args.regen:
        _regen()
    else:
        ap.print_help()
