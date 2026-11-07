"""Chat templating, the single-token label check, and renormalising label log-probabilities.

Everything here works with any Hugging Face tokenizer and needs no GPU. Prompts are
always encoded with add_special_tokens=False, because the chat template already writes
any BOS token into the text; the same token ids go to vLLM and to the transformers check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .ladder import Rendered


def chat_prefix(tokenizer: Any, rendered: Rendered, chat_kwargs: dict | None = None) -> str:
    """Full prompt text up to and including the reply prefix (the action is next)."""
    messages = [
        {"role": "system", "content": rendered.system},
        {"role": "user", "content": rendered.user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(chat_kwargs or {})
    )
    return text + rendered.assistant


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def single_token_id(tokenizer: Any, prompt: str, label: str) -> int | None:
    """The token id of `label` after `prompt`, or None if it is not exactly one token.

    The check is encode(prompt + label) == encode(prompt) + [one token]: the label must
    be one token *and* must not merge with, or re-split, the end of the prompt.
    """
    base = encode(tokenizer, prompt)
    full = encode(tokenizer, prompt + label)
    if len(full) == len(base) + 1 and full[: len(base)] == base:
        return full[-1]
    return None


@dataclass
class LabelCheck:
    labels: tuple[str, ...]
    ok: bool
    ids: tuple[int, ...] | None
    problems: list[str] = field(default_factory=list)


def check_labels(tokenizer: Any, prompts: Iterable[str], labels: Sequence[str]) -> LabelCheck:
    """Check that every label is a single token, with the same id, after every prompt.

    The ids must also be distinct, and each must decode back to its label (up to leading
    and trailing spaces, which SentencePiece tokenizers drop when decoding one token).
    """
    ids: list[int | None] = [None] * len(labels)
    problems: list[str] = []
    for prompt in prompts:
        for j, label in enumerate(labels):
            tid = single_token_id(tokenizer, prompt, label)
            if tid is None:
                pieces = encode(tokenizer, prompt + label)[len(encode(tokenizer, prompt)) - 1:]
                problems.append(f"{label!r} is not one token here (tail tokens {pieces})")
            elif ids[j] is None:
                ids[j] = tid
            elif ids[j] != tid:
                problems.append(f"{label!r} has id {tid} here but {ids[j]} elsewhere")
    if not problems:
        if len(set(ids)) != len(ids):
            problems.append(f"labels share a token id: {ids}")
        for label, tid in zip(labels, ids):
            if tokenizer.decode([tid]).strip() != label.strip():
                problems.append(f"{label!r} decodes as {tokenizer.decode([tid])!r}")
    ok = not problems
    return LabelCheck(tuple(labels), ok, tuple(ids) if ok else None, sorted(set(problems)))


def renormalise(logprobs: Sequence[float]) -> tuple[list[float], float]:
    """From raw log-probabilities of the allowed tokens, return (probs, p_valid).

    probs are renormalised over the allowed set; p_valid is their total raw probability.
    """
    m = max(logprobs)
    if m == -math.inf:
        raise ValueError("every allowed token has probability zero")
    w = [math.exp(lp - m) for lp in logprobs]
    z = sum(w)
    return [x / z for x in w], math.exp(m) * z


def entropy_bits(probs: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)
