"""Shared fixtures: an offline toy tokenizer and a fake label-probability server."""

import hashlib
import math

import pytest


@pytest.fixture(scope="session")
def toy_tokenizer():
    """A small byte-level BPE tokenizer trained here, so tests need no download."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = (["Bid: H", "Bid: L", "Message: high", "Message: low", "Message: none"] * 50
              + ["the history of it"] * 5)
    tok.train_from_iterator(corpus, trainer)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.chat_template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
                          "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    return fast


def fake_call(prompts, ids, kind):
    """Deterministic stand-in for Server.label_logprobs: the log-probabilities are a
    function of the prompt's token ids only, as the kernel K_i(. | h) should be."""
    out = []
    for p in prompts:
        h = hashlib.sha256(repr((p, list(ids))).encode()).digest()
        raw = [0.9 * (0.05 + h[j] / 255) / len(ids) for j in range(len(ids))]
        lp = [math.log(x) for x in raw]
        z = sum(raw)
        out.append({"logprobs": lp, "probs": [x / z for x in raw], "p_valid": z})
    return out


@pytest.fixture(scope="session")
def fake_server():
    return fake_call
