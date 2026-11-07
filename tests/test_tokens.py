import math

import pytest

from swarm_audit.ladder import Labels, render
from swarm_audit.tokens import check_labels, chat_prefix, entropy_bits, renormalise, single_token_id


@pytest.fixture(scope="module")
def toy_tokenizer():
    """A small byte-level BPE tokenizer trained here, so the test needs no download."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<s>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["Bid: H", "Bid: L"] * 50 + ["the history of it"] * 5, trainer)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.chat_template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
                          "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    return fast


def test_single_token_check_on_toy_tokenizer(toy_tokenizer):
    t = toy_tokenizer
    assert single_token_id(t, "Bid:", " H") is not None
    assert single_token_id(t, "Bid:", " high") is None          # several tokens
    assert single_token_id(t, "Bid: ", "H") is None             # " H" re-splits the prompt's space
    res = check_labels(t, ["Bid:", "x\nBid:"], [" H", " L"])
    assert res.ok and len(set(res.ids)) == 2
    assert not check_labels(t, ["Bid:"], [" H", " high"]).ok


def test_chat_prefix_ends_with_reply_prefix(toy_tokenizer):
    r = render("counts", (), d=2, labels=Labels(" H", " L"))
    text = chat_prefix(toy_tokenizer, r)
    assert text.startswith("<|system|>You are") and text.endswith("<|assistant|>Bid:")


def test_single_token_check_on_qwen3_tokenizer():
    """A known real tokenizer (skipped if it cannot be loaded, e.g. offline)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
    except Exception as e:  # no network and no cache
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")
    prompt = tok.apply_chat_template([{"role": "user", "content": "Bid H or L."}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) + "Bid:"
    assert "<think>\n\n</think>" in prompt
    assert check_labels(tok, [prompt], [" H", " L"]).ok
    assert single_token_id(tok, prompt, "H") is None  # ":H" merges into one token
    assert single_token_id(tok, prompt, " 1") is None  # space and digit are separate tokens


def test_renormalise_and_p_valid():
    probs, p_valid = renormalise([math.log(0.6), math.log(0.2)])
    assert probs == pytest.approx([0.75, 0.25])
    assert p_valid == pytest.approx(0.8)
    probs, p_valid = renormalise([-1000.0, -1001.0])  # tiny mass, no underflow in probs
    assert probs[0] == pytest.approx(1 / (1 + math.exp(-1)))
    assert entropy_bits([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy_bits([1.0, 0.0]) == 0.0
