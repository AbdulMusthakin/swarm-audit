## Recommendation (provisional, for Abdul to decide)

**Status, 7 Nov 2026.** Done: the tokenizer screen, label choice and L4 sizing for every
non-gated candidate. Not done: the screening grid and the exactness check. The GPU work
is blocked because this environment has no Modal credentials, and the hand-off says to
use a teammate's account, not Abdul's. So nothing below rests on model behaviour yet. The
pipeline has only been checked end to end on CPU with a 0.6B stand-in model, and those
numbers are not reported here.

**Hard requirements (section 3), as far as they can be checked without a GPU:**

1. *No reasoning before the answer.* Holds for every candidate by construction: the reply
   is prefilled with `Bid:`, so the action is the first generated token. Qwen3/3.5 also
   need `enable_thinking=False`; the rendered prompt shows the empty think block.
   Gemma 4 has thinking off by default, and its generation prompt opens no thought
   channel.
2. *Single-token labels.* Every screened model passes with ` H`/` L` for bids and
   ` high`/` low`/` none` for messages, at the position where they are generated. Labels
   without the leading space merge with the colon for Qwen and OLMo, and ` 1`/` 2`/` 3`
   split into two tokens for every model. For Ministral, the Hugging Face tokenizer gives
   the same ids as Mistral's reference tokenizer (`mistral_common`) on all 104 grid prompts.
3. *Fits on one L4 in bf16.* Language weights are 13.6–16.7 GiB, against the L4's 24 GB.
   Qwen3.5-9B is the tightest. vLLM's reported KV-cache capacity at load will confirm this
   (`Server.info`, logged in `validation.json`).
4. *Licence.* All candidates are Apache-2.0 and ungated, except Llama 3.1 8B. Llama is
   gated, needs Abdul's Hugging Face token, and was not screened.

**Shortlist: Qwen3.5-9B, Ministral-3-8B-Instruct-2512, OLMo-3-7B-Instruct.** These are the
newest dense ~8B instruct models in three families, and all three are ungated. Gemma-4-E4B
is held in reserve: it stores 8B parameters, but about 2.9B of those are per-layer
embeddings, so it is effectively a 4B model. Llama 3.1 8B is held in reserve until a token
is available; nothing newer at ≤ 9B exists in that family. Qwen3-8B is the fallback if
Qwen3.5-9B does not leave enough KV cache on the L4.

**Provisional preference.** Main model: Qwen3.5-9B, if it clears the bars and the
exactness check. It is the newest and most capable model at this size, and Qwen models are
widely used in multi-agent LLM work, which helps comparison. Second family: OLMo-3-7B. Its
weights, data and training code are all open, which suits a paper about auditing, and its
tokenizer passed every check Qwen did. Ministral-3-8B is the alternative second family. This
preference rests only on the facts above. Once the grid has run, the decision rule I suggest
is this. Among the models that pass exactness (≤ 0.01) and have p_valid that isn't tiny,
pick as main the one that is unsaturated in the most states, provided it meets both
responsiveness bars. The second family is the best remaining model from a different family.
If the grid shows Qwen3.5-9B saturated or unresponsive, this ranking should change.

**To run the GPU part** (about 10 L4-minutes per model expected; the worst case is capped
by container timeouts at $1.60 for three models, under the $3 cap):

    modal token set --token-id ... --token-secret ...      # the teammate's workspace
    python scripts/screen_models.py --dry-run               # labels, prompt counts, cost bound
    python scripts/screen_models.py                         # the three shortlisted models
    python scripts/screen_models.py --summarise-only        # rebuild this summary

**Prompt templates.** `swarm_audit/prompts/counts.txt` and `cheap_talk.txt` are drafts for
Abdul's review. They list open questions at the top: payoff formula vs table, last
payoff only, "unknown number of rounds", the agent's own past messages, and who receives
messages on a directed graph. Rendered examples for each model are in
`results/screening/prompts/`.
