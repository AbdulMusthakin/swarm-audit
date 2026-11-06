# Hand-off: model screening (choose the ~7B model)

For Claude Code, working in this repository. Read this file in full before starting.

**When:** run this on or after **Friday 6 November 2026**. That's when the Apart Research Swarm Alignment Hackathon sprint starts. Code written before Friday counts as pre-existing and must be declared. If you are run earlier, stop and tell Abdul.

The Warwick team chooses one idea together: Friday's workshop shortlists up to two, both are tried at 11:30 on Saturday, and one is chosen at 14:00. Run this either as a feasibility demo for Saturday's 11:30 slot, if this idea is shortlisted, or after it's chosen.

**Owner:** Abdul Musthakin. He owns the mathematics and all paper text, and is happy for the code to be AI-generated. Do not edit anything under `setup-paper/` or `report/`.

## 1. Context

The project tests whether auditing LLM agents one at a time predicts what a swarm of them does. The set-up paper is in `setup-paper/sections/`. Read `2-setup.tex` and `4-experiment.tex` for the definitions and notation. Use the same names in code where it is natural: `K_i`, `kappa_i`, `p_test`, `rho_i`, `q`, `pi_k`.

Key facts from the paper:

- **Game:** repeated procurement auction. Each agent bids H (high) or L (low) each round.
- **Signals:** on the *cheap-talk step*, each agent also emits one of three signals: "I'll bid high", "I'll bid low", or silence. Signals never affect payoffs.
- **Payoff:** agent i's payoff depends only on its own bid and f, the fraction of its in-neighbours who bid high. Counts exclude the agent itself.
  - `u(H, f) = uHL + f * (uHH - uHL)`
  - `u(L, f) = uLL + f * (uLH - uLL)`
  - Starting values: `(uLH, uHH, uLL, uHL) = (12, 10, 4, 0)`. These may be tuned later, so keep them as parameters.
- **What an agent sees in round t:**
  - its own previous actions
  - last round's payoff
  - in-neighbour action counts from earlier rounds
  - on the cheap-talk step, in-neighbour signal counts too

  It never sees a running total, and is never told the number of rounds T (T = 20). Round 1 is the prompt alone.
- **Every agent uses the same model and the same prompt template.** An agent differs only through its in-degree d.
- **Exactness matters.** The method needs the model's exact probability of each action given a prompt. These probabilities define the kernel K_i(· | h).

## 2. Goal

Screen 2–3 candidate open-weights instruct models of about 7–8B parameters. Produce the evidence Abdul needs to choose:

- the **main model**
- the **second model family**, which is conditional extra 2 and must come from a different family

Recommend a choice, but **do not decide**. Abdul decides.

## 3. Hard requirements (a candidate failing any of these is out)

1. **No reasoning before the answer.** The action must be the first generated token. Thinking models are acceptable only with thinking switched off. For example, Qwen3 has `enable_thinking=False` in the chat template: check the rendered prompt to confirm the empty think block.
2. **Single-token labels.** H and L, and the three signal labels, must each be a single token in that model's tokenizer, *at the position where they are generated*. Choose labels per model.
3. **Fits on one Modal L4 (24 GB) in bf16 under vLLM,** with room for the KV cache.
4. **Licence allows research use.** Gated models (Llama, Gemma) need Abdul's Hugging Face token and licence acceptance.

## 4. Candidates

Start from these families:

- Llama 3.1 8B Instruct
- Qwen3 8B (thinking off)
- Mistral 7B Instruct
- Gemma
- OLMo 2 7B Instruct

Before screening, check Hugging Face for a **newer version in each family at ≤ 9B** and use it if one exists. Abdul's list dates from mid-2026. Shortlist 3 candidates, from at least 2 families.

## 5. What to build

Put reusable pieces in the package, since the full harness will build on them.

### 5.1 `swarm_audit/game.py`
- The payoff function `u(action, f, params)`.
- A small dataclass for the four payoff parameters.

### 5.2 `swarm_audit/prompts/counts.txt` and `swarm_audit/prompts/cheap_talk.txt`
- Prompt templates for the two ladder steps.
- **The labels are template parameters**, because they depend on the tokenizer.
- The templates must:
  - describe the game and the payoff rule as numbers
  - show the history round by round, as specified in §1
  - end so that the next generated token is the action label
- **These are drafts for Abdul to review.** The prompt is part of the experimental design. Put a comment at the top of each file saying so.

### 5.3 `swarm_audit/ladder.py`
Renders an agent's observation and history into the template for a given step.

### 5.4 `swarm_audit/serving.py`
A Modal app that serves a given model with vLLM on an L4, with model weights cached in a Modal Volume. It exposes a batch function:

- **Input:** prompts, plus a list of allowed single-token ids.
- **Output:** for each prompt,
  - the exact probability of each allowed token, renormalised over the allowed set
  - `p_valid`, the total raw probability of the allowed tokens before renormalising

Getting exact probabilities for *specific* token ids from vLLM depends on the vLLM version. Options include:

- `prompt_logprobs` with each label appended
- processed logprobs together with `allowed_token_ids`

Whichever you use, **validate it.** On at least 10 prompts per model, compare against a plain `transformers` forward pass in bf16 and report the maximum absolute difference in renormalised P(H). Up to about 0.01 is acceptable, given bf16 numerics. Log every model call to `logs/`, which git ignores.

### 5.5 `scripts/screen_tokenizers.py`
CPU only, no GPU. For each candidate and each candidate label set:

- **Label sets to try:** `H/L`, ` H/ L`, `high/low`, ` high/ low`, `A/B`, and for signals `1/2/3` and words.
- **The check:** `encode(prompt + label) == encode(prompt) + [exactly one token]`, using the full chat-templated prompt.
- **Output:** report which label sets pass for each model.

### 5.6 `scripts/screen_models.py` and `configs/screening.yaml`
Run the screening grid on each shortlisted model.

**Counts step** (about 72 prompts per model):
- in-degree d ∈ {2, 4, 10}
- f ∈ {0, 0.5, 1}
- own last action ∈ {H, L}
- last round's payoff computed with `game.py`
- each state at history length 1, plus 3 random fillers at length 5 (random earlier rounds)

**Cheap-talk step** (about 32 prompts per model):
- d = 4, with the action counts fixed at 2 high and 2 low
- signal counts in four variants: all "high", all "low", all silent, 2 high + 2 low
- own last action ∈ {H, L}
- the same length-1 and length-5 variants

### 5.7 Tests (`tests/`, pytest)
Test:
- the payoff corners: `u(H, 1) = uHH`, `u(H, 0) = uHL`, `u(L, 1) = uLH`, `u(L, 0) = uLL`
- that the prompt renders correctly for both steps, including round 1 as an empty history
- that the tokenizer check works on a known tokenizer

Agent "unit tests", meaning K_i and kappa_i, are a different thing and will live in `swarm_audit/audit.py` later. Don't confuse the two.

## 6. Outputs

Write to `results/screening/`.

**`screening.csv`:** one row per (model, prompt), with the state, P(H), `p_valid` and entropy.

**`summary.md`:** one table per model, with:
- **Saturation:** the fraction of states where 0.02 ≤ P(H) ≤ 0.98.
- **Neighbour responsiveness:** |P(H | f=1) − P(H | f=0)|, for each own action and d.
- **Signal responsiveness:** max − min of P(H) across the four signal variants, at fixed action counts.
- **History sensitivity:** the spread of P(H) across the length-5 fillers.
- **p_valid:** the mean and the minimum.
- **The exactness check:** the maximum difference from `transformers`.
- **The chosen labels.**

Finish with a **recommendation**: main model and second family, with a one-paragraph reason.

**Suggested bars.** These are heuristics; report the numbers whatever they are.
- At least half the states are unsaturated.
- Neighbour responsiveness is at least 0.1.
- Signal responsiveness is at least 0.05.
- `p_valid` isn't tiny.

A model failing these would make the swarm runs uninformative:
- **Saturated:** runs lock into consensus.
- **Unresponsive:** Δ ≈ 0 for a trivial reason.

## 7. Limits and stop conditions

- **Cost cap: $3 of Modal GPU time** for all screening.
- **Account:** a teammate's Modal account, using leftover credits from a previous sprint. Abdul will say whose, and whether you get a workspace invite or the teammate runs the commands. Don't use Abdul's own account without asking.
- **Weights:** the download dominates the cost, so cache them in a Modal Volume.
- **Not Adaption.** The sprint's Adaption credits aren't used here, because Abdul believes Adaption's API doesn't return exact token log-probabilities.
- **Time box: about 2 hours.** Stop and report what you have.
- **Stop and ask Abdul if:**
  - Modal is not authenticated (`modal token`)
  - the Hugging Face token or a licence acceptance is missing
  - a cost would exceed the cap
  - no candidate passes the hard requirements
- **Don't commit or push unless Abdul asks.** If you do commit, it must be dated during the sprint.
- **No invented numbers.** Every figure in `summary.md` must come from an actual run.

## 8. Done when

- [ ] Tests pass.
- [ ] Tokenizer screen run; labels chosen for each shortlisted model.
- [ ] Exact-probability method validated against `transformers`.
- [ ] Screening grid run on 2–3 models, with `screening.csv` and `summary.md` written.
- [ ] Recommendation given; prompt templates flagged for Abdul's review.
