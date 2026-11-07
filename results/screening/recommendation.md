## Recommendation (for Abdul to decide)

**Status, 7 Nov 2026, 14:50 UTC.** All four models were screened on Modal L4s (vLLM
0.32, bf16) with the full grid of 104 prompts. Exactness checks:

- **Qwen3.5-9B and OLMo-3-7B:** both checks ran (against transformers, and against
  vLLM's second method).
- **Ministral-3-8B:** the transformers check ran. No L4 became free for the cross-method
  check before the time box ended.
- **Llama 3.1 8B:** neither check ran, for the same reason.

The runs were finished from the per-call logs (`SWARM_AUDIT_RESUME=1`), so no model was
queried twice.

**Hard requirements.**

1. *No reasoning before the action:* met by construction. The reply is prefilled with
   `Bid:`, Qwen's empty think block is confirmed, and Gemma and Llama open no thought
   channel.
2. *Single-token labels:* every model uses ` H`/` L` and ` high`/` low`/` none`.
3. *Fits on one L4 in bf16 with KV cache:* met by all four. KV capacity at load was
   22,032 tokens (Ministral) to 62,832 tokens (Qwen3.5), against prompts of at most
   about 680 tokens.
4. *Licence:* Apache-2.0 for three models. Llama is under the Llama 3.1 licence, and
   Abdul's access was approved.

**Exactness.** No model meets the suggested 0.01. The cause is bf16, not the way the
probabilities are read out.

- Every vLLM-vs-transformers difference is a whole number of steps of 0.125 in the
  log-odds of H (at most 2 steps). 0.125 is the bf16 spacing for logits between 16 and
  32. So both engines produce the same logits up to bf16 rounding, and P(H) itself
  lives on a grid about 3 percentage points wide near 0.5.
- vLLM's two read-out methods agree to within 7e-7 for Qwen3.5.
- For OLMo they differ by one step (0.022). OLMo runs on vLLM's transformers backend,
  so a different batch can flip one rounding.

Before the main runs, I suggest:

- run vLLM with `VLLM_BATCH_INVARIANT=1`, so a history gets the same K_i whether it is
  queried in a unit test or in a swarm batch;
- either compute the LM head in fp32, or accept a tolerance of about two bf16 steps.

Which one is a design decision for Abdul.

**Behaviour.** P(H) by own last bid is in `screening.csv`.

- **Qwen3.5-9B:** unsaturated everywhere, but low P(H) overall (0.07–0.50). Neighbour
  responsiveness passes only barely (0.149). It **fails the signal bar** (0.020).
- **Ministral-3-8B:** unsaturated (0.98) with strong neighbour responsiveness (0.29). It
  **fails the signal bar** (0.027), and its bid p_valid dips to 0.94.
- **OLMo-3-7B:** passes every bar on paper (unsaturated 0.66, neighbour 0.41, signal
  0.11). But on the cheap-talk step P(H) is 0.000 in all 16 states after its own low
  bid. Low is absorbing on the key step, so swarm runs would lock into all-low
  consensus, which is the hand-off's "saturated" failure.
- **Llama 3.1 8B:** the only model that **passes every behavioural bar without
  saturating anywhere**: unsaturated 1.00, neighbour 0.24, signal 0.051 (just over 0.05),
  p_valid 0.999. Its exactness checks were not run.

**Recommendation.**

- **Main model: Llama 3.1 8B Instruct.** Run its transformers check first; I expect the
  same bf16-step pattern as the others. Its signal responsiveness is only just over the
  bar, so watch it in the Friday-style pilot.
- **Second family: Ministral-3-8B-Instruct-2512.** It is responsive and unsaturated, but
  weak on signals.
- **Alternative second family: OLMo-3-7B,** if Abdul prefers a fully open model and can
  accept low being absorbing on the cheap-talk step. A prompt change might fix that, but
  the prompt is still Abdul's to review.

Qwen3.5-9B, the provisional favourite before the GPU runs, is not recommended. It is the
least responsive to neighbours, fails the signal bar, and stays at low P(H) throughout.

**Reproduce:**

    python scripts/screen_tokenizers.py --via-modal       # SWARM_AUDIT_HF_SECRET=huggingface
    python scripts/screen_models.py --models <name> --out results/screening/runs/<name>
    python scripts/screen_models.py --merge results/screening/runs/*

**Prompt templates.** `swarm_audit/prompts/counts.txt` and `cheap_talk.txt` are drafts for
Abdul's review, with the open questions listed at the top. Every number above depends on
them.
