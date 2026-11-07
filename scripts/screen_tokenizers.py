"""Tokenizer screen (CPU only): which label sets are single tokens for each candidate model.

For every model in configs/screening.yaml and every candidate label set, render all
screening-grid prompts with the model's chat template and check
    encode(prompt + label) == encode(prompt) + [exactly one token]
at the position where the label is generated (actions after "Bid:", signals after
"<action label>\\nMessage:"). Also records the chat template's thinking switch and the
rendered prompt lengths.

Writes results/screening/tokenizers.json and results/screening/tokenizers.md, and an
example prompt per model to results/screening/prompts/ for review.

    python scripts/screen_tokenizers.py [--models qwen3.5-9b olmo-3-7b] [--config ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit.screening import (  # noqa: E402
    build_grid, build_prompts, choose_labels, labels_from_sets, load_config,
    load_tokenizer, payoff_params, screen_label_sets, signal_prefix_text,
)

OUT = REPO / "results" / "screening"


def param_counts(model_cfg: dict) -> dict:
    """Parameter counts (billions) by part, from the safetensors headers (no weights
    downloaded), and the bf16 size of the parts vLLM loads for text-only use."""
    import re

    from huggingface_hub import get_safetensors_metadata

    md = get_safetensors_metadata(model_cfg["hf_id"], revision=model_cfg.get("revision"))
    parts: dict[str, int] = {}
    for fm in md.files_metadata.values():
        for name, t in fm.tensors.items():
            n = 1
            for dim in t.shape:
                n *= dim
            part = ("vision" if re.search(r"vision|visual|multi_modal_projector|embed_vision", name)
                    else "audio" if "audio" in name
                    else "per_layer_embeddings" if "per_layer" in name
                    else "mtp" if re.search(r"(^|\.)mtp\.", name)
                    else "language")
            parts[part] = parts.get(part, 0) + n
    total = sum(parts.values())
    return {"billion": {k: round(v / 1e9, 3) for k, v in sorted(parts.items())},
            "total_billion": round(total / 1e9, 3),
            "bf16_gib_language": round((parts.get("language", 0) + parts.get(
                "per_layer_embeddings", 0)) * 2 / 2**30, 1),
            "bf16_gib_total": round(total * 2 / 2**30, 1)}


def screen_model(model_cfg: dict, cfg: dict, states, params) -> dict:
    res: dict = {"name": model_cfg["name"], "hf_id": model_cfg["hf_id"],
                 "revision": model_cfg.get("revision")}
    try:
        tok = load_tokenizer(model_cfg)
    except Exception as e:  # gated repo, missing token, network
        res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        return res

    try:
        res["params"] = param_counts(model_cfg)
    except Exception as e:
        res["params"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    screen = screen_label_sets(tok, model_cfg, cfg, states, params)
    res["actions"] = {n: {"ok": c.ok, "ids": c.ids, "problems": c.problems[:3]}
                      for n, c in screen["actions"].items()}
    res["signals"] = {n: {"ok": c.ok, "ids": c.ids, "problems": c.problems[:3]}
                      for n, c in screen["signals"].items()}
    res["signals_checked_after"] = screen["signal_checked_after"]
    choice = choose_labels(cfg, model_cfg, screen)
    res["chosen"] = choice

    # Rendered prompts with the chosen (or default) labels: lengths, thinking check, examples.
    a_set, s_set = choice or (cfg["label_preference"]["actions"][0],
                              cfg["label_preference"]["signals"][0])
    labels = labels_from_sets(cfg, a_set, s_set)
    prompts = build_prompts(tok, model_cfg, states, labels, params)
    lengths = [len(p.ids) for p in prompts]
    res["prompt_tokens"] = {"min": min(lengths), "max": max(lengths)}
    res["chat_kwargs"] = model_cfg.get("chat_kwargs", {})
    first = prompts[0].text
    res["think_block_in_prompt"] = "<think>" in first
    res["empty_think_block"] = "<think>\n\n</think>" in first
    res["template_mentions_thinking"] = "think" in str(tok.chat_template or "")

    ex_dir = OUT / "prompts"
    ex_dir.mkdir(parents=True, exist_ok=True)
    picks = [p for p in prompts if p.state.filler in (None, 0)]
    examples = [picks[0]] + [p for p in picks if p.state.length > 1][:1]
    cheap = [p for p in picks if p.state.step == "cheap_talk"]
    examples += [cheap[0]] + [p for p in cheap if p.state.length > 1][:1]
    with open(ex_dir / f"{model_cfg['name']}.txt", "w") as fh:
        fh.write(f"# {model_cfg['hf_id']} @ {model_cfg.get('revision')}\n"
                 f"# labels: actions {a_set}, signals {s_set}. Exact prompt text, chat template"
                 f" included; the next token is the label.\n")
        for p in examples:
            fh.write(f"\n===== {p.state.prompt_id} ({len(p.ids)} tokens) =====\n{p.text}")
            if p.state.step == "cheap_talk":
                sp = signal_prefix_text(p, labels, params, labels.H)
                fh.write(f"\n----- signal position, after bid {labels.H!r} -----\n"
                         f"...{sp[-40:]}")
            fh.write("\n")
    return res


def write_markdown(results: list[dict], cfg: dict) -> str:
    act_names = list(cfg["label_sets"]["actions"])
    sig_names = list(cfg["label_sets"]["signals"])
    lines = ["# Tokenizer screen", "",
             "Generated by `scripts/screen_tokenizers.py` from `configs/screening.yaml`.",
             "A set passes if every label is one token, with the same id, at its generation",
             "position in every screening-grid prompt (`encode(prompt + label) == encode(prompt)"
             " + [one token]`, full chat-templated prompt). `_` stands for a leading space.", "",
             "| model | " + " | ".join(act_names) + " | " + " | ".join(sig_names)
             + " | chosen | prompt tokens | thinking off |",
             "|---" * (len(act_names) + len(sig_names) + 4) + "|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | not screened: {r['error']} |")
            continue
        cells = ["pass" if r["actions"][n]["ok"] else "FAIL" for n in act_names]
        cells += ["pass" if r["signals"][n]["ok"] else "FAIL" for n in sig_names]
        chosen = " + ".join(r["chosen"]) if r["chosen"] else "none"
        think = ("empty think block" if r["empty_think_block"]
                 else "no think block" if not r["think_block_in_prompt"] else "THINKING ON")
        pt = r["prompt_tokens"]
        lines.append(f"| {r['name']} | " + " | ".join(cells)
                     + f" | {chosen} | {pt['min']}–{pt['max']} | {think} |")
    lines += ["", "Signal sets were checked after the action labels in the "
              "`signals_checked_after` field of tokenizers.json.", "", "## Failures", ""]
    for r in results:
        for kind in ("actions", "signals"):
            for n, c in r.get(kind, {}).items():
                if not c["ok"]:
                    lines.append(f"- {r['name']}, {kind} `{n}`: {'; '.join(c['problems'])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "screening.yaml"))
    ap.add_argument("--models", nargs="*", help="model names (default: all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    params = payoff_params(cfg)
    states = build_grid(cfg)
    models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]

    results = []
    for m in models:
        print(f"screening {m['name']} ({m['hf_id']}) ...", flush=True)
        r = screen_model(m, cfg, states, params)
        results.append(r)
        print("   ", r.get("error") or f"chosen labels: {r['chosen']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tokenizers.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "tokenizers.md").write_text(write_markdown(results, cfg))
    print(f"wrote {OUT / 'tokenizers.json'} and {OUT / 'tokenizers.md'}")


if __name__ == "__main__":
    main()
