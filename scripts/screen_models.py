"""Model screening grid: P(H) on the counts and cheap-talk steps, exactness check, summary.

For each model, render the grid in configs/screening.yaml, get exact label probabilities
from vLLM on a Modal L4, compare a subset against a plain transformers forward pass, and
write results/screening/screening.csv, validation.json and summary.md.

    python scripts/screen_models.py --dry-run              # no GPU: labels, prompt sizes, cost
    python scripts/screen_models.py                         # shortlisted models, on Modal
    python scripts/screen_models.py --models olmo-3-7b      # chosen models, on Modal
    python scripts/screen_models.py --summarise-only        # rebuild summary.md from the CSV

    # CPU smoke test of the pipeline with a small model. Not a screening result: writes
    # to --out, and the "exactness check" compares transformers with itself.
    python scripts/screen_models.py --backend local --local-model Qwen/Qwen3-0.6B \\
        --out /tmp/smoke

Modal must be authenticated, and the account is a teammate's (hand-off section 7).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit.screening import (  # noqa: E402
    Prompt, build_grid, build_prompts, choose_labels, labels_from_sets, load_config,
    load_tokenizer, payoff_params, screen_label_sets, signal_prefix_text, summarise_model,
)
from swarm_audit.tokens import encode  # noqa: E402

CSV_FIELDS = [
    "model", "hf_id", "step", "prompt_id", "state_id", "d", "f", "own_last",
    "signal_variant", "length", "filler", "n_tokens", "p_H", "p_L", "p_valid",
    "entropy_bits", "logprob_H", "logprob_L",
    "p_sig_high_after_H", "p_sig_low_after_H", "p_sig_none_after_H", "p_valid_sig_after_H",
    "p_sig_high_after_L", "p_sig_low_after_L", "p_sig_none_after_L", "p_valid_sig_after_L",
    "action_labels", "signal_labels", "method",
]


# ----------------------------------------------------------------------- backends


class ModalBackend:
    """vLLM on a Modal L4, plus the transformers reference."""

    def __init__(self, model_cfg: dict, serving_cfg: dict):
        from swarm_audit import serving

        self.s = serving
        self.m = model_cfg
        eng = {"max_model_len": serving_cfg["max_model_len"],
               "gpu_memory_utilization": serving_cfg["gpu_memory_utilization"],
               "enforce_eager": serving_cfg["enforce_eager"]}
        eng.update(model_cfg.get("engine_kwargs", {}))
        self.server = serving.Server(hf_id=model_cfg["hf_id"], revision=model_cfg["revision"],
                                     engine_kwargs_json=json.dumps(eng))
        self.meta = {"model": model_cfg["name"], "hf_id": model_cfg["hf_id"],
                     "revision": model_cfg["revision"]}

    def prepare(self) -> dict:
        out = self.s.call_logged(self.s.download_weights, self.m["hf_id"], self.m["revision"],
                                 log_name=self.m["name"], meta={**self.meta, "call": "download"})
        info = self.s.call_logged(self.server.info, log_name=self.m["name"],
                                  meta={**self.meta, "call": "info"})
        return {"download": out, "server": info}

    def label_probs(self, prompts: list[Prompt | tuple[str, list[int]]], ids: list[int],
                    method: str, kind: str) -> list[dict]:
        pids, pts = _unzip(prompts)
        return self.s.call_logged(self.server.label_logprobs, pts, ids, method=method,
                                  log_name=self.m["name"], prompt_ids=pids, prompts=pts,
                                  meta={**self.meta, "call": kind, "method": method,
                                        "allowed_token_ids": ids})

    def reference(self, batches: list[tuple[list, list[int]]], kind: str) -> dict:
        flat = [(_unzip(p)[1], ids) for p, ids in batches]
        pids = [pid for p, _ in batches for pid in _unzip(p)[0]]
        pts = [pt for p, _ in flat for pt in p]
        return self.s.call_logged(self.s.hf_label_logprobs, self.m["hf_id"], self.m["revision"],
                                  flat, log_name=self.m["name"], prompt_ids=pids, prompts=pts,
                                  meta={**self.meta, "call": kind})


class LocalBackend:
    """transformers on CPU, for smoke tests of the pipeline only."""

    def __init__(self, hf_id: str):
        from swarm_audit import serving

        self.s = serving
        self.hf_id = hf_id
        self.model = serving.load_hf_model(hf_id, None, device="cpu", dtype="float32")

    def prepare(self) -> dict:
        return {"local": self.hf_id}

    def label_probs(self, prompts, ids, method, kind):
        pids, pts = _unzip(prompts)
        return self.s.call_logged(self.s.hf_logprobs, self.model, pts, ids,
                                  log_name="smoke_" + self.hf_id.replace("/", "_"),
                                  prompt_ids=pids, prompts=pts,
                                  meta={"model": self.hf_id, "call": kind, "method": "local-hf"})

    def reference(self, batches, kind):
        return {"results": [self.label_probs(p, ids, "local-hf", kind) for p, ids in batches],
                "note": "local smoke test: reference is the same transformers model"}


def _unzip(prompts) -> tuple[list[str], list[list[int]]]:
    pids, pts = [], []
    for p in prompts:
        if isinstance(p, Prompt):
            pids.append(p.state.prompt_id)
            pts.append(p.ids)
        else:
            pids.append(p[0])
            pts.append(p[1])
    return pids, pts


# ---------------------------------------------------------------------- one model


def screen_one(model_cfg: dict, cfg: dict, backend: Any, method: str) -> tuple[list[dict], dict]:
    params = payoff_params(cfg)
    states = build_grid(cfg)
    tok = load_tokenizer(model_cfg)
    screen = screen_label_sets(tok, model_cfg, cfg, states, params)
    choice = choose_labels(cfg, model_cfg, screen)
    if choice is None:
        raise SystemExit(f"{model_cfg['name']}: no label set passes the tokenizer screen")
    a_set, s_set = choice
    labels = labels_from_sets(cfg, a_set, s_set)
    act_ids = list(screen["actions"][a_set].ids)
    sig_ids = list(screen["signals"][s_set].ids)

    prompts = build_prompts(tok, model_cfg, states, labels, params)
    cheap = [p for p in prompts if p.state.step == "cheap_talk"]
    sig_prompts: dict[str, list[tuple[str, list[int]]]] = {}
    for a, a_label in zip("HL", labels.actions):
        sig_prompts[a] = [(f"{p.state.prompt_id}_after{a}",
                           encode(tok, signal_prefix_text(p, labels, params, a_label)))
                          for p in cheap]

    prep = backend.prepare()
    act = backend.label_probs(prompts, act_ids, method, "actions")
    sig = {a: backend.label_probs(sp, sig_ids, method, f"signals_after_{a}")
           for a, sp in sig_prompts.items()}

    rows = []
    sig_index = {p.state.prompt_id: i for i, p in enumerate(cheap)}
    for p, r in zip(prompts, act):
        s = p.state
        row = {
            "model": model_cfg["name"], "hf_id": model_cfg["hf_id"], "step": s.step,
            "prompt_id": s.prompt_id, "state_id": s.state_id, "d": s.d, "f": s.f,
            "own_last": s.own_last, "signal_variant": s.signal_variant or "",
            "length": s.length, "filler": "" if s.filler is None else s.filler,
            "n_tokens": len(p.ids), "p_H": r["probs"][0], "p_L": r["probs"][1],
            "p_valid": r["p_valid"], "entropy_bits": r["entropy_bits"],
            "logprob_H": r["logprobs"][0], "logprob_L": r["logprobs"][1],
            "action_labels": "/".join(labels.actions), "signal_labels": "/".join(labels.signals),
            "method": method,
        }
        if s.prompt_id in sig_index:
            i = sig_index[s.prompt_id]
            for a in "HL":
                sr = sig[a][i]
                for name, pr in zip(("high", "low", "none"), sr["probs"]):
                    row[f"p_sig_{name}_after_{a}"] = pr
                row[f"p_valid_sig_after_{a}"] = sr["p_valid"]
        rows.append(row)

    # Exactness check on a spread-out subset: vLLM vs transformers (and vLLM's second method).
    v = cfg["validation"]
    counts_p = [p for p in prompts if p.state.step == "counts"]
    sub_c = counts_p[:: max(1, len(counts_p) // v["n_counts"])][: v["n_counts"]]
    sub_t = cheap[:: max(1, len(cheap) // v["n_cheap_talk"])][: v["n_cheap_talk"]]
    sub_a = sub_c + sub_t
    sub_s = [sig_prompts["H"][sig_index[p.state.prompt_id]] for p in sub_t]
    ref = backend.reference([(sub_a, act_ids), (sub_s, sig_ids)], "reference")
    ref_a, ref_s = ref["results"]
    by_pid = {r["prompt_id"]: r for r in rows}
    sig_by_pid = {sig_prompts["H"][i][0]: sig["H"][i] for i in range(len(cheap))}
    comp = []
    for p, rr in zip(sub_a, ref_a):
        vr = by_pid[p.state.prompt_id]
        comp.append({"prompt_id": p.state.prompt_id, "kind": "action",
                     "vllm_p_H": vr["p_H"], "hf_p_H": rr["probs"][0],
                     "abs_diff_p_H": abs(vr["p_H"] - rr["probs"][0]),
                     "abs_diff_p_valid": abs(vr["p_valid"] - rr["p_valid"])})
    for (pid, _), rr in zip(sub_s, ref_s):
        vr = sig_by_pid[pid]
        comp.append({"prompt_id": pid, "kind": "signal",
                     "vllm_probs": vr["probs"], "hf_probs": rr["probs"],
                     "abs_diff_max": max(abs(a - b) for a, b in zip(vr["probs"], rr["probs"])),
                     "abs_diff_p_valid": abs(vr["p_valid"] - rr["p_valid"])})
    other = "allowed_token_ids" if method == "logprob_token_ids" else "logprob_token_ids"
    cross = None
    if isinstance(backend, ModalBackend):
        alt = backend.label_probs(sub_a, act_ids, other, "cross_method")
        cross = max(abs(by_pid[p.state.prompt_id]["p_H"] - r["probs"][0])
                    for p, r in zip(sub_a, alt))
    act_diffs = [c["abs_diff_p_H"] for c in comp if c["kind"] == "action"]
    sig_diffs = [c["abs_diff_max"] for c in comp if c["kind"] == "signal"]
    validation = {
        "model": model_cfg["name"], "hf_id": model_cfg["hf_id"],
        "revision": model_cfg.get("revision"), "method": method,
        "n_action_prompts": len(act_diffs), "n_signal_prompts": len(sig_diffs),
        "max_abs_diff_p_H": max(act_diffs), "max_abs_diff_signal": max(sig_diffs),
        "max_abs_diff_p_valid": max(c["abs_diff_p_valid"] for c in comp),
        "tolerance": v["tolerance"], "passes": max(act_diffs) <= v["tolerance"],
        f"max_abs_diff_p_H_vs_{other}": cross,
        "labels": {"actions": a_set, "signals": s_set, "action_ids": act_ids,
                   "signal_ids": sig_ids},
        "prepare": prep,
        "reference_env": {k: ref.get(k) for k in ("transformers", "torch", "load_seconds", "note")},
        "comparisons": comp,
    }
    return rows, validation


# ------------------------------------------------------------------------ summary


def _fmt(x: float | None, nd: int = 3) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def write_summary(out_dir: Path, cfg: dict) -> None:
    csv_path = out_dir / "screening.csv"
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    val_path = out_dir / "validation.json"
    validations = json.loads(val_path.read_text()) if val_path.exists() else {}
    tok_path = out_dir / "tokenizers.json"
    toks = {t["name"]: t for t in json.loads(tok_path.read_text())} if tok_path.exists() else {}

    L = ["# Model screening: summary", "",
         "Generated by `scripts/screen_models.py` from `screening.csv`, `validation.json` and",
         "`tokenizers.json` in this folder. Every number below comes from those files.",
         "The prompt templates (`swarm_audit/prompts/*.txt`) are drafts awaiting Abdul's review.",
         "", "## Tokenizer screen", "",
         "See `tokenizers.md` for every label set. Chosen labels (first passing set in the",
         "preference order of `configs/screening.yaml`; `_` is a leading space):", ""]
    for m in cfg["models"]:
        t = toks.get(m["name"])
        if t is None:
            continue
        ch = " + ".join(t["chosen"]) if t.get("chosen") else t.get("error", "none passed")
        L.append(f"- **{m['name']}** (`{m['hf_id']}`): {ch}")

    L += ["", "## Candidates", "",
          "The newest instruct model at ≤ 9B in each family (Hugging Face, 7 Nov 2026), plus the"
          " versions on the original list as fallbacks. Sizes are from the safetensors headers;"
          " vLLM loads the language part only (`language_model_only`). An L4 has 24 GB.", "",
          "| model | family | licence | params (B): language / total | bf16 GiB, language |"
          " thinking off | shortlisted |", "|---|---|---|---|---|---|---|"]
    for m in cfg["models"]:
        t = toks.get(m["name"], {})
        pr = t.get("params") or {}
        lang = pr.get("billion", {}).get("language", 0) + pr.get("billion", {}).get(
            "per_layer_embeddings", 0)
        size = (f"{lang:.2f} / {pr['total_billion']:.2f}"
                if "billion" in pr else "–")
        gib = f"{pr['bf16_gib_language']:.1f}" if "bf16_gib_language" in pr else "–"
        think = ("–" if "error" in t or not t else "empty think block" if t.get("empty_think_block")
                 else "no think block" if not t.get("think_block_in_prompt") else "THINKING ON")
        L.append(f"| {m['name']} (`{m['hf_id']}`) | {m['family']} | {m['licence']} | {size} | "
                 f"{gib} | {think} | {'yes' if m.get('shortlist') else 'no'} |")

    names = list(dict.fromkeys(r["model"] for r in rows))
    L += ["", "## Screening grid", ""]
    if not names:
        L += ["**Not run yet.** No model has been run on a GPU, so there are no saturation,",
              "responsiveness, p_valid or exactness numbers. See the recommendation below for",
              "what is blocking the run and how to start it."]
    bars = []
    for name in names:
        mr = [r for r in rows if r["model"] == name]
        s = summarise_model(mr)
        val = validations.get(name, {})
        L += [f"### {name} (`{mr[0]['hf_id']}`)", "",
              f"Labels: actions `{mr[0]['action_labels']!r}`, signals `{mr[0]['signal_labels']!r}`."
              f" Method: `{mr[0]['method']}`.", "",
              "| metric | value |", "|---|---|",
              f"| unsaturated (0.02 ≤ P(H) ≤ 0.98), counts step | {_fmt(s.get('unsaturated_counts'))} of {s.get('n_counts')} prompts |",
              f"| unsaturated, cheap-talk step | {_fmt(s.get('unsaturated_cheap_talk'))} of {s.get('n_cheap_talk')} prompts |",
              f"| unsaturated, all | {_fmt(s.get('unsaturated_all'))} |",
              f"| p_valid (bid) mean / min | {_fmt(s['p_valid_mean'])} / {_fmt(s['p_valid_min'])} |",
              f"| p_valid (message) mean / min | {_fmt(s.get('p_valid_sig_mean'))} / {_fmt(s.get('p_valid_sig_min'))} |",
              f"| exactness: max abs diff in P(H), vLLM vs transformers | {_fmt(val.get('max_abs_diff_p_H'), 4)} ({val.get('n_action_prompts', 0)} prompts) |",
              f"| exactness: max abs diff in message probs | {_fmt(val.get('max_abs_diff_signal'), 4)} ({val.get('n_signal_prompts', 0)} prompts) |",
              ""]
        neigh = s["neighbour"]
        L += ["Neighbour responsiveness |P(H | f=1) − P(H | f=0)| (counts step; length 5 is the",
              "mean over the fillers):", "", "| own last | d | length 1 | length 5 (mean) |",
              "|---|---|---|---|"]
        for a in ("H", "L"):
            for d in sorted({n["d"] for n in neigh}):
                l1 = [n["delta"] for n in neigh if n["own_last"] == a and n["d"] == d and n["length"] == 1]
                l5 = [n["delta"] for n in neigh if n["own_last"] == a and n["d"] == d and n["length"] > 1]
                L.append(f"| {a} | {d} | {_fmt(l1[0] if l1 else None)} | "
                         f"{_fmt(sum(l5) / len(l5) if l5 else None)} |")
        sig = s["signal"]
        L += ["", "Signal responsiveness, max − min of P(H) over the four message variants",
              "(cheap-talk step, 2 high and 2 low of 4 neighbours):", "",
              "| own last | length 1 | length 5 (mean) |", "|---|---|---|"]
        for a in ("H", "L"):
            l1 = [g["spread"] for g in sig if g["own_last"] == a and g["length"] == 1]
            l5 = [g["spread"] for g in sig if g["own_last"] == a and g["length"] > 1]
            L.append(f"| {a} | {_fmt(l1[0] if l1 else None)} | {_fmt(sum(l5) / len(l5) if l5 else None)} |")
        hist = list(s["history"].values())
        L += ["", f"History sensitivity (spread of P(H) across the {cfg['grid']['fillers']} "
              f"length-{cfg['grid']['filler_length']} fillers of each state): mean "
              f"{_fmt(sum(hist) / len(hist))}, max {_fmt(max(hist))} over {len(hist)} states.", ""]
        n1 = [n["delta"] for n in neigh if n["length"] == 1]
        s1 = [g["spread"] for g in sig if g["length"] == 1]
        bars.append((name, s.get("unsaturated_all"), sum(n1) / len(n1), sum(s1) / len(s1),
                     s["p_valid_mean"], val.get("max_abs_diff_p_H")))
    if bars:
        L += ["## Suggested bars", "",
              "Heuristics from the hand-off: unsaturated ≥ 0.5, neighbour responsiveness ≥ 0.1,"
              " signal responsiveness ≥ 0.05, p_valid not tiny, exactness ≤ 0.01. Responsiveness"
              " here is the mean of the length-1 values above.", "",
              "| model | unsaturated | neighbour | signal | p_valid mean | exactness |",
              "|---|---|---|---|---|---|"]
        for name, un, ne, si, pv, ex in bars:
            mark = lambda ok: "" if ok else " ✗"  # noqa: E731
            L.append(f"| {name} | {_fmt(un)}{mark(un is not None and un >= 0.5)} | "
                     f"{_fmt(ne)}{mark(ne >= 0.1)} | {_fmt(si)}{mark(si >= 0.05)} | "
                     f"{_fmt(pv)} | {_fmt(ex, 4)}{mark(ex is not None and ex <= 0.01)} |")
    rec = out_dir / "recommendation.md"
    L += ["", rec.read_text().strip() if rec.exists() else "## Recommendation\n\nNot written yet."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "screening.yaml"))
    ap.add_argument("--models", nargs="*", help="model names (default: shortlist)")
    ap.add_argument("--backend", choices=["modal", "local"], default="modal")
    ap.add_argument("--local-model", help="HF id for --backend local (smoke test)")
    ap.add_argument("--out", default=str(REPO / "results" / "screening"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--merge", nargs="*", metavar="DIR",
                    help="merge screening.csv and validation.json from these run folders "
                         "(one per model, run in parallel) into --out, then summarise")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows: list[dict] = []
        vals: dict = {}
        for d in map(Path, args.merge):
            rows += list(csv.DictReader(open(d / "screening.csv")))
            vals.update(json.loads((d / "validation.json").read_text()))
        with open(out_dir / "screening.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        (out_dir / "validation.json").write_text(json.dumps(vals, indent=2, default=str) + "\n")
        args.summarise_only = True
    if args.summarise_only:
        write_summary(out_dir, cfg)
        print(f"wrote {out_dir / 'summary.md'}")
        return

    srv = cfg["serving"]
    if args.backend == "local":
        if not args.local_model:
            raise SystemExit("--backend local needs --local-model")
        models = [{"name": "smoke-" + args.local_model.split("/")[-1], "hf_id": args.local_model,
                   "revision": None, "chat_kwargs": {"enable_thinking": False}}]
    else:
        models = [m for m in cfg["models"]
                  if (m["name"] in args.models if args.models else m.get("shortlist"))]

    # Cost guard: worst case is every container running to its timeout.
    worst = len(models) * 2 * srv["timeout_minutes"] / 60 * srv["l4_usd_per_hour"]
    print(f"{len(models)} model(s): {[m['name'] for m in models]}")
    print(f"worst-case GPU cost ${worst:.2f} (cap ${srv['cost_cap_usd']:.2f}); "
          f"expected well below, about 10 L4-minutes per model")
    if args.backend == "modal" and worst > srv["cost_cap_usd"]:
        raise SystemExit("worst-case cost exceeds the cap: screen fewer models per run")

    if args.dry_run:
        params = payoff_params(cfg)
        states = build_grid(cfg)
        for m in models:
            tok = load_tokenizer(m)
            screen = screen_label_sets(tok, m, cfg, states, params)
            ch = choose_labels(cfg, m, screen)
            print(f"  {m['name']}: {len(states)} prompts, labels {ch}")
        return

    all_rows: list[dict] = []
    csv_path = out_dir / "screening.csv"
    if csv_path.exists():  # keep other models' rows
        done = {m["name"] for m in models}
        all_rows = [r for r in csv.DictReader(open(csv_path)) if r["model"] not in done]
    val_path = out_dir / "validation.json"
    validations = json.loads(val_path.read_text()) if val_path.exists() else {}

    def run_all() -> None:
        for m in models:
            print(f"== {m['name']} ({m['hf_id']})", flush=True)
            backend = (LocalBackend(m["hf_id"]) if args.backend == "local"
                       else ModalBackend(m, srv))
            method = srv["method"] if args.backend == "modal" else "local-transformers"
            rows, val = screen_one(m, cfg, backend, method)
            all_rows.extend(rows)
            validations[m["name"]] = val
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(all_rows)
            val_path.write_text(json.dumps(validations, indent=2, default=str) + "\n")
            print(f"   max |dP(H)| vs transformers: {val['max_abs_diff_p_H']:.4g}", flush=True)

    if args.backend == "modal":
        import modal

        from swarm_audit.serving import app

        with modal.enable_output(), app.run():
            run_all()
    else:
        run_all()
    write_summary(out_dir, cfg)
    print(f"wrote {csv_path}, {val_path} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
