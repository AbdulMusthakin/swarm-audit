"""Modal app serving the open-weights model with vLLM; exact action log-probabilities.

What it provides
----------------
- `download_weights` (CPU only): caches a model snapshot in the Modal Volume
  `swarm-audit-weights`, so GPU containers never pay for the download.
- `Server` (one L4, vLLM, bf16): `label_logprobs(prompts, allowed_token_ids)` returns, for
  each prompt (a list of token ids), the raw log-probability of each allowed token at the
  next position, the probabilities renormalised over the allowed set, and `p_valid`, the
  total raw probability of the allowed set.
- `hf_label_logprobs` (one L4, transformers, bf16): the same quantities from a plain
  forward pass, used to validate the vLLM numbers.
- `call_logged`: client-side wrapper that writes every call to logs/serving/.

Exact probabilities from vLLM
-----------------------------
Method "logprob_token_ids" (default, vLLM >= 0.32): `SamplingParams(logprob_token_ids=ids)`
returns the log-probabilities of exactly these ids. With the default
`logprobs_mode="raw_logprobs"` they are computed in float32 as log_softmax over the full
vocabulary of the model's logits, before any logits processor (vllm/v1/sample/sampler.py).
One request per prompt.

Method "allowed_token_ids" (cross-check, works on older vLLM too): one request per allowed
token, with `allowed_token_ids=[id]` and `logprobs=0`. The sampled token is then forced to
be `id`, and vLLM always returns the raw log-probability of the sampled token. Prefix
caching makes the extra requests cheap.

Prompts are token ids, not text, so vLLM and transformers see identical inputs.

Running
-------
Needs an authenticated Modal client (`modal token new`, or MODAL_TOKEN_ID/SECRET).
For gated models, create a Modal secret holding HF_TOKEN and set
SWARM_AUDIT_HF_SECRET=<secret name> before running. scripts/screen_models.py drives this.
"""

# No `from __future__ import annotations` here: Modal reads the class-parameter
# annotations at runtime and needs real types, not strings.
import datetime as _dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "swarm-audit-serving"
VLLM_VERSION = os.environ.get("SWARM_AUDIT_VLLM_VERSION", "0.32.0")
GPU = os.environ.get("SWARM_AUDIT_GPU", "L4")
TIMEOUT_S = int(float(os.environ.get("SWARM_AUDIT_TIMEOUT_MIN", "20")) * 60)
# Build hand-off section 4: batch invariance (vLLM >= 0.32 beta, compute capability >= 8.0;
# the L4 is 8.9) is off by default until gate G3 decides. It is baked into the image
# environment, so the client and the container agree on it.
BATCH_INVARIANT = os.environ.get("SWARM_AUDIT_BATCH_INVARIANT", "0") == "1"
# More than one container only when an arm is split across containers, up to the
# account's GPU concurrency limit.
MAX_CONTAINERS = int(os.environ.get("SWARM_AUDIT_MAX_CONTAINERS", "1"))
# Seconds a GPU container stays up after its last call. Matters for the deployed app
# (scripts/deploy_server.py), which outlives each script, so consecutive steps reuse a
# warm GPU instead of queueing for a new one.
SCALEDOWN_S = int(os.environ.get("SWARM_AUDIT_SCALEDOWN_S", "60"))
CACHE = "/cache"
HF_HOME = f"{CACHE}/hf"
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "serving"

app = modal.App(APP_NAME)

# The Modal secret holding HF_TOKEN (needed for gated models). Its name is baked into
# the image's environment so that the container defines the functions with the same
# secrets as the client; otherwise Modal refuses to start them.
_hf_secret = os.environ.get("SWARM_AUDIT_HF_SECRET", "")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub")
    .env({"HF_HOME": HF_HOME, "TOKENIZERS_PARALLELISM": "false",
          "SWARM_AUDIT_HF_SECRET": _hf_secret,
          "SWARM_AUDIT_BATCH_INVARIANT": "1" if BATCH_INVARIANT else "0",
          "SWARM_AUDIT_SCALEDOWN_S": str(SCALEDOWN_S),
          "SWARM_AUDIT_MAX_CONTAINERS": str(MAX_CONTAINERS),
          **({"VLLM_BATCH_INVARIANT": "1"} if BATCH_INVARIANT else {})})
    .add_local_python_source("swarm_audit")
)

weights = modal.Volume.from_name("swarm-audit-weights", create_if_missing=True)

secrets = [modal.Secret.from_name(_hf_secret)] if _hf_secret else []

# Weight files only: skip duplicate Mistral-format and original/ checkpoints.
_ALLOW = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"]
_IGNORE = ["consolidated*", "original/*", "*.pth", "*.gguf"]


@app.function(image=image, volumes={CACHE: weights}, secrets=secrets, cpu=2,
              memory=4096, timeout=3600)
def download_weights(hf_id: str, revision: str) -> dict:
    """Cache a model snapshot in the Volume (CPU container; no GPU time)."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(hf_id, revision=revision, allow_patterns=_ALLOW,
                             ignore_patterns=_IGNORE)
    weights.commit()
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {"path": path, "gb": size / 1e9, "seconds": time.time() - t0}


_TOKENIZER_FILES = ["tokenizer*", "*.jinja", "special_tokens_map.json", "config.json",
                    "generation_config.json", "tekken.json"]


@app.function(image=image, secrets=secrets, cpu=1, memory=2048, timeout=600)
def tokenizer_files(hf_id: str, revision: str) -> dict:
    """Tokenizer and chat-template files of a model, fetched with the Modal HF secret.

    Lets the CPU-side tokenizer screen run on gated models without a local HF token."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(hf_id, revision=revision, allow_patterns=_TOKENIZER_FILES))
    return {str(f.relative_to(path)): f.read_bytes() for f in path.rglob("*") if f.is_file()}


def _result(raw: list[float], sampled: int | None = None) -> dict:
    from swarm_audit.tokens import entropy_bits, renormalise

    probs, p_valid = renormalise(raw)
    return {"logprobs": raw, "probs": probs, "p_valid": p_valid,
            "entropy_bits": entropy_bits(probs), "sampled": sampled}


@app.cls(image=image, gpu=GPU, volumes={CACHE: weights}, secrets=secrets,
         timeout=TIMEOUT_S, scaledown_window=SCALEDOWN_S, max_containers=MAX_CONTAINERS)
class Server:
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()
    engine_kwargs_json: str = modal.parameter(default="{}")

    @modal.enter()
    def load(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from vllm import LLM

        kw: dict[str, Any] = dict(
            model=self.hf_id, revision=self.revision, dtype="bfloat16",
            max_model_len=4096, gpu_memory_utilization=0.90, enforce_eager=True,
            enable_prefix_caching=True, seed=0,
        )
        kw.update(json.loads(self.engine_kwargs_json))
        t0 = time.time()
        self.llm = LLM(**kw)
        self.load_seconds = time.time() - t0
        self.engine_kwargs = kw

    @modal.method()
    def info(self) -> dict:
        import torch
        import vllm

        try:  # KV-cache capacity in tokens, as profiled by vLLM at load
            cc = self.llm.llm_engine.vllm_config.cache_config
            kv_tokens = cc.num_gpu_blocks * cc.block_size
        except Exception:
            kv_tokens = None
        return {"vllm": vllm.__version__, "torch": torch.__version__,
                "kv_cache_tokens": kv_tokens,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "load_seconds": self.load_seconds, "engine_kwargs": self.engine_kwargs,
                "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1"}

    @modal.method()
    def label_logprobs(self, prompts: list[list[int]], allowed_token_ids: list[int],
                       method: str = "logprob_token_ids") -> list[dict]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        ids = [int(t) for t in allowed_token_ids]
        if method == "logprob_token_ids":
            sp = SamplingParams(max_tokens=1, temperature=1.0, logprob_token_ids=ids,
                                allowed_token_ids=ids)
            outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                     sp, use_tqdm=False)
            res = []
            for o in outs:
                step = o.outputs[0].logprobs[0]
                res.append(_result([step[t].logprob for t in ids], o.outputs[0].token_ids[0]))
            return res
        if method == "allowed_token_ids":
            reqs = [(i, t) for i in range(len(prompts)) for t in ids]
            sps = [SamplingParams(max_tokens=1, temperature=1.0, allowed_token_ids=[t],
                                  logprobs=0) for _, t in reqs]
            outs = self.llm.generate(
                [TokensPrompt(prompt_token_ids=prompts[i]) for i, _ in reqs], sps,
                use_tqdm=False)
            raw: dict[tuple[int, int], float] = {}
            for (i, t), o in zip(reqs, outs):
                if o.outputs[0].token_ids[0] != t:
                    raise RuntimeError(f"forced token {t} but sampled {o.outputs[0].token_ids[0]}")
                raw[(i, t)] = o.outputs[0].logprobs[0][t].logprob
            return [_result([raw[(i, t)] for t in ids]) for i in range(len(prompts))]
        raise ValueError(f"unknown method {method!r}")


# ------------------------------------------------------------ transformers reference


def load_hf_model(hf_id: str, revision: str | None, device: str, dtype: str = "bfloat16"):
    """A plain transformers model for the reference forward pass."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(hf_id, revision=revision)
    multimodal = any(a.endswith("ForConditionalGeneration") for a in (cfg.architectures or []))
    order = ([AutoModelForImageTextToText, AutoModelForCausalLM] if multimodal
             else [AutoModelForCausalLM, AutoModelForImageTextToText])
    err: Exception | None = None
    for cls in order:
        try:
            model = cls.from_pretrained(hf_id, revision=revision, dtype=getattr(torch, dtype))
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            err = e  # architecture not in this auto class: try the other one
            continue
        return model.to(device).eval()
    raise RuntimeError(f"could not load {hf_id} with transformers") from err


def hf_logprobs(model: Any, prompts: list[list[int]], allowed_token_ids: list[int]) -> list[dict]:
    """Raw next-token log-probabilities of the allowed ids, one forward pass per prompt."""
    import torch

    ids = torch.tensor(allowed_token_ids)
    res = []
    with torch.no_grad():
        for p in prompts:
            x = torch.tensor([p], device=model.device)
            try:
                logits = model(input_ids=x, logits_to_keep=1).logits[0, -1].float()
            except TypeError:  # model without logits_to_keep
                logits = model(input_ids=x).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)[ids.to(logits.device)].tolist()
            res.append(_result(lp))
    return res


@app.function(image=image, gpu=GPU, volumes={CACHE: weights}, secrets=secrets,
              timeout=TIMEOUT_S, max_containers=1)
def hf_label_logprobs(hf_id: str, revision: str,
                      batches: list[tuple[list[list[int]], list[int]]]) -> dict:
    """transformers reference: for each (prompts, allowed_token_ids) batch, the same
    output as Server.label_logprobs. Loads the model once for all batches."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers

    t0 = time.time()
    model = load_hf_model(hf_id, revision, device="cuda")
    load_s = time.time() - t0
    out = [hf_logprobs(model, prompts, ids) for prompts, ids in batches]
    return {"results": out, "transformers": transformers.__version__,
            "torch": torch.__version__, "load_seconds": load_s}


# ------------------------------------------------------------------------- logging


def _find_logged(log_name: str, function: str, meta: dict, prompts: list[list[int]]) -> Any:
    """The output of the latest logged call with the same function, metadata and prompt
    token ids, or None. Used to resume a run without repeating finished GPU calls."""
    for path in sorted(LOG_DIR.glob(f"*_{log_name}.jsonl"), reverse=True):
        for line in reversed(path.read_text().splitlines()):
            e = json.loads(line)
            if e.get("function") != function or any(e.get(k) != v for k, v in meta.items()):
                continue
            if [p["ids"] for p in e.get("prompts", [])] == prompts:
                return e["output"]
    return None


def call_logged(fn: Any, *args: Any, log_name: str, meta: dict, prompt_ids: list[str] | None = None,
                prompts: list[list[int]] | None = None, **kwargs: Any) -> Any:
    """Call a Modal function (`fn.remote`) or a local callable, and append the call, its
    inputs and its outputs to logs/serving/<date>_<log_name>.jsonl.

    With SWARM_AUDIT_RESUME=1, an identical call already in the log (same function,
    metadata and prompt token ids) is not repeated: its logged output is returned."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fname = getattr(fn, "__name__", str(fn))
    if os.environ.get("SWARM_AUDIT_RESUME") == "1":
        out = _find_logged(log_name, fname, meta, prompts or [])
        if out is not None:
            print(f"resumed from log: {log_name} {meta.get('call')}", flush=True)
            return out
    t0 = time.time()
    remote = getattr(fn, "remote", None)
    out = remote(*args, **kwargs) if remote else fn(*args, **kwargs)
    entry = {
        "time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seconds": round(time.time() - t0, 3),
        "function": fname,
        **meta,
        "prompts": [
            {"prompt_id": pid, "n_tokens": len(p),
             "sha256": hashlib.sha256(json.dumps(p).encode()).hexdigest()[:16], "ids": p}
            for pid, p in zip(prompt_ids or [], prompts or [])
        ],
        "output": out,
    }
    path = LOG_DIR / f"{_dt.date.today().isoformat()}_{log_name}.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return out
