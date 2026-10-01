"""Qwen3.5 (topk-embed-v1) under packed-encoders on Modal GPUs: GPU tests + benchmark, per GPU x stack.

    modal run benchmarks/scripts/modal_qwen35.py --data sample.json --gpus L40S,A100-40GB,H100 \\
        --models xsmall,small --stack pinned --out results/

Stacks (the stack is one image; every requested GPU runs concurrently):
  pinned  topk's own requirements.txt: torch 2.11, transformers 5.9.0 — attention via torch's varlen kernel
  fa2     torch 2.8 + flash-attn 2.8.3 wheel + transformers 5.16.1 (the packed-encoders pin)
  fa4     torch 2.8 + flash-attn-4 (CuteDSL; sm_90 / sm_100) + transformers 5.16.1
The attention kernel is chosen by probe inside `pack()`, so each result records what actually ran.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# This module is imported twice: locally (builds the images from the repo) and inside the container
# (hydrates `run`), where the file sits at /root and PE_STACK comes from the image env.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root/pe")
STACK = os.environ.get("PE_STACK", "pinned")
FA2_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
             "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
MODELS = {"xsmall": "topk-io/topk-embed-v1-xsmall", "small": "topk-io/topk-embed-v1-small"}

if STACK == "pinned":
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
             .pip_install("transformers==5.9.0", "sentence-transformers==6.0.1", "flash-linear-attention==0.5.1",
                          "kernels==0.14.1", "safetensors>=0.7.0", "Pillow>=12.0", "numpy", "pytest")
             # No nvidia-cutlass-dsl here: 4.5.2 pulls cuda-bindings 13 and resolves torch 2.11+cu128 up to a
             # CUDA 13 torch (torchvision::nms then fails to load). The Qwen3.5 path never imports CuteDSL.
             .run_commands("python -c \"import torch; assert torch.__version__.startswith('2.11'), torch.__version__\""))
else:
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
             .pip_install("transformers==5.16.1", "sentence-transformers==6.0.1", "flash-linear-attention==0.5.1",
                          "accelerate==1.14.0", "safetensors==0.8.0", "tokenizers==0.23.1", "einops==0.8.2",
                          "pillow", "numpy", "pytest", "nvidia-cutlass-dsl==4.5.2"))
    image = image.pip_install(FA2_WHEEL) if STACK == "fa2" else image.pip_install("flash-attn-4==4.0.0b16",
                                                                                  "quack-kernels==0.5.0")
image = (image.env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/pe", "PE_STACK": STACK,
                    "TRITON_CACHE_DIR": f"/cache/triton-pe-{STACK}", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(str(REPO / "packed_encoders"), "/root/pe/packed_encoders", ignore=["__pycache__"])
         .add_local_dir(str(REPO / "benchmarks"), "/root/pe/benchmarks", ignore=["__pycache__", "docs"])
         .add_local_dir(str(REPO / "tests"), "/root/pe/tests", ignore=["__pycache__"]))
app = modal.App(f"packed-encoders-qwen35-{STACK}", image=image)
cache = modal.Volume.from_name("topk-hf-cache", create_if_missing=True)


LADDER = {"xsmall": "topk:topk-io/topk-embed-v1-xsmall", "small": "topk:topk-io/topk-embed-v1-small",
          "gte": "st:Alibaba-NLP/gte-modernbert-base"}


@app.function(gpu="L40S", timeout=4 * 3600, volumes={"/cache": cache})
def run(models: list[str], data: str, bench_args: str, tests: str, ladder: list[str] | None = None,
        probe: list[str] | None = None, ladder_args: str = "") -> dict:
    import shlex
    import subprocess

    import torch

    # Plain str/list only: the local client has no torch to unpickle a TorchVersion with.
    out = {"stack": STACK, "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability()),
           "torch": str(torch.__version__), "tests": None, "bench": {}}
    Path("/tmp/data.json").write_text(data)
    if tests != "none":
        targets = (["/root/pe/tests/"] if tests == "all"
                   else [f"/root/pe/tests/test_{t}.py" for t in tests.split(",") if t])
        # no:logging: a failing test otherwise dumps every captured torch trace DEBUG record, burying the report.
        p = subprocess.run(["python", "-m", "pytest", "-q", *targets, "-rfEs", "--tb=short", "-p", "no:cacheprovider",
                            "-p", "no:logging"], cwd="/root/pe", capture_output=True, text=True,
                           env={**os.environ, "PE_TEST_TOPK": "1"})
        print(p.stdout[-12000:], p.stderr[-3000:], flush=True)
        out["tests"] = {"returncode": p.returncode, "tail": p.stdout[-6000:]}
    for m in models:
        cmd = ["python", "-u", "/root/pe/benchmarks/qwen35_topk_bench.py", "--model", MODELS.get(m, m),
               "--data", "/tmp/data.json", "--out", f"/tmp/bench_{m}.json", *shlex.split(bench_args)]
        print("$", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd="/root/pe")
        f = Path(f"/tmp/bench_{m}.json")
        out["bench"][m] = json.loads(f.read_text()) if p.returncode == 0 and f.exists() else {"error": p.returncode}
    out["probe"] = {}
    for m in probe or []:
        f = Path(f"/tmp/probe_{m}.json")
        cmd = ["python", "-u", "/root/pe/benchmarks/qwen35_parity_probe.py", "--model", MODELS.get(m, m),
               "--data", "/tmp/data.json", "--out", str(f)]
        print("$", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd="/root/pe")
        out["probe"][m] = json.loads(f.read_text()) if f.exists() else {"error": p.returncode}
    out["ladder"] = {}
    for spec in ladder or []:
        family, model = LADDER.get(spec, spec).split(":", 1)
        f = Path(f"/tmp/ladder_{spec.replace('/', '_')}.json")
        cmd = ["python", "-u", "/root/pe/benchmarks/practical_ladder.py", "--family", family, "--model", model,
               "--data", "/tmp/data.json", "--out", str(f), *shlex.split(ladder_args)]
        print("$", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd="/root/pe")
        out["ladder"][spec] = json.loads(f.read_text()) if f.exists() else {"error": p.returncode}
    cache.commit()
    return out


@app.local_entrypoint()
def main(data: str, gpus: str = "L40S", models: str = "xsmall", out: str = "qwen35_results", bench_args: str = "",
         tests: str = "qwen35", ladder: str = "", probe: str = "", ladder_args: str = ""):
    """tests: all (the whole suite, ModernBERT GPU tests included) | none | comma list of test file stems
    (qwen35,varlen -> tests/test_qwen35.py tests/test_varlen.py).
    models: topk sizes for the forward/encode bench ("" to skip). ladder: comma list of practical-ladder
    models (xsmall, small, gte, or family:model_id)."""
    payload = Path(data).read_text()
    Path(out).mkdir(parents=True, exist_ok=True)
    specs = [x for x in ladder.split(",") if x]
    probes = [x for x in probe.split(",") if x]
    calls = {g: run.with_options(gpu=g).spawn([m for m in models.split(",") if m], payload, bench_args, tests, specs,
                                              probes, ladder_args)
             for g in gpus.split(",")}
    for g, call in calls.items():
        try:
            r = call.get()
        except Exception as exc:  # noqa: BLE001
            r = {"error": repr(exc)}
        path = Path(out) / f"{STACK}_{g}.json"
        path.write_text(json.dumps(r, indent=1, default=str))
        print(f"wrote {path}")
