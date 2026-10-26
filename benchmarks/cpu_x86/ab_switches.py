"""Alternating A/B of the engine's private switches, one process, real engine.

python ab_switches.py THREADS ROUNDS VARIANT [VARIANT...] [--workloads a,b]
VARIANT is name=attr:value,attr:value (e.g. off=fused_rope:0,fused_geglu:0,fused_norm:0): attributes of the
engine's CPUAttention, or of the engine itself (e.g. fast_masks). name=@REF[:PRECISION[:rehome]] (e.g. head=@HEAD,
int8=@.:int8, anon=@.:fp32:rehome) instead runs packed_encoders as of git REF (. for the working tree) on its own
copy of the model, at its defaults and the given precision (fp32 by default); rehome packs it with
ModernBertCPU(rehome_weights=True). The first variant is the reference for timing
ratios and output differences. Prints median ms per variant and workload.
"""
import argparse
import importlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import inputs, workloads  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("threads", type=int)
p.add_argument("rounds", type=int)
p.add_argument("variants", nargs="+")
p.add_argument("--workloads", default="b1_s16_pad32,b1_s32,b8_s32")
a = p.parse_args()
torch.set_num_threads(a.threads)
from transformers import AutoModel  # noqa: E402
import packed_encoders as pe  # noqa: E402

root = Path(__file__).resolve().parents[2]
refs = tempfile.TemporaryDirectory(prefix="ab-refs-")


def package_at(ref, name):
    # The package as of `ref` ("." for the working tree), renamed so it imports next to the
    # working tree's own; its model copy is placed like any other @ variant's.
    out = Path(refs.name)
    if ref == ".":
        shutil.copytree(root / "packed_encoders", out / name, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        archive = subprocess.run(["git", "-C", str(root), "archive", ref, "packed_encoders"], check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(out)], input=archive, check=True)
        (out / "packed_encoders").rename(out / name)
    for path in (out / name).rglob("*.py"):
        path.write_text(path.read_text().replace("packed_encoders", name))
    sys.path.insert(0, str(out))
    return importlib.import_module(name)


def load():
    return AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa").float().eval()


variants, models = {}, {}
for n, spec in enumerate(a.variants):
    name, _, settings = spec.partition("=")
    if settings.startswith("@"):
        ref, _, precision = settings[1:].partition(":")
        precision, _, rehome = precision.partition(":")
        package = package_at(ref, f"pe_ref{n}")
        models[name] = load()
        engine = package.ModernBertCPU(threads=a.threads, **({"rehome_weights": True} if rehome else {}))
        package.pack(models[name], engine=engine, precision=precision or "fp32")
        variants[name] = {}
    else:
        variants[name] = {k: bool(int(v)) for k, v in (s.split(":") for s in settings.split(",") if s)}
model = load()
pe.pack(model, engine=pe.ModernBertCPU(threads=a.threads))
eng = pe.get_engine(model)
att = eng.attention
print(f"threads={a.threads} backend={att.backend} rows={'native' if att.rows is not None else 'none'} "
      f"rounds={a.rounds}", flush=True)

for name in a.workloads.split(","):
    seq, lengths = workloads()[name]
    ids, mask = inputs(seq, lengths)
    m = mask.bool()
    reps = 3 if a.threads == 1 and name == "b8_s32" else 10
    times, outs = {v: [] for v in variants}, {}
    order = list(variants)
    with torch.inference_mode():
        for r in range(a.rounds + 1):
            for v in (order if r % 2 == 0 else order[::-1]):
                for attr, value in variants[v].items():
                    target = att if hasattr(att, attr) else eng
                    if not hasattr(target, attr):
                        raise SystemExit(f"unknown switch {attr}")
                    setattr(target, attr, value)
                run = models.get(v, model)
                t = time.perf_counter()
                for _ in range(reps):
                    out = run(input_ids=ids, attention_mask=mask).last_hidden_state
                if r:  # round 0 warms up
                    times[v].append((time.perf_counter() - t) / reps * 1e3)
                outs[v] = out[m].clone()
    ref = order[0]
    base = statistics.median(times[ref])
    line = []
    for v in order:
        med = statistics.median(times[v])
        q1, q3 = statistics.quantiles(times[v], n=4)[0], statistics.quantiles(times[v], n=4)[2]
        rel = ((outs[v] - outs[ref]).norm() / outs[ref].norm()).item()
        line.append(f"{v} {med:8.2f} [{q1:.2f}-{q3:.2f}] ({base - med:+6.2f} ms, {base / med:4.2f}x, relL2 {rel:.1e})")
    print(f"{name:13s} | " + " | ".join(line), flush=True)
