# Torch installation review for PR #5

## Decision

Require Torch 2.11 / Triton 3.6 for all engines and supported Python versions
(3.10–3.14), with CUDA 12.8 selected by the repository lockfile. Users retaining
Torch 2.8 should pin `packed-encoders==0.1.0`. This release does not maintain a
legacy Torch installation or CI matrix.

The initial review held back the upgrade because the configured FA2 wheel was
built only for CPython 3.11 / Torch 2.8. The official FlashAttention GitHub assets
inspected did not provide a 2.11 replacement. **Astral's GPU wheel index does**:
`flash-attn==2.8.3.post1+cu.12.8.torch.2.11`, with metadata requiring
`torch==2.11.*`, covers Linux x86-64/aarch64 and CPython 3.10–3.14. Replacing the
hard-coded URL with that explicit, FA2-only index removes the installation blocker.
These are Astral-provided builds, not the previously configured upstream wheel.

A fresh, normally resolved installation from the lockfile passes the complete
local RTX 5090 suite with both FA2 and Qwen, including the real topk checkpoint.
No source compilation or `--no-deps` workaround is required. This establishes
local correctness, not performance parity on the contributor's GPUs.

## Dependency contract

- Base requirements are Torch `>=2.11,<2.12` and Triton `>=3.6,<3.7` on all
  supported Python versions. The direct Triton requirement also covers CPU-only
  Torch environments. `uv.lock` records the validated 2.11 stack.
- Keep the explicit CUDA 12.8 PyTorch index. Moving to PyPI's default CUDA build
  would independently change runtime and driver requirements.
- The FA2 requirement remains `2.8.3.post1`; Astral's wheel metadata selects the
  appropriate Torch ABI. Do not reuse the old Torch 2.8 binary with Torch 2.11.
- `qwen3_5` includes Torch 2.11, Transformers 5.9, FLA, SentenceTransformers 6.x
  and matching torchvision, covering the topk checkpoint's remote wrapper.
  Transformers is bounded below 5.10 because the adapters use internal APIs.
- PyLate currently pins SentenceTransformers 5.3 and cannot share the topk 6.x
  environment. The explicit extras conflict makes this clear to uv; it is not a
  Torch or FA2 incompatibility. Use separate environments for these two wrappers.
- The lockfile includes the optional FA4 profile. Its actual GPU paths still need
  validation on supported hardware; installation resolution is not GPU validation.

## Reproduction

From the repository root:

```bash
uv sync --locked --no-dev --group test --extra fa2 --extra qwen3_5
PE_TEST_TOPK=1 uv run --no-sync pytest tests -q --disable-warnings
```

For CPU tests, resolve against published package metadata and the CPU index.
`--no-sources` prevents the checkout's CUDA source settings overriding that index:

```bash
uv venv /tmp/pe-cpu
uv pip install --python /tmp/pe-cpu/bin/python --no-sources --torch-backend cpu \
  -e '.[qwen3_5]' 'torch==2.11.*' pytest
uv pip check --python /tmp/pe-cpu/bin/python
/tmp/pe-cpu/bin/python -c 'import torch; assert torch.version.cuda is None'
/tmp/pe-cpu/bin/python -m pytest tests -q --disable-warnings
```

Use `--torch-backend cu128` for CUDA. Installing FA2 outside `uv sync` also
requires `--index https://wheels.astral.sh/simple/cu128/`; see the README.

## Validation

Local GPU: RTX 5090 (sm_120), driver 615.71.09. Fresh locked environment:
Python 3.12.12, Torch 2.11.0+cu128, Triton 3.6.0, FA2
2.8.3.post1+cu.12.8.torch.2.11, Transformers 5.9.0, FLA/fla-core 0.5.2,
CuteDSL 4.5.2, SentenceTransformers 6.1.0, torchvision 0.26.0+cu128.

| Run | Result |
| --- | --- |
| Fresh locked FA2 + Qwen environment, full suite, `PE_TEST_TOPK=1` | 168 passed, 1 skipped (requires two GPUs) |
| Fresh complete Qwen CPU install, Python 3.12 / Torch 2.11.0+cpu | 64 passed, 105 skipped (CUDA) |
| Repaired project environment, locked Torch 2.11 / FA2 / PyLate, full GPU suite | 143 passed, 1 skipped (optional FLA absent) |
| Earlier Torch 2.8.0+cu128 / Triton 3.4.0 / upstream FA2 check, full suite | 143 passed, 1 skipped (optional FLA absent) |

The combined GPU run covers ModernBERT inference, autograd/training graphs, the
Qwen engine and real topk eager/graphed encode parity. Logs are retained locally
at `/tmp/pe-install-locked-full.log`, `/tmp/pe-install-project-tests.log` and
`/tmp/pe-pr5-modern-28.log`. The repaired project `.venv` retains the PyLate profile;
the combined Qwen profile was tested in `/tmp/pe-install-combined`.
All tested installations pass `uv pip check`.

CI resolves CPU installations for Torch 2.11 on Python 3.10–3.14, plus the
complete Qwen extra on Python 3.12. It checks dependency consistency and asserts that Torch is CPU-only.
A separate job checks the committed lockfile, builds the wheel and resolves the
FA2/Qwen, FA2/PyLate and FA4/Qwen profiles.

Performance reproduction on the reported GPUs remains deferred. Existing
benchmark numbers and dispatch calibration describe their original environments;
they have not been re-established for the new lockfile. FA4 GPU validation and
multi-GPU Qwen testing also remain outstanding.

## Sources

- [Astral GPU wheel index](https://wheels.astral.sh/)
- [uv: GPU-enabled PyTorch extensions](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PyTorch 2.11 release announcement](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- [PyTorch release compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [Official FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases)
- [Topk model and requirements](https://huggingface.co/topk-io/topk-embed-v1-xsmall/tree/main)
