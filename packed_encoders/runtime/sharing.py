"""Rows that start with the same tokens: run each shared prefix once.

A decision model asking several questions about one context gets a batch whose rows repeat
that context (system prompt + context, then a different question per row); a shared system
prompt or few-shot block does the same. On a causal model a prefix's hidden states do not
depend on what follows it, so the prefix only needs computing once; every row can then
continue from it.

A mask cannot express that on a hybrid backbone. Attention could be told "see the prefix and
your own tokens", but a recurrent mixer (GatedDeltaNet) carries its context in a state that
every earlier token has updated, and a short causal conv reads the previous tokens directly.
So the batch becomes a two-level forest instead:

- roots: each shared prefix once, and each row that shares nothing; from a zero state.
- children: the rest of each sharing row, continued from its root. A recurrent mixer starts
  from the root's final state, the conv's first taps read the root's last tokens, and
  attention reads the root's keys and values (a query segment shorter than its key segment).

`plan_shared_prefixes` finds the groups and builds every index an engine needs, on the host;
the engines (Qwen3.5, LFM2) run the forest and return hidden states in the caller's layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from packed_encoders.runtime.staging import packed_layout_host


@dataclass
class SharedPlan:
    """Host int64 indices for one batch. Segment order: roots, then children."""

    lengths: list[int]        # query tokens per segment
    kv_lengths: list[int]     # key tokens per segment: a root's own; a child's root prefix + its own
    n_roots: int
    root_tokens: int          # tokens in the roots region (the children region follows)
    src: Tensor               # [T_s] caller token feeding each forward token
    out: Tensor               # [T] forward token whose hidden state each caller token takes
    rope_pos: Tensor          # [T_s] position of each token in its row
    conv_pos: Tensor          # [T_s] position in its segment (the packed conv restarts per segment)
    parent: Tensor            # [children] root of each child
    fix_rows: Tensor          # [F] child tokens whose conv window reaches back into the root
    fix_taps: Tensor          # [F, W] forward rows under each such token's conv taps, oldest first
    kv_idx: Tensor            # [sum(kv_lengths)] forward rows forming each segment's keys and values
    saved_tokens: int         # tokens the plan does not recompute


def plan_shared_prefixes(ids: Tensor, lengths: Sequence[int], *, min_prefix: int,
                         conv_width: int, _use_cute: bool = True) -> SharedPlan | None:
    """Group rows (back to back in `ids`, `lengths` host ints) whose first `min_prefix` tokens agree;
    a group shares its longest common prefix, keeping at least one token of every row for itself.
    Returns None when no group forms. One device-to-host read, and only when two rows are longer
    than `min_prefix`."""
    lens = [int(n) for n in lengths]
    if min_prefix < max(conv_width, 1):
        raise ValueError(f"min_prefix {min_prefix} must cover the conv width {conv_width}")
    rows = [i for i, n in enumerate(lens) if n > min_prefix]
    if len(rows) < 2:
        return None
    cu, _ = packed_layout_host(lens)
    starts = cu[:-1].tolist()
    # Pairwise work is bounded to small batches. Large batches retain the sort-based
    # planner; CPU-only and Qwen-only installations need no Cutlass import.
    if ids.is_cuda and len(lens) <= 64 and _use_cute:
        try:
            from packed_encoders._kernels.prefix_match import pairwise_prefix_lengths
        except ImportError:
            pass
        else:
            common = pairwise_prefix_lengths(ids, cu.to(ids.device)).tolist()
            groups = {}
            for r in rows:
                first = next((p for p in groups if common[r][p] >= min_prefix), r)
                groups.setdefault(first, []).append(r)
            shared = {}
            for first, rs in groups.items():
                if len(rs) > 1:
                    prefix = min(min(common[r][first] for r in rs[1:]), min(lens[r] for r in rs) - 1)
                    if prefix >= min_prefix:
                        shared[first] = (prefix, rs)
            return _layout(lens, starts, shared, conv_width) if shared else None

    # Device: rows past min_prefix tokens, padded with -1; group by the first min_prefix tokens, then
    # measure each row's common prefix with its group's first row.
    dev = ids.device
    st = torch.tensor([starts[r] for r in rows], device=dev)
    ln = torch.tensor([lens[r] for r in rows], device=dev)
    col = torch.arange(int(max(lens[r] for r in rows)), device=dev)
    valid = col[None] < ln[:, None]
    tok = torch.where(valid, ids[(st[:, None] + col).clamp_max(ids.numel() - 1)], -1)
    _, group = torch.unique(tok[:, :min_prefix], dim=0, return_inverse=True)
    order = torch.arange(len(rows), device=dev)
    first = torch.full_like(order, len(rows)).scatter_reduce(0, group, order, "amin")[group]
    lcp = ((tok == tok[first]) & valid & valid[first]).int().cumprod(1).sum(1)
    group, lcp = torch.stack([group, lcp]).tolist()

    members: dict[int, list[int]] = {}
    for j, g in enumerate(group):
        members.setdefault(g, []).append(j)
    shared: dict[int, tuple[int, list[int]]] = {}        # first row -> (prefix, rows)
    for js in members.values():
        if len(js) < 2:
            continue
        rs = [rows[j] for j in js]                       # js[0] is the group's first row
        prefix = min(min(lcp[j] for j in js[1:]), min(lens[r] for r in rs) - 1)
        if prefix >= min_prefix:
            shared[rs[0]] = (prefix, rs)
    if not shared:
        return None
    return _layout(lens, starts, shared, conv_width)


def _layout(lens, starts, shared, width) -> SharedPlan:
    grouped = {r for _, rs in shared.values() for r in rs}
    roots = []                                  # (caller row, tokens)
    root_of = {}                                # caller row -> root index
    for r, n in enumerate(lens):
        if r in shared:
            root_of.update({m: len(roots) for m in shared[r][1]})
            roots.append((r, shared[r][0]))
        elif r not in grouped:
            root_of[r] = len(roots)
            roots.append((r, n))
    children = [(r, shared[first][0]) for first in shared for r in shared[first][1]]   # (row, prefix)

    lengths = [n for _, n in roots] + [lens[r] - p for r, p in children]
    seg_src = [starts[r] for r, _ in roots] + [starts[r] + p for r, p in children]
    cu, conv_pos = packed_layout_host(lengths)
    seg = cu[:-1].tolist()
    n_roots, root_tokens = len(roots), int(cu[len(roots)])
    lt = torch.tensor(lengths)
    src = torch.repeat_interleave(torch.tensor(seg_src) - cu[:-1], lt) + torch.arange(int(cu[-1]))

    child_of = {r: n_roots + c for c, (r, _) in enumerate(children)}
    out, rope, parent, fix_rows, fix_taps, kv = [], [], [], [], [], [torch.arange(root_tokens)]
    for r, n in enumerate(lens):
        k = root_of[r]
        if r not in child_of:
            out.append(torch.arange(seg[k], seg[k] + n))
            continue
        c, p = child_of[r], roots[k][1]
        out += [torch.arange(seg[k], seg[k] + p), torch.arange(seg[c], seg[c] + n - p)]
    for k, (_, n) in enumerate(roots):
        rope.append(torch.arange(n))
    for c, (r, p) in enumerate(children):
        s, k, n = seg[n_roots + c], root_of[r], lens[r] - p
        rope.append(torch.arange(p, p + n))
        parent.append(k)
        kv += [torch.arange(seg[k], seg[k] + p), torch.arange(s, s + n)]
        for i in range(min(width - 1, n)):      # taps oldest first: lag width-1 .. 0
            fix_rows.append(s + i)
            fix_taps.append([s + i - lag if i >= lag else seg[k] + p - (lag - i) for lag in range(width - 1, -1, -1)])
    saved = sum((len(rs) - 1) * p for p, rs in shared.values())
    return SharedPlan(
        lengths=lengths, kv_lengths=lengths[:n_roots] + [lens[r] for r, _ in children],
        n_roots=n_roots, root_tokens=root_tokens, src=src, out=torch.cat(out), rope_pos=torch.cat(rope),
        conv_pos=conv_pos, parent=torch.tensor(parent, dtype=torch.long), fix_rows=torch.tensor(fix_rows, dtype=torch.long),
        fix_taps=torch.tensor(fix_taps, dtype=torch.long).view(-1, width), kv_idx=torch.cat(kv), saved_tokens=saved)
