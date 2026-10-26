"""Median-of-process-medians table for interleaved runs: python summarize.py results.jsonl"""
import collections, json, statistics as st, sys
d = collections.defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line); d[(r["tag"], r["workload"])].append(r["median_ms"])
ws = list(dict.fromkeys(k[1] for k in d))
print(f"{'config':22s}" + "".join(f"{w:>24s}" for w in ws))
for t in sorted({k[0] for k in d}, key=lambda t: (t.split("_t")[-1].zfill(3), t)):
    print(f"{t:22s}" + "".join(f"{st.median(d[(t, w)]):>10.2f} [{min(d[(t, w)]):6.1f}-{max(d[(t, w)]):6.1f}]" for w in ws))
