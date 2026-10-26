import sys, time, numpy as np, onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 16; so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(sys.argv[1], sess_options=so, providers=["CPUExecutionProvider"])
ids = np.random.default_rng(1).integers(100, 50000, (1, 8192)); m = np.ones_like(ids)
for i in range(3):
    t = time.perf_counter(); s.run(None, {"input_ids": ids, "attention_mask": m}); print(sys.argv[1].rsplit('/',1)[1], i, round(time.perf_counter() - t, 2), "s", flush=True)
print("peak MB", int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) // 1024)
