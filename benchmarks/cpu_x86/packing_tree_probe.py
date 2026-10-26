import json, pathlib, subprocess, sys, time
source, output = sys.argv[1:]
command = [sys.executable, 'benchmarks/cpu_x86/packing_bench.py', '--worker', '--source', source, '--label', 'tree_probe', '--threads', '8', '--precision', 'int8', '--digest']
p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
peak_total = peak_children = 0
samples = 0
while p.poll() is None:
    stack = [p.pid]
    rss = []
    while stack:
        pid = stack.pop()
        try:
            status = pathlib.Path(f'/proc/{pid}/status').read_text()
            value = next(int(line.split()[1]) / 1024 for line in status.splitlines() if line.startswith('VmRSS:'))
            rss.append((pid, value))
            stack.extend(map(int, pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
            # GCC may be launched by a Python thread other than the thread-group leader.
            for task in pathlib.Path(f'/proc/{pid}/task').iterdir():
                if task.name != str(pid):
                    stack.extend(map(int, (task / 'children').read_text().split()))
        except (OSError, StopIteration):
            pass
    total = sum(v for _,v in rss)
    children = sum(v for pid,v in rss if pid != p.pid)
    peak_total = max(peak_total, total)
    peak_children = max(peak_children, children)
    samples += 1
    time.sleep(.01)
stdout, stderr = p.communicate()
if p.returncode:
    raise RuntimeError(stderr)
r = json.loads(stdout.strip().splitlines()[-1])
r.update(tree_peak_rss_mib=peak_total, compiler_children_peak_rss_mib=peak_children, tree_samples=samples)
pathlib.Path(output).write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r))
