"""Run the attention calibration comparison in a CPU-only Modal container.

    python benchmarks/cpu_review/modal_attention_calibration.py \
      --function --region eu --cpu 1 --threads 1 --require-zen3 \
      --out hardware-results/modal-calibration

The runner records the actual CPU before measuring; Modal placement alone does
not guarantee Zen 3. Each thread profile runs in a fresh process. The Sandbox is
terminated on exit (or the Function app is stopped). Install the Modal client
and authenticate locally first. The Function client must also use Python 3.12.
"""

import argparse
import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
PROBE = """
import hashlib, json, os
from pathlib import Path
import torch
from packed_encoders.cpu.topology import default_threads
fields = {}
for line in Path('/proc/cpuinfo').read_text().splitlines():
    if ':' in line:
        key, value = line.split(':', 1)
        fields.setdefault(key.strip(), value.strip())
configured_threads = torch.get_num_threads()
torch.set_num_threads(max(configured_threads, len(os.sched_getaffinity(0)) + 8))
fallback = {'requested': torch.get_num_threads(), 'selected': default_threads(),
            'expected': len(os.sched_getaffinity(0))}
assert fallback['selected'] == fallback['expected'], fallback
torch.set_num_threads(configured_threads)
print(json.dumps({
    'cpu': fields.get('model name'), 'vendor': fields.get('vendor_id'), 'family': fields.get('cpu family'),
    'model': fields.get('model'), 'flags': fields.get('flags', '').split(),
    'affinity': sorted(os.sched_getaffinity(0)), 'torch': torch.__version__,
    'torch_threads': torch.get_num_threads(), 'default_threads': default_threads(),
    'topology_visible': Path('/sys/devices/system/cpu/cpu0/topology/core_id').exists(),
    'cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
    'gcc': __import__('subprocess').check_output(['gcc', '--version'], text=True).splitlines()[0],
    'thread_environment': {key: os.environ.get(key) for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_WAIT_POLICY')},
    'modal_region': os.environ.get('MODAL_REGION'), 'modal_cloud': os.environ.get('MODAL_CLOUD_PROVIDER'),
    'affinity_fallback_check': fallback,
    'source_sha256': {path: hashlib.sha256(Path('/workspace', path).read_bytes()).hexdigest() for path in (
        'packed_encoders/cpu/attention.py', 'packed_encoders/cpu/topology.py', 'attention_calibration.py')},
}))
"""


def checked(sandbox, *command):
    process = sandbox.exec(*command, timeout=900)
    stdout, stderr = process.stdout.read(), process.stderr.read()
    if process.wait():
        raise RuntimeError(f"Command failed: {command}\n{stdout}\n{stderr}")
    return stdout


def function_profile(probe, threads, rounds, require_zen3, skip_tests):
    import json
    from pathlib import Path
    import subprocess

    environment = json.loads(subprocess.check_output(["python", "-c", probe], text=True))
    zen3 = (environment["vendor"] == "AuthenticAMD" and environment["family"] == "25"
            and environment["model"] == "1" and "avx512f" not in environment["flags"])
    report = {"environment": environment, "zen3": zen3, "profiles": {}}
    print(json.dumps(environment), flush=True)
    if require_zen3 and not zen3:
        return report
    for count in threads:
        target = f"/tmp/calibration-t{count}.json"
        process = subprocess.Popen(["python", "/workspace/attention_calibration.py", "--threads", str(count),
                                    "--rounds", str(rounds), "--out", target], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, cwd="/workspace")
        lines = []
        for line in process.stdout:
            lines.append(line)
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(line, end="", flush=True)
                continue
            if "seconds" in row:
                print(f"t{count} round {row['round']} {row['method']}: {row['seconds']:.3f} s", flush=True)
        if process.wait():
            raise RuntimeError("".join(lines))
        report["profiles"][str(count)] = {"data": json.loads(Path(target).read_text()), "log": "".join(lines)}
    if not skip_tests:
        tests = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_cpu_topology.py",
                                "tests/test_cpu_attention_policy.py", "tests/test_modernbert_cpu.py",
                                "tests/test_cpu_short_attention.py"], capture_output=True, text=True,
                               cwd="/workspace")
        report.update(tests=tests.stdout + tests.stderr, test_returncode=tests.returncode)
        print(report["tests"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--threads", default="1,8")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--require-zen3", action="store_true")
    parser.add_argument("--cloud")
    parser.add_argument("--region")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--function", action="store_true", help="Use the Modal Function pool instead of a Sandbox")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    threads = [int(value) for value in args.threads.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("gcc")
        .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("transformers==5.3.0", "pytest")
        .add_local_dir(ROOT / "packed_encoders", "/workspace/packed_encoders", copy=True,
                       ignore=["**/__pycache__/**"])
        .add_local_dir(ROOT / "tests", "/workspace/tests", copy=True, ignore=["**/__pycache__/**"])
        .add_local_file(Path(__file__).with_name("attention_calibration.py"),
                        "/workspace/attention_calibration.py", copy=True)
        .env({"PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1",
              "PACKED_ENCODERS_CPU_CACHE_DIR": "/tmp/packed-encoders-cpu-cache"})
    )
    if args.function:
        app = modal.App("packed-encoders-cpu-calibration")
        function = app.function(image=image, cpu=(args.cpu, args.cpu), memory=4096, timeout=1800,
                                region=args.region, cloud=args.cloud, serialized=True)(function_profile)
        with modal.enable_output(), app.run():
            app_id = app.app_id
            report = function.remote(PROBE, threads, args.rounds, args.require_zen3, args.skip_tests)
        environment = {**report["environment"], "cpu_request": args.cpu, "cpu_limit": args.cpu,
                       "backend": "function", "region": args.region, "cloud": args.cloud,
                       "modal_client": modal.__version__, "app_id": app_id, "terminated": True}
        (args.out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        for count, profile in report["profiles"].items():
            (args.out / f"t{count}.json").write_text(json.dumps(profile["data"], indent=2) + "\n")
            (args.out / f"t{count}.log").write_text(profile["log"])
        if "tests" in report:
            (args.out / "tests.log").write_text(report["tests"])
        if args.require_zen3 and not report["zen3"]:
            raise RuntimeError("The allocated host is not an AVX2-only EPYC Zen 3")
        if report.get("test_returncode", 0):
            raise RuntimeError("CPU tests failed; see tests.log")
        return
    app = modal.App.lookup("packed-encoders-cpu-calibration", create_if_missing=True)
    with modal.enable_output():
        sandbox = modal.Sandbox.create(app=app, image=image, runtime="gvisor", cpu=(args.cpu, args.cpu),
                                       cloud=args.cloud, region=args.region,
                                       memory=4096, timeout=1800, workdir="/workspace")
    metadata = {"sandbox_id": sandbox.object_id, "cpu_request": args.cpu, "cpu_limit": args.cpu,
                "runtime": "gvisor", "cloud": args.cloud, "region": args.region,
                "modal_client": modal.__version__, "terminated": False}
    try:
        metadata.update(json.loads(checked(sandbox, "python", "-c", PROBE)))
        (args.out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
        zen3 = (metadata["vendor"] == "AuthenticAMD" and metadata["family"] == "25"
                and metadata["model"] == "1" and "avx512f" not in metadata["flags"])
        if args.require_zen3 and not zen3:
            raise RuntimeError("The allocated host is not an AVX2-only EPYC Zen 3; refusing to label it as one")
        for count in threads:
            target = f"/tmp/calibration-t{count}.json"
            process = sandbox.exec("python", "attention_calibration.py", "--threads", str(count),
                                   "--rounds", str(args.rounds), "--out", target, timeout=900)
            with (args.out / f"t{count}.log").open("w") as output:
                for line in process.stdout:
                    output.write(line)
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "seconds" in row:
                        print(f"t{count} round {row['round']} {row['method']}: {row['seconds']:.3f} s", flush=True)
                    else:
                        print(f"t{count}: {row}", flush=True)
            stderr = process.stderr.read()
            if process.wait():
                raise RuntimeError(f"Calibration failed: {stderr}")
            (args.out / f"t{count}.json").write_text(checked(sandbox, "cat", target))
        if not args.skip_tests:
            tests = checked(sandbox, "python", "-m", "pytest", "-q", "tests/test_cpu_topology.py",
                            "tests/test_cpu_attention_policy.py", "tests/test_modernbert_cpu.py",
                            "tests/test_cpu_short_attention.py")
            (args.out / "tests.log").write_text(tests)
            print(tests, flush=True)
    finally:
        sandbox.terminate()
        metadata["terminated"] = True
        (args.out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
