"""Independent x86 compiler jobs must finish before linking or cleaning up a failed build."""
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

from packed_encoders.cpu import attention, build
from packed_encoders.errors import PackedEncodersError


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('cpus', [1, 3])
def test_x86_compilers_finish_before_link_or_cleanup(monkeypatch, fail, cpus):
    monkeypatch.setenv('PACKED_ENCODERS_CPU_CACHE_DIR', '')
    monkeypatch.setattr(attention, '_openmp_runtime', lambda: '/tmp/libgomp.so')
    monkeypatch.setattr(attention.shutil, 'which', lambda name: '/usr/bin/gcc')
    monkeypatch.setattr(build.os, 'sched_getaffinity', lambda pid: set(range(cpus)), raising=False)
    barrier = threading.Barrier(cpus, timeout=5)
    failed = threading.Event()
    completed = set()
    compiler_threads = set()
    lock = threading.Lock()
    directories = []
    linked = []

    def run(command, **kwargs):
        output = Path(command[command.index('-o') + 1])
        if '-c' in command:
            with lock:
                compiler_threads.add(threading.get_ident())
            barrier.wait()  # Use all available compiler workers without exceeding affinity.
            if fail and output.name == 'rows.c.o':
                directories.append(output.parent)
                failed.set()
                raise subprocess.CalledProcessError(1, command)
            if fail:
                assert failed.wait(5)
            # A failing compiler must not delete the directory while peers still use it.
            assert output.parent.is_dir()
            output.touch()
            with lock:
                completed.add(output.name)
        else:
            assert not fail
            assert completed == {'rows.c.o', 'local_attention_x86.c.o', 'gemm_x86.c.o'}
            assert all(Path(arg).is_file() for arg in command if arg.endswith('.o'))
            linked.append(output)
            output.touch()

    monkeypatch.setattr(attention.subprocess, 'run', run)
    library = SimpleNamespace()
    monkeypatch.setattr(attention.ctypes, 'CDLL', lambda path: library)
    monkeypatch.setattr(attention, '_signatures', lambda lib: lib)
    if fail:
        with pytest.raises(PackedEncodersError) as caught:
            attention._x86_library.__wrapped__()
        assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)
        peers = {'local_attention_x86.c.o', 'gemm_x86.c.o'}
        # map() may cancel queued jobs after the failure; every already-running peer must finish.
        assert completed <= peers
        if cpus == 3:
            assert completed == peers
        assert not linked
        assert not directories[0].exists()
    else:
        actual, directory = attention._x86_library.__wrapped__()
        try:
            assert actual is library
            assert linked == [Path(directory.name) / 'native.so']
        finally:
            directory.cleanup()
    assert len(compiler_threads) == cpus
