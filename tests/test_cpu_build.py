"""Native cache invalidation, failure recovery and cross-process publication."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from packed_encoders.cpu import build

pytestmark = pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='POSIX native build cache')


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    monkeypatch.setenv('PACKED_ENCODERS_CPU_CACHE_DIR', str(root))
    monkeypatch.setattr(build, '_compiler_identity', lambda *args: ['compiler', 'ISA'])
    calls = []

    def compile_(commands, directory, parallel):
        calls.append(commands)
        (Path(directory) / 'test.so').write_bytes(b'library')

    monkeypatch.setattr(build, '_compile', compile_)
    return root, calls


def test_reuse_invalidation_and_corruption(cache, tmp_path, monkeypatch):
    root, calls = cache
    runtime = tmp_path / 'libgomp.so'
    runtime.write_bytes(b'runtime 1')
    commands = [['gcc', '-march=native', '-o', '{build}/test.so']]
    path, lifetime = build.shared_library('test.so', commands, runtime=runtime)
    assert lifetime is None
    assert build.shared_library('test.so', commands, runtime=runtime)[0] == path
    assert len(calls) == 1
    path.write_bytes(b'truncated')
    assert build.shared_library('test.so', commands, runtime=runtime)[0].read_bytes() == b'library'
    assert len(calls) == 2
    runtime.write_bytes(b'runtime 2')
    assert build.shared_library('test.so', commands, runtime=runtime)[0] != path
    monkeypatch.setattr(build, '_compiler_identity', lambda *args: ['other compiler', 'other ISA'])
    assert build.shared_library('test.so', commands, runtime=runtime)[0] != path
    assert len(calls) == 4
    old_digest = build._digest
    monkeypatch.setattr(build, '_digest', lambda p: 'changed source' if Path(p).name == 'rows.c' else old_digest(p))
    build.shared_library('test.so', commands, runtime=runtime)
    build.shared_library('test.so', [['gcc', '-O2', '-o', '{build}/test.so']], runtime=runtime)
    assert len(calls) == 6


def test_failed_build_is_not_published(cache, monkeypatch):
    root, calls = cache
    original = build._compile

    def fail(commands, directory, parallel):
        original(commands, directory, parallel)
        raise subprocess.CalledProcessError(1, commands)

    monkeypatch.setattr(build, '_compile', fail)
    with pytest.raises(subprocess.CalledProcessError):
        build.shared_library('test.so', [['gcc']])
    assert not list(root.glob('*.so')) and not list(root.glob('build-*'))
    monkeypatch.setattr(build, '_compile', original)
    assert build.shared_library('test.so', [['gcc']])[0].is_file()


def test_corrupted_checksum_is_rebuilt(cache):
    root, calls = cache
    path, _ = build.shared_library('test.so', [['gcc']])
    next(root.glob('*.sha256')).write_bytes(b'\xff')
    assert build.shared_library('test.so', [['gcc']])[0] == path
    assert len(calls) == 2


def test_cache_write_failure_falls_back_after_locking(cache, monkeypatch):
    root, calls = cache
    original = build.tempfile.TemporaryDirectory

    def temporary(*args, **kwargs):
        if kwargs.get('dir') == root:
            raise PermissionError('read-only cache')
        return original(*args, **kwargs)

    monkeypatch.setattr(build.tempfile, 'TemporaryDirectory', temporary)
    path, lifetime = build.shared_library('test.so', [['gcc']])
    try:
        assert lifetime is not None and path.read_bytes() == b'library'
        assert len(calls) == 1
    finally:
        lifetime.cleanup()


def test_xcrun_fingerprints_selected_clang_binary(tmp_path, monkeypatch):
    compiler = tmp_path / 'clang'
    compiler.write_bytes(b'compiler implementation')
    monkeypatch.setattr(build.shutil, 'which', lambda _: '/usr/bin/xcrun')
    monkeypatch.setattr(build.subprocess, 'check_output',
                        lambda command, **kwargs: str(compiler) if '--find' in command else 'clang version')
    monkeypatch.setattr(build.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='ISA macros'))
    identity = build._compiler_identity.__wrapped__(('xcrun', 'clang'), ('-mcpu=apple-m4',))
    assert identity == [str(compiler), build._digest(compiler), 'clang version', 'ISA macros']


@pytest.mark.parametrize('unsafe', ['permissions', 'symlink', 'disabled'])
def test_unsafe_or_disabled_cache_uses_private_build(cache, tmp_path, monkeypatch, unsafe):
    root, _ = cache
    if unsafe == 'permissions':
        root.mkdir(mode=0o755)
        root.chmod(0o755)
    elif unsafe == 'symlink':
        target = tmp_path / 'target'
        target.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
    else:
        monkeypatch.setenv('PACKED_ENCODERS_CPU_CACHE_DIR', '')
    path, lifetime = build.shared_library('test.so', [['gcc']])
    try:
        assert lifetime is not None and root not in path.parents
    finally:
        lifetime.cleanup()


def test_concurrent_fresh_processes_publish_once(tmp_path):
    # A fake build makes the lock race deterministic without requiring an ISA.
    script = '''
import sys, time
from pathlib import Path
from packed_encoders.cpu import build
build._compiler_identity = lambda *args: ['compiler', 'ISA']
def compile_(commands, directory, parallel):
    with open(sys.argv[1], 'a') as log:
        log.write('build\\n')
    time.sleep(.2)
    (Path(directory) / 'test.so').write_bytes(b'library')
build._compile = compile_
path, lifetime = build.shared_library('test.so', [['gcc']])
assert lifetime is None and path.read_bytes() == b'library'
'''
    env = {**os.environ, 'PACKED_ENCODERS_CPU_CACHE_DIR': str(tmp_path / 'cache')}
    log = tmp_path / 'builds'
    processes = [subprocess.Popen([sys.executable, '-c', script, str(log)], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    for process in processes:
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
    assert log.read_text().splitlines() == ['build']
