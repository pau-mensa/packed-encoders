"""Private, content-addressed native build cache for Linux and macOS."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _compiler_digest(path, size, modified, changed):
    # Apple clang can be large. Hash it once per process, across SME target flags,
    # without retaining the binary in memory. Stat fields distinguish updates.
    return _digest(path)


@lru_cache(maxsize=16)
def _compiler_identity(prefix, target_flags):
    executable = shutil.which(prefix[0])
    if executable is None:
        raise OSError(f"compiler not found: {prefix[0]}")
    if Path(prefix[0]).name == 'xcrun':
        executable = subprocess.check_output([prefix[0], '--find', prefix[1]], text=True, timeout=15).strip()
    version = subprocess.check_output([*prefix, '--version'], text=True, timeout=15)
    macros = subprocess.run([*prefix, *target_flags, '-dM', '-E', '-x', 'c', '-'],
                            input='', text=True, capture_output=True, check=True, timeout=15).stdout
    executable = Path(executable).resolve()
    info = executable.stat()
    checksum = _compiler_digest(str(executable), info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return [str(executable), checksum, version, macros]


def _key(commands, runtime):
    first = commands[0]
    prefix = tuple(first[:2] if Path(first[0]).name == 'xcrun' else first[:1])
    flags = tuple(arg for arg in first if arg.startswith(('-march=', '-mcpu=', '-mmacosx-version-min=')))
    here = Path(__file__).parent
    identity = {'schema': 1, 'commands': commands, 'compiler': _compiler_identity(prefix, flags),
                'platform': [platform.system(), platform.machine(), platform.release(), platform.mac_ver()[0]],
                'sources': {p.name: _digest(p) for p in sorted(here.iterdir()) if p.suffix in ('.c', '.h')},
                'runtime': [str(Path(runtime).resolve()), _digest(runtime)] if runtime else None,
                'environment': {name: os.environ.get(name) for name in
                                ('PATH', 'DEVELOPER_DIR', 'SDKROOT', 'MACOSX_DEPLOYMENT_TARGET', 'CPATH',
                                 'C_INCLUDE_PATH', 'LIBRARY_PATH', 'GCC_EXEC_PREFIX', 'COMPILER_PATH')}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _cache_root():
    configured = os.environ.get('PACKED_ENCODERS_CPU_CACHE_DIR')
    if configured == '':
        return None
    if configured is not None:
        root = Path(configured).expanduser()
    elif platform.system() == 'Darwin':
        root = Path.home() / 'Library/Caches/packed-encoders/cpu'
    else:
        root = Path(os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')) / 'packed-encoders/cpu'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    # Do not load executable code from a cache writable by other users.
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise OSError('native cache must be a private directory owned by the current user')
    return root


def _compile(commands, directory, parallel):
    def run(command):
        subprocess.run([arg.replace('{build}', str(directory)) for arg in command],
                       check=True, capture_output=True, timeout=60)

    if parallel and len(commands) > 1:
        try:
            cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpus = os.cpu_count() or 1
        # Join all jobs before linking or cleaning up, including on failure.
        with ThreadPoolExecutor(max(1, min(len(commands) - 1, cpus))) as pool:
            list(pool.map(run, commands[:-1]))
        run(commands[-1])
    else:
        for command in commands:
            run(command)


def shared_library(name, commands, *, runtime=None, parallel=False):
    """Return (library path, TemporaryDirectory lifetime or None).

    Commands use ``{build}`` for their output directory. An empty
    PACKED_ENCODERS_CPU_CACHE_DIR disables disk reuse. Unavailable/untrusted
    caches fall back to a process-private build; compiler failures still propagate.
    """
    try:
        root = _cache_root()
        key = _key(commands, runtime) if root is not None else None
    except (OSError, subprocess.SubprocessError):
        root = None
    if root is not None:
        import fcntl
        try:
            fd = os.open(root / f'{key}.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        except OSError:
            root = None
        else:
            try:
                with os.fdopen(fd, 'rb') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    destination = root / f'{key}-{name}'
                    checksum = root / f'{key}.sha256'
                    try:
                        if (destination.is_file() and not destination.is_symlink()
                                and _digest(destination) == checksum.read_text()):
                            return destination, None
                    except (OSError, UnicodeError):
                        pass
                    # Only publish complete artifacts, while holding the cross-process lock.
                    with tempfile.TemporaryDirectory(prefix='build-', dir=root) as staging:
                        _compile(commands, staging, parallel)
                        artifact = Path(staging) / name
                        digest = _digest(artifact)
                        os.replace(artifact, destination)
                        checksum.write_text(digest)
                    return destination, None
            except OSError:
                # Cache I/O is optional, even on read-only or full filesystems.
                pass
    directory = tempfile.TemporaryDirectory(prefix='packed-encoders-native-')
    try:
        _compile(commands, directory.name, parallel)
        return Path(directory.name) / name, directory
    except BaseException:
        directory.cleanup()
        raise
