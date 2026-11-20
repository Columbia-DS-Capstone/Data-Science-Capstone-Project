"""One OpenMP runtime for torch, scikit-learn and LightGBM on macOS.

Each of the three loads its own libomp.dylib. Once torch's copy is in the process, a LightGBM
fit segfaults or deadlocks. macOS reads DYLD_LIBRARY_PATH only when a process starts, so the
interpreter is started again with that variable pointing at scikit-learn's bundled copy.
scripts/run_e2e_bank.sh and web.py set the same variable before they launch the orchestrator.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path

LIBRARY_PATH = "DYLD_LIBRARY_PATH"
# Set on the restarted process. If macOS strips DYLD_LIBRARY_PATH, this stops a restart loop.
RESTART_MARKER = "AGENTIC_GAN_OPENMP_RESTARTED"


def sklearn_openmp_dir() -> Path | None:
    """The folder that holds scikit-learn's bundled libomp.dylib, or None. Imports nothing."""
    spec = importlib.util.find_spec("sklearn")
    if spec is None or spec.origin is None:
        return None
    folder = Path(spec.origin).parent / ".dylibs"
    return folder if (folder / "libomp.dylib").exists() else None


def shared_openmp_env(
    environ: Mapping[str, str],
    *,
    platform: str = sys.platform,
    library_dir: Path | None = None,
) -> dict[str, str] | None:
    """The environment for a restarted process, or None when no restart is needed."""
    if platform != "darwin" or RESTART_MARKER in environ:
        return None
    folder = library_dir if library_dir is not None else sklearn_openmp_dir()
    if folder is None:
        return None
    entries = [entry for entry in environ.get(LIBRARY_PATH, "").split(os.pathsep) if entry]
    if any(Path(entry) == folder for entry in entries):
        return None
    return {
        **environ,
        LIBRARY_PATH: os.pathsep.join([str(folder), *entries]),
        RESTART_MARKER: "1",
    }


def use_shared_openmp() -> None:
    """Restart this interpreter with the shared runtime. Returns only when none is needed."""
    env = shared_openmp_env(os.environ)
    if env is None or not sys.executable:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
