"""The orchestrator restarts once on macOS so torch and LightGBM share one OpenMP runtime."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_gan.utils import openmp
from agentic_gan.utils.openmp import (
    LIBRARY_PATH,
    RESTART_MARKER,
    shared_openmp_env,
    sklearn_openmp_dir,
    use_shared_openmp,
)

FOLDER = Path("/venv/site-packages/sklearn/.dylibs")


def test_darwin_gets_the_library_path_and_the_marker():
    env = shared_openmp_env({"HOME": "/home/me"}, platform="darwin", library_dir=FOLDER)
    assert env == {"HOME": "/home/me", LIBRARY_PATH: str(FOLDER), RESTART_MARKER: "1"}


def test_existing_library_path_is_kept_after_the_shared_folder():
    env = shared_openmp_env({LIBRARY_PATH: "/opt/lib"}, platform="darwin", library_dir=FOLDER)
    assert env[LIBRARY_PATH] == f"{FOLDER}{os.pathsep}/opt/lib"


@pytest.mark.parametrize(
    ("environ", "platform"),
    [
        ({}, "linux"),
        ({}, "win32"),
        ({RESTART_MARKER: "1"}, "darwin"),
        ({LIBRARY_PATH: f"/opt/lib{os.pathsep}{FOLDER}"}, "darwin"),
    ],
)
def test_no_restart_when_not_needed(environ, platform):
    assert shared_openmp_env(environ, platform=platform, library_dir=FOLDER) is None


def test_no_restart_without_a_bundled_library(monkeypatch):
    monkeypatch.setattr(openmp, "sklearn_openmp_dir", lambda: None)
    assert shared_openmp_env({}, platform="darwin") is None


def test_restart_keeps_the_interpreter_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(openmp, "shared_openmp_env", lambda environ: {"MARK": "1"})
    monkeypatch.setattr(sys, "executable", "/venv/bin/python")
    monkeypatch.setattr(
        sys, "orig_argv", ["python", "-u", "-m", "agentic_gan.orchestrator", "--dataset", "default"]
    )
    monkeypatch.setattr(os, "execve", lambda *args: calls.append(args))
    use_shared_openmp()
    assert calls == [
        (
            "/venv/bin/python",
            ["/venv/bin/python", "-u", "-m", "agentic_gan.orchestrator", "--dataset", "default"],
            {"MARK": "1"},
        )
    ]


def test_no_restart_leaves_the_process_alone(monkeypatch):
    monkeypatch.setattr(openmp, "shared_openmp_env", lambda environ: None)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("the process was restarted"))
    use_shared_openmp()


PROBE = """
from agentic_gan.utils.openmp import use_shared_openmp
use_shared_openmp()
import numpy as np
import torch
from lightgbm import LGBMClassifier
gru = torch.nn.GRU(8, 48, batch_first=True)
out, _ = gru(torch.randn(128, 6, 8))
out.pow(2).mean().backward()
rng = np.random.default_rng(0)
X = rng.random((5000, 20))
LGBMClassifier(n_estimators=20, verbose=-1).fit(X, (X[:, 0] > 0.5).astype(int))
print("FIT_OK")
"""


@pytest.mark.skipif(
    sys.platform != "darwin"
    or sklearn_openmp_dir() is None
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("lightgbm") is None,
    reason="needs macOS with torch, LightGBM and scikit-learn's bundled libomp",
)
def test_lightgbm_fits_after_torch_in_one_process():
    """Without the restart this process segfaults or hangs in the LightGBM fit."""
    environ = {
        name: value
        for name, value in os.environ.items()
        if name not in (LIBRARY_PATH, RESTART_MARKER)
    }
    done = subprocess.run(
        [sys.executable, "-c", PROBE], env=environ, capture_output=True, text=True, timeout=180
    )
    assert done.returncode == 0, done.stderr[-500:]
    assert "FIT_OK" in done.stdout
