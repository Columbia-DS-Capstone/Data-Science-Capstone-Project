"""Small file helpers shared across the package."""

import hashlib
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in chunks so large files do not fill memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """Write through a temp file in the same folder, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
