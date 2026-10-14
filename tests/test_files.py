"""Tests for the shared file helpers."""

import hashlib
import os

import pytest

from agentic_gan.utils.files import sha256_file, write_text_atomic


def test_sha256_matches_known_values(tmp_path):
    abc = tmp_path / "abc.txt"
    abc.write_text("abc")
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    assert sha256_file(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_of_a_file_larger_than_one_chunk(tmp_path):
    data = os.urandom(3 * 1024 * 1024 + 17)
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_of_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "nope")


def test_writes_the_file_and_makes_missing_folders(tmp_path):
    path = tmp_path / "a" / "b" / "out.txt"
    write_text_atomic(path, "hello")
    assert path.read_text() == "hello"


def test_replaces_existing_content_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text("old")
    write_text_atomic(path, "new")
    assert path.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_a_failed_write_keeps_the_old_file_and_cleans_up(tmp_path, monkeypatch):
    path = tmp_path / "out.txt"
    path.write_text("old")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(path, "new")
    assert path.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]
