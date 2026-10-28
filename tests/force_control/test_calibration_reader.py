"""Real temporary files and OS handles, without any robot or sensor connection."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lerobot.robots.rebot_b601_follower import calibration_reader
from lerobot.robots.rebot_b601_follower.full_arm_control_types import EvidenceFileIdentity


@pytest.fixture
def reader(tmp_path):
    path = tmp_path / "校准.json"
    path.write_bytes(b'{"range_max": 200.0}')
    pin, _ = EvidenceFileIdentity.read(path)
    value = calibration_reader._CalibrationReader(pin)
    value.open()
    yield value
    value.close()


def test_real_handle_is_read_only_unbuffered_and_not_inherited(reader):
    assert not reader._stream.writable()
    assert not os.get_inheritable(reader._stream.fileno())
    with pytest.raises(OSError):
        os.write(reader._stream.fileno(), b"changed")
    reader.require_contents_unchanged()


def test_foreign_thread_cannot_move_or_close_the_shared_file_cursor(reader):
    before = reader._stream.tell()
    with ThreadPoolExecutor(max_workers=1) as pool:
        for operation in (reader.require_contents_unchanged, reader.close):
            with pytest.raises(RuntimeError, match="single owner"):
                pool.submit(operation).result()
    assert reader.has_resources and reader._stream.tell() == before
    reader.require_contents_unchanged()


@pytest.mark.parametrize("change", ["same_stat", "truncate", "grow", "replace", "remove"])
def test_each_read_rejects_mutation_without_reopen_or_stale_python_buffer(reader, monkeypatch, change):
    path = Path(reader.identity.path)
    before = path.stat()
    if change == "replace":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        # Windows MoveFileEx may refuse overwrite of an open destination even
        # with SHARE_DELETE. Rename the old path, then install its replacement;
        # the still-open old object must never masquerade as the new path.
        path.rename(path.with_suffix(".old"))
        replacement.rename(path)
    elif change == "remove":
        path.unlink()
    else:
        raw = path.read_bytes()
        path.write_bytes(
            raw.replace(b"200", b"199")
            if change == "same_stat"
            else b""
            if change == "truncate"
            else raw + b" "
        )
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    monkeypatch.setattr(
        calibration_reader, "_open_readonly", lambda *_: pytest.fail("Failure reopened the file")
    )
    for _ in range(2):
        with pytest.raises((ValueError, OSError)):
            reader.require_contents_unchanged()


def test_normal_checks_reuse_exactly_one_os_handle(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_bytes(b"original")
    pin, _ = EvidenceFileIdentity.read(path)
    calls, original = [], calibration_reader._open_readonly

    def opened(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(calibration_reader, "_open_readonly", opened)
    value = calibration_reader._CalibrationReader(pin)
    assert not value.has_resources
    try:
        value.open()
        for _ in range(100):
            value.require_contents_unchanged()
        assert calls == [pin.path]
    finally:
        value.close()
    value.close()
    with pytest.raises(RuntimeError, match="closed"):
        value.require_contents_unchanged()
    assert calls == [pin.path] and not value.has_resources


def test_replacement_between_path_check_and_open_is_rejected_and_closed(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_bytes(b"original")
    pin, _ = EvidenceFileIdentity.read(path)
    original, opened = calibration_reader._open_readonly, []

    def race(path):
        replacement = Path(path).with_suffix(".replacement")
        replacement.write_bytes(b"original")
        os.utime(replacement, ns=(pin.mtime_ns, pin.mtime_ns))
        replacement.replace(path)
        handle = original(path)
        opened.append(handle)
        return handle

    monkeypatch.setattr(calibration_reader, "_open_readonly", race)
    value = calibration_reader._CalibrationReader(pin)
    with pytest.raises(ValueError, match="pinned path identity"):
        value.open()
    assert opened[0].closed and not value.has_resources


def test_post_read_identity_check_still_detects_a_racing_edit(reader, monkeypatch):
    original = reader._stream

    class Race:
        closed = False

        def seek(self, offset):
            return original.seek(offset)

        def read(self, size):
            raw = original.read(size)
            Path(reader.identity.path).write_bytes(b"changed and grown while reading")
            return raw

        def close(self):
            original.close()

    monkeypatch.setattr(reader, "_stream", Race())
    with pytest.raises(ValueError, match="evidence changed"):
        reader.require_contents_unchanged()


@pytest.mark.skipif(os.name != "nt", reason="Windows kernel handle count")
def test_repeated_reader_lifetimes_do_not_leak_native_handles(tmp_path):
    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    path = tmp_path / "calibration.json"
    path.write_bytes(b"unchanged")
    pin, _ = EvidenceFileIdentity.read(path)
    warm = calibration_reader._CalibrationReader(pin)
    warm.open()
    warm.close()
    before = process.num_handles()
    for _ in range(100):
        value = calibration_reader._CalibrationReader(pin)
        value.open()
        value.require_contents_unchanged()
        value.close()
        assert not value.has_resources
    assert process.num_handles() <= before
