"""Owner-scoped calibration reads without reopening the file before each axis.

Only the OS file handle is reused. Bytes, SHA256 and path identity are checked
on every use; there is no content/mtime cache or automatic reopen on failure.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
from contextlib import suppress

from .full_arm_control_types import EvidenceFileIdentity

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes


def _open_readonly(path: str) -> io.FileIO:
    """Open unbuffered and permit Windows shared read/write/delete access."""
    if os.name != "nt":
        return io.FileIO(path, "rb")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    # GENERIC_READ; SHARE_READ | SHARE_WRITE | SHARE_DELETE; OPEN_EXISTING.
    # Keep path rename/unlink and in-place edits detectable. Some Windows rename
    # APIs still refuse replacing an open destination; no write permission here.
    handle = create(path, 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise
    # Ownership moves to the descriptor, then FileIO; never CloseHandle twice.
    try:
        os.set_inheritable(fd, False)
        return io.FileIO(fd, "rb", closefd=True)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        raise


class _CalibrationReader:
    """One read-only handle for one admitted bus-owner lifetime."""

    def __init__(self, identity: EvidenceFileIdentity):
        self.identity = identity
        self._stream: io.FileIO | None = None
        self._owner_thread = None
        self._owner_pid = None

    @property
    def has_resources(self) -> bool:
        """Retain an unconfirmed close so owner cleanup can retry it."""
        return self._stream is not None

    def open(self) -> None:
        """Validate the opened object and its contents before transport startup."""
        if self._stream is not None:
            raise RuntimeError("Calibration reader is already open")
        self._owner_thread, self._owner_pid = threading.current_thread(), os.getpid()
        self.identity.require_unchanged()
        self._stream = _open_readonly(self.identity.path)
        try:
            actual = os.fstat(self._stream.fileno())
            if (actual.st_size, actual.st_mtime_ns, actual.st_ino) != (
                self.identity.size_bytes,
                self.identity.mtime_ns,
                self.identity.inode,
            ):
                raise ValueError("Opened calibration differs from the pinned path identity")
            self.require_contents_unchanged()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "Calibration open and cleanup failed", [error, cleanup_error]
                ) from None
            raise

    def require_contents_unchanged(self) -> None:
        """Read fresh bytes between the same two identity checks as the old path."""
        stream = self._stream
        if stream is None or stream.closed:
            raise RuntimeError("Calibration reader is closed; a fresh session is required")
        self._check_owner()
        self.identity.require_unchanged()
        stream.seek(0)
        # An unbuffered, bounded read sees same-size edits even when mtime is
        # restored. Truncation/growth cannot silently reuse earlier bytes.
        raw = stream.read(self.identity.size_bytes + 1)
        self.identity.require_unchanged()
        if hashlib.sha256(raw).hexdigest() != self.identity.sha256:
            raise ValueError(f"Pinned validation file contents changed: {self.identity.path}")

    def close(self) -> None:
        """Idempotent cleanup; no file mutation and no hidden reopening."""
        if self._stream is not None:
            self._check_owner()
            self._stream.close()
            self._stream = None

    def _check_owner(self) -> None:
        """A shared seek cursor must never be used from another thread/process."""
        if self._owner_pid != os.getpid() or self._owner_thread is not threading.current_thread():
            raise RuntimeError("Calibration file I/O belongs to its single owner thread and process")
