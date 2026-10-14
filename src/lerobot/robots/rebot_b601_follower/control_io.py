"""Fail-closed capability audit for the separately reviewed B601 control path."""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _bus_identity(adapter: str, port: str) -> str:
    if adapter not in ("damiao", "socketcan") or not isinstance(port, str) or not port.strip():
        raise ValueError("A supported B601 adapter and nonempty port are required")
    port = port.strip()
    if adapter == "damiao":
        # Windows accepts COM13 and \\.\COM13 for the same serial device.
        candidate = port.removeprefix("\\\\.\\")
        if re.fullmatch(r"COM[0-9]+", candidate, flags=re.IGNORECASE):
            port = f"COM{int(candidate[3:])}"
        else:
            # Resolve POSIX /dev/serial/by-id aliases before deriving ownership.
            port = os.path.normcase(os.path.realpath(port))
    return f"{adapter}:{port}"


class B601BusLease:
    """Nonblocking OS lease shared by all B601 entrypoints in this checkout.

    The lock is advisory: external SDK users or other software that ignore this
    protocol must still be stopped by the operator. The file is intentionally
    never unlinked; unlinking could let a competing process lock a different
    inode. Closing the descriptor or process termination releases the OS lock.
    """

    def __init__(self, adapter: str, port: str, *, lock_root: Path | None = None):
        identity = _bus_identity(adapter, port)
        key = hashlib.sha256(identity.encode()).hexdigest()
        root = Path(tempfile.gettempdir()) / "lerobot-b601-bus-locks" if lock_root is None else lock_root
        self.path = root / f"{key}.lock"
        self._handle = None
        self._pid = os.getpid()

    def acquire(self) -> None:
        """Acquire before opening a transport; never wait behind another owner."""
        if self._handle is not None:
            raise RuntimeError("B601 bus lease is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                "B601 bus already has an owner, or its ownership lock is inaccessible"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        """Release only after confirmed transport closure in the creating process."""
        if os.getpid() != self._pid:
            raise RuntimeError("A forked process cannot release its parent's B601 lease")
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class MotorbridgeCapabilities:
    version: str | None
    per_joint_receive_evidence: bool
    bounded_poll: bool
    command_acknowledgement: bool
    validated_fault_stop: bool
    reasons: tuple[str, ...]

    @property
    def teacher_ready(self) -> bool:
        return self.per_joint_receive_evidence and self.bounded_poll and self.validated_fault_stop

    def require_teacher_ready(self) -> None:
        if not self.teacher_ready:
            raise RuntimeError("B601 Teacher hardware preflight refused: " + "; ".join(self.reasons))


def audit_motorbridge_capabilities() -> MotorbridgeCapabilities:
    """Inspect package metadata without loading its native ABI or opening a bus.

    Reviewed 0.3.9 Python sources: ``core.py`` Controller.poll_feedback_once has
    no timeout argument; Motor.get_state wraps CState without RX time/sequence.
    ``abi.py`` CState and ``models.py`` MotorState confirm those missing fields.
    Native v0.3.9 ``motor_core/src/controller.rs`` also starts a polling thread
    when adding a motor; application polling would compete with that reader.
    Neither send method returns command-level acknowledgement. Other versions
    require a new explicit source/ABI review; version changes never opt in.
    """
    try:
        installed = version("motorbridge")
    except PackageNotFoundError:
        installed = None
    if installed == "0.3.9":
        reasons = (
            "motorbridge 0.3.9 MotorState/CState expose no receive timestamp, sequence, or callback",
            "poll_feedback_once has no timeout/deadline parameter or verified maximum duration",
            "native add_device starts an internal polling thread; application poll would compete with it",
            "send methods expose software return status only, not command acknowledgement",
            "a supported-object gripper fault stop/hold has not been validated on hardware",
        )
    else:
        reasons = (f"motorbridge version {installed!r} has no reviewed bounded feedback/fault adapter",)
    return MotorbridgeCapabilities(installed, False, False, False, False, reasons)
