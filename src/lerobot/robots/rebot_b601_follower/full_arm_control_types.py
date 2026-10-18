"""Shared calibration identity and leader-cache contracts; no motion-evidence admission."""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .readonly_transport import DEFAULT_B601_JOINTS

JOINT_NAMES = tuple(item.name for item in DEFAULT_B601_JOINTS)
ARM_NAMES = JOINT_NAMES[:-1]


def require_sha256(value, label):
    """Require a complete explicit lowercase content digest."""
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} requires a lowercase SHA256")


def leader_configuration_digest(configuration: dict) -> str:
    """Pin the runner's explicit immutable leader port/mapping/calibration projection."""
    if not isinstance(configuration, dict) or not configuration:
        raise ValueError("A nonempty explicit leader configuration projection is required")
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class EvidenceFileIdentity:
    """Content checked at admission; fixed identity checked before control writes."""

    path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    inode: int

    @classmethod
    def read(cls, path: str | Path, expected_sha256: str | None = None):
        """Read and pin one stable resolved file without modifying it."""
        source = Path(path).expanduser().resolve()
        before = source.stat()
        raw = source.read_bytes()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError(f"Evidence changed while reading: {source}")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Evidence SHA256 differs from the explicit binding: {source}")
        return cls(str(source), digest, after.st_size, after.st_mtime_ns, after.st_ino), raw

    def require_unchanged(self):
        """Reject replacement or edits before further control commands."""
        current = os.stat(self.path)
        if (current.st_size, current.st_mtime_ns, current.st_ino) != (
            self.size_bytes,
            self.mtime_ns,
            self.inode,
        ):
            raise ValueError(f"Pinned validation evidence changed: {self.path}")

    def require_contents_unchanged(self):
        """Recheck calibration bytes between fixed identity checks, without reparsing JSON."""
        self.require_unchanged()
        with open(self.path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        self.require_unchanged()
        if digest != self.sha256:
            raise ValueError(f"Pinned validation file contents changed: {self.path}")


@dataclass(frozen=True)
class FullArmLeaderSample:
    """Immutable native-angle cache tied to one explicit mapping identity and epoch."""

    positions_deg: tuple[tuple[str, float], ...]
    captured_ns: int
    response_completed_ns: int
    sequence: int
    source_config_sha256: str
    epoch: int = 0

    def validate(self, configuration, now_ns: int | None = None):
        """Require a complete causal sample within the explicitly configured freshness age."""
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        if len(self.positions_deg) != 7 or set(dict(self.positions_deg)) != set(JOINT_NAMES):
            raise ValueError("Validation leader sample must contain all seven unique native joint positions")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) for _, value in self.positions_deg
        ):
            raise ValueError("Validation leader positions must be finite numbers")
        if any(
            type(value) is not int or value < 0
            for value in (self.captured_ns, self.response_completed_ns, self.epoch)
        ):
            raise ValueError("Validation leader sample requires actual monotonic timestamps/epoch")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("Validation leader sample requires a positive source sequence")
        if not 0 < self.captured_ns <= self.response_completed_ns <= now_ns:
            raise ValueError("Validation leader sample is noncausal")
        if now_ns - self.captured_ns > configuration.leader_max_age_s * 1e9:
            raise ValueError("Validation leader sample is stale")
        if self.source_config_sha256 != configuration.leader_config_sha256:
            raise ValueError("Validation leader mapping/configuration identity changed")


# Compatibility name retained for diagnostic consumers of the same cache contract.
ValidationLeaderSample = FullArmLeaderSample
