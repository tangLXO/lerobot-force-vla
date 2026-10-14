"""Immutable I/O evidence shared by the runtime and robot adapters.

Successful software submission is deliberately distinct from hardware execution.
All timestamps, when available, use this host's ``perf_counter_ns`` clock domain.
"""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class JointFeedback:
    """One joint's position and real receive provenance, or explicit invalidity."""

    name: str
    position_deg: float | None
    is_valid: bool
    receive_timestamp_ns: int | None
    receive_sequence: int | None
    age_ns: int | None
    time_source: str = "unavailable"
    error: str | None = None


@dataclass(frozen=True)
class JointSnapshot:
    """Immutable result of one bounded joint acquisition attempt."""

    joints: tuple[JointFeedback, ...]
    completed_ns: int
    deadline_ns: int
    error: str | None = None

    def get(self, name: str) -> JointFeedback:
        """Return the named evidence entry; missing entries are never zero-filled."""
        return next(joint for joint in self.joints if joint.name == name)

    @property
    def is_valid(self) -> bool:
        """Whether every requested joint supplied valid acquisition evidence."""
        return bool(self.joints) and self.error is None and all(joint.is_valid for joint in self.joints)


@dataclass(frozen=True)
class JointSendResult:
    """Per-joint submission evidence; an SDK exception leaves execution unknown."""

    name: str
    candidate_deg: float
    clipped_deg: float
    submitted_deg: float | None
    status: Literal["success", "unknown", "not_attempted"]
    send_start_ns: int | None
    send_end_ns: int | None
    hardware_ack: Literal["unknown"] = "unknown"
    error: str | None = None


@dataclass(frozen=True)
class SendReceipt:
    """Sequential submission results without implying an atomic bus transaction."""

    results: tuple[JointSendResult, ...]
    send_start_ns: int
    send_end_ns: int

    def get(self, name: str) -> JointSendResult:
        """Return the result of the named candidate command."""
        return next(result for result in self.results if result.name == name)

    @property
    def all_submitted(self) -> bool:
        """Whether every SDK submission returned without an exception."""
        return bool(self.results) and all(result.status == "success" for result in self.results)


class BoundedJointFeedbackSource(Protocol):
    """Explicit adapter contract, NOT attributes purportedly offered by motorbridge.

    An implementation must own real receive evidence and bound all request/poll
    calls by the supplied deadline. No production implementation is supplied for
    motorbridge 0.3.9, whose public ABI cannot meet that contract.
    """

    def read_snapshot(self, names: tuple[str, ...], *, deadline_ns: int) -> JointSnapshot:
        """Obtain actual new receive evidence within a bounded I/O deadline."""
        ...
