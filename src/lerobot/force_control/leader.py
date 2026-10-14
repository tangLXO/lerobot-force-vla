"""Independent read-only 102 sampling with a single immutable latest value.

No robot command, force controller, Sensor cursor, or dataset writer runs here.
The SDK response budget is an acceptance deadline; it cannot interrupt a native
call. The reader thread owns open/read/close, while ``latest`` never performs I/O.
"""

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol

from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import (
    LeaderActionSnapshot,
    LeaderJointDiagnostic,
    LeaderReadError,
)


@dataclass(frozen=True)
class LeaderIntent:
    """One sampling attempt; invalid attempts contain no reusable action.

    ``sequence`` counts publication attempts, including failures. SDK response
    sequence is separate and is present only after a reliable full-joint read.
    All degree keys follow native absolute mapping; arbitrary processors are
    deliberately unsupported until their action-space conversion is specified.
    """

    sequence: int
    is_valid: bool
    raw_action_deg: tuple[tuple[str, float], ...] = ()
    action_deg: tuple[tuple[str, float], ...] = ()
    request_started_ns: int | None = None
    response_completed_ns: int | None = None
    published_ns: int = 0
    source_sequence: int | None = None
    timestamp_source: str = "host_sync_monitor_return"
    action_space: str = "rebot_b601_absolute_degrees_v1"
    error: str | None = None
    joint_diagnostics: tuple[LeaderJointDiagnostic, ...] = ()


class ReadonlyLeader(Protocol):
    """Application contract supplied by the audited leader or an offline fake."""

    def connect_readonly(self) -> None:
        """Open the leader without actuator configuration."""
        ...

    def read_action_snapshot(self, *, max_read_duration_s: float) -> LeaderActionSnapshot:
        """Read a complete, reliable snapshot or raise an exception."""
        ...

    def disconnect(self) -> None:
        """Close only the leader transport."""
        ...


class LeaderSamplerWorker:
    """Own leader I/O in a separate thread; retain only its latest attempt."""

    def __init__(
        self,
        leader: ReadonlyLeader,
        *,
        frequency_hz: float = 50.0,
        max_read_duration_s: float = 0.15,
        action_processor=None,
    ):
        """Validate sampling budgets without opening hardware or starting threads."""
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        if not math.isfinite(max_read_duration_s) or max_read_duration_s <= 0:
            raise ValueError("max_read_duration_s must be finite and positive")
        if action_processor is not None:
            raise ValueError("102 Teacher sampling only supports native absolute degree actions")
        self.leader = leader
        self.frequency_hz = frequency_hz
        self.max_read_duration_s = max_read_duration_s
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest = LeaderIntent(sequence=0, is_valid=False, error="not_started")
        self._thread: threading.Thread | None = None
        self._close_error: str | None = None
        self._resources_owned = False

    @property
    def has_resources(self) -> bool:
        """Whether this worker still owns a running reader/serial lifecycle."""
        return self._resources_owned or (self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        """Launch once; opening a serial port never runs on the owner thread."""
        if self._thread is not None:
            raise RuntimeError("102 worker is single-use")
        self._thread = threading.Thread(target=self._run, name="force-teacher-102-readonly", daemon=True)
        self._thread.start()

    def latest(self, now_ns: int | None = None, max_age_s: float = 0.15) -> LeaderIntent:
        """Return immediately; stale/future data loses its action and validity.

        Age is measured from request start, conservatively including the whole
        SDK read duration. Reading a cached snapshot never refreshes its clock.
        """
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if now_ns is None:
            now_ns = time.perf_counter_ns()
        with self._condition:
            intent = self._latest
        if not intent.is_valid:
            return intent
        if (
            intent.request_started_ns is None
            or intent.response_completed_ns is None
            or intent.request_started_ns > intent.response_completed_ns
            or intent.response_completed_ns > now_ns
            or now_ns - intent.request_started_ns > max_age_s * 1e9
        ):
            return replace(intent, is_valid=False, raw_action_deg=(), action_deg=(), error="stale_or_future")
        return intent

    def wait_for_valid(self, *, timeout_s: float = 2.0, max_age_s: float = 0.15) -> LeaderIntent:
        """Wait outside the owner for startup evidence with a bounded timeout."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                intent = self.latest(max_age_s=max_age_s)
                if intent.is_valid:
                    return intent
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"102 has no valid snapshot: {intent.error}")
                if self._thread is not None and not self._thread.is_alive():
                    raise RuntimeError(f"102 worker stopped: {intent.error}")
                self._condition.wait(remaining)

    def stop(self, *, timeout_s: float = 0.5) -> None:
        """Stop with a bounded join, never close a bus concurrently with its read."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            self._publish_error("stop_timeout_native_read_still_running")
            raise TimeoutError("102 reader did not stop; serial close remains its owner's responsibility")
        if self._close_error is not None:
            raise RuntimeError(self._close_error)

    def _publish_error(
        self,
        error: str,
        *,
        request_started_ns: int | None = None,
        response_completed_ns: int | None = None,
        joint_diagnostics: tuple[LeaderJointDiagnostic, ...] = (),
    ) -> None:
        with self._condition:
            self._latest = LeaderIntent(
                sequence=self._latest.sequence + 1,
                is_valid=False,
                published_ns=time.perf_counter_ns(),
                error=error,
                request_started_ns=request_started_ns,
                response_completed_ns=response_completed_ns,
                joint_diagnostics=joint_diagnostics,
            )
            self._condition.notify_all()

    def _run(self) -> None:
        connected = False
        try:
            self.leader.connect_readonly()
            connected = True
            self._resources_owned = True
            while not self._stop.is_set():
                started = time.perf_counter()
                request_started_ns = time.perf_counter_ns()
                try:
                    snapshot = self.leader.read_action_snapshot(max_read_duration_s=self.max_read_duration_s)
                    if self._stop.is_set():
                        break
                    with self._condition:
                        self._latest = LeaderIntent(
                            sequence=self._latest.sequence + 1,
                            is_valid=True,
                            raw_action_deg=snapshot.raw_action_deg,
                            action_deg=snapshot.action_deg,
                            request_started_ns=snapshot.request_started_ns,
                            response_completed_ns=snapshot.response_completed_ns,
                            published_ns=time.perf_counter_ns(),
                            source_sequence=snapshot.sequence,
                            timestamp_source=snapshot.timestamp_source,
                            joint_diagnostics=snapshot.joint_diagnostics,
                        )
                        self._condition.notify_all()
                except Exception as exc:
                    self._publish_error(
                        f"{type(exc).__name__}: {exc}",
                        request_started_ns=exc.request_started_ns
                        if isinstance(exc, LeaderReadError)
                        else request_started_ns,
                        response_completed_ns=exc.response_completed_ns
                        if isinstance(exc, LeaderReadError)
                        else time.perf_counter_ns(),
                        joint_diagnostics=exc.joint_diagnostics if isinstance(exc, LeaderReadError) else (),
                    )
                # Python 3.12 time.sleep uses a Windows high-resolution timer;
                # Event.wait can quantize a 20 ms period to 31.25 ms. Short
                # sleeps release the GIL and bound stop latency without spinning.
                deadline = started + 1.0 / self.frequency_hz
                while not self._stop.is_set():
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.005))
        except Exception as exc:
            self._resources_owned = bool(getattr(self.leader, "has_resources", connected))
            self._publish_error(f"connect_failed: {type(exc).__name__}: {exc}")
        finally:
            if connected or self._resources_owned:
                try:
                    self.leader.disconnect()
                    self._resources_owned = False
                except Exception as exc:
                    self._close_error = f"102 close failed: {type(exc).__name__}: {exc}"
            self._publish_error(self._close_error or "stopped")


class FakeReadonlyLeader:
    """Stationary synthetic gripper input for offline hybrid runs only."""

    def __init__(self, position_deg: float = 0.0):
        """Construct deterministic input without allocating a transport."""
        if not math.isfinite(position_deg):
            raise ValueError("Fake leader position must be finite")
        self.position_deg = position_deg
        self.connected = False
        self.sequence = 0

    @property
    def has_resources(self) -> bool:
        """Expose the fake connection for lifecycle tests."""
        return self.connected

    def connect_readonly(self) -> None:
        """Mark the synthetic reader as connected."""
        self.connected = True

    def read_action_snapshot(self, *, max_read_duration_s: float) -> LeaderActionSnapshot:
        """Publish a new source sequence and actual host timestamp."""
        if not self.connected:
            raise RuntimeError("Fake leader is disconnected")
        now = time.perf_counter_ns()
        self.sequence += 1
        return LeaderActionSnapshot(
            raw_action_deg=(("gripper.pos", -self.position_deg / 6.0),),
            action_deg=(("gripper.pos", self.position_deg),),
            request_started_ns=now,
            response_completed_ns=now,
            sequence=self.sequence,
            timestamp_source="synthetic_fake_leader",
        )

    def disconnect(self) -> None:
        """Release the synthetic connection."""
        self.connected = False
