"""Threaded bus owner and cache-only recorder view; external work never owns the bus."""

from __future__ import annotations

import copy
import json
import queue
import threading
import time
from dataclasses import dataclass

from .config import positive
from .hardware_types import JointSnapshot
from .runtime import ArmIntent, ForceRuntime
from .telemetry import control_row_summary
from .types import ControlPhase


@dataclass(frozen=True)
class OwnerSnapshot:
    """Isolated tick evidence; a caller cannot mutate the owner's published cache."""

    row: dict
    joint_snapshot: JointSnapshot | None
    last_submitted: tuple[tuple[str, float], ...]
    epoch: int
    mode: str
    tick_id: int


class OwnerSubscription:
    """Bounded completed-tick handoff; overflow invalidates capture, never blocks I/O."""

    def __init__(self, capacity: int):
        """Allocate a fixed queue and a latched capture-only error."""
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Owner subscription capacity must be a positive integer")
        self.queue: queue.Queue[OwnerSnapshot] = queue.Queue(maxsize=capacity)
        self.error: str | None = None
        self.closed = False

    def check_health(self) -> None:
        """Reject a capture after any completed tick was lost."""
        if self.error is not None:
            raise RuntimeError(self.error)

    def get(self, timeout: float = 0.05) -> OwnerSnapshot:
        """Read the next detached completed tick with a bounded wait."""
        self.check_health()
        return self.queue.get(timeout=timeout)


class OwnerService:
    """Run connect/read/poll/partial writes/fault disposition/close on one thread."""

    def __init__(self, runtime: ForceRuntime):
        """Construct without opening any transport or acquiring a sensor consumer cursor."""
        self.runtime = runtime
        self._condition = threading.Condition()
        self._shutdown = threading.Event()
        self._thread = None
        self._error = None
        self._latest = None
        self._fault_origin = None
        self._command_version = -1
        self._arm_version = -1
        self._reset_version = None
        self._submission_lock = threading.Lock()
        self._subscriptions: list[OwnerSubscription] = []

    def subscribe(self, capacity: int = 128) -> OwnerSubscription:
        """Subscribe to future complete ticks without reading any hardware."""
        subscription = OwnerSubscription(capacity)
        with self._condition:
            self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: OwnerSubscription) -> None:
        """Freeze a subscription boundary so a recorder can drain before saving."""
        with self._condition:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
            subscription.closed = True

    @property
    def is_running(self):
        """Whether the owner worker is alive and has not been asked to stop."""
        return self._thread is not None and self._thread.is_alive() and not self._shutdown.is_set()

    @property
    def has_resources(self):
        """Retain dependencies until both the worker and its transport have closed."""
        return (self._thread is not None and self._thread.is_alive()) or bool(
            getattr(self.runtime.device, "has_resources", False)
        )

    @property
    def fault_origin(self) -> OwnerSnapshot | None:
        """Return the first failed tick in the current fault epoch, detached from owner state."""
        with self._condition:
            return copy.deepcopy(self._fault_origin)

    def start(self, timeout_s=2.0):
        """Wait boundedly for first tick; startup failure never connects in the caller thread."""
        positive("timeout_s", timeout_s)
        if self._thread is not None:
            raise RuntimeError("OwnerService is single-use")
        self._thread = threading.Thread(target=self._run, name="force-bus-owner", daemon=False)
        self._thread.start()
        try:
            return self._wait(lambda snapshot: snapshot, timeout_s)
        except BaseException:
            self.stop(timeout_s)
            raise

    def _publish(self, row):
        snapshot = OwnerSnapshot(
            copy.deepcopy(control_row_summary(row)),
            self.runtime.latest_joint_snapshot,
            tuple(self.runtime.last_submitted.items()),
            self.runtime.epoch,
            self.runtime.mode,
            row["tick_id"],
        )
        with self._condition:
            if snapshot.row.get("phase") == str(ControlPhase.FAULT):
                if self._fault_origin is None or self._fault_origin.epoch != snapshot.epoch:
                    self._fault_origin = snapshot
            else:
                self._fault_origin = None
            self._latest = snapshot
            for subscription in self._subscriptions:
                if subscription.error is None:
                    try:
                        subscription.queue.put_nowait(copy.deepcopy(snapshot))
                    except queue.Full:
                        subscription.error = (
                            "Completed control-tick queue overflow; demonstration is incomplete"
                        )
            self._condition.notify_all()

    def _run(self):
        opened = False
        try:
            self.runtime.open()
            opened = True
            period = round(1e9 / self.runtime.config.control_hz)
            deadline = time.perf_counter_ns()
            while not self._shutdown.is_set():
                self._publish(self.runtime.tick())
                deadline += period
                now = time.perf_counter_ns()
                if now >= deadline:
                    deadline += ((now - deadline) // period + 1) * period
                # Windows Event.wait can round a 20 ms period to coarse timer ticks.
                # time.sleep uses the high-resolution waitable timer; short pieces bound stop latency.
                while not self._shutdown.is_set():
                    remaining = (deadline - time.perf_counter_ns()) / 1e9
                    if remaining <= 0:
                        break
                    time.sleep(min(self.runtime.config.scheduler_sleep_slice_ms / 1000, remaining))
            if (
                self.runtime.controller.phase not in (ControlPhase.DISABLED, ControlPhase.FAULT)
                or self.runtime.mode == "TELEOP"
            ):
                self.runtime.request_stop()
                self._publish(self.runtime.tick())
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            try:
                if opened:
                    self.runtime.close()
            except BaseException as exc:
                with self._condition:
                    self._error = (
                        exc
                        if self._error is None
                        else BaseExceptionGroup("Owner execution and cleanup failed", [self._error, exc])
                    )
            with self._condition:
                self._condition.notify_all()

    def stop(self, timeout_s=2.0):
        """Wake the owner and boundedly join; active motion receives owner fault disposition."""
        positive("timeout_s", timeout_s)
        self.runtime.request_stop()
        self._shutdown.set()
        self._join(timeout_s)

    def quiesce_after_release(self, timeout_s=2.0):
        """Close a normally released owner without setting the shared user-stop event."""
        positive("timeout_s", timeout_s)
        with self._condition:
            if (
                self._latest is None
                or self._latest.row.get("phase") != str(ControlPhase.DISABLED)
                or self.runtime.controller.phase != ControlPhase.DISABLED
                or self.runtime.mode == "TELEOP"
            ):
                raise RuntimeError("Normal owner quiescence requires completed release to DISABLED")
            self._shutdown.set()
        self._join(timeout_s)

    def _join(self, timeout_s):
        """Retain ownership until close completes; both stop paths share the same diagnostics."""
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("Owner has not stopped; transport ownership remains retained")
        if self._error is not None:
            raise self._error

    def post(self, kind, value=None):
        """Enqueue a mode/reference event without performing any bus operation."""
        with self._submission_lock:
            self._command_version += 1
            epoch = self.runtime.epoch
            version = self._command_version
            if not self.runtime.post(kind, version=version, epoch=epoch, value=value):
                raise RuntimeError("Owner event rejected after ownership changed")
            self._reset_version = version if kind == "reset" else None
            return version

    def submit_arm_intent(self, action, *, captured_ns, version=None, epoch=None, ttl_ms=None, metadata=None):
        """Publish timestamped degree intent, preserving capture age across processing and queueing."""
        with self._submission_lock:
            if version is None:
                version = self._arm_version + 1
            ttl_ms = self.runtime.config.command_ttl_ms if ttl_ms is None else ttl_ms
            positive("ttl_ms", ttl_ms)
            intent = ArmIntent(
                tuple(action.items()),
                version,
                self.runtime.epoch if epoch is None else epoch,
                captured_ns,
                round(ttl_ms * 1e6),
                json.dumps(metadata or {}, allow_nan=False),
            )
            if not self.runtime.post_arm_intent(intent):
                raise RuntimeError("Arm intent rejected: stale version or ownership epoch")
            self._arm_version = max(self._arm_version, version)
            return version

    def snapshot(self):
        """Copy the latest published evidence without reading a device or sensor."""
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._latest is None:
                raise RuntimeError("No owner tick has been published")
            return copy.deepcopy(self._latest)

    def _wait(self, predicate, timeout_s, *, allow_fault=False):
        positive("timeout_s", timeout_s)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                snapshot = self._latest
                if snapshot is not None:
                    if not allow_fault and snapshot.row.get("phase") == str(ControlPhase.FAULT):
                        origin = self._fault_origin or snapshot
                        error = RuntimeError(f"Owner fault: {origin.row.get('failure_reason')}")
                        if origin.row.get("feedback_valid") is False and origin.row.get("joint_error"):
                            error.add_note(f"joint_error: {origin.row['joint_error']}")
                        raise error
                    if predicate(snapshot):
                        return copy.deepcopy(snapshot)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for owner evidence")
                if self._shutdown.is_set():
                    raise RuntimeError("Owner stopped before requested evidence")
                self._condition.wait(remaining)

    def await_tick_after(self, anchor_ns, arm_version, timeout_s=1.0):
        """Bind a frame only to a later, entirely submitted tick for its exact arm intent."""
        if type(anchor_ns) is not int or anchor_ns < 0:
            raise ValueError("anchor_ns must be an actual monotonic frame timestamp")

        def matches(snapshot):
            row = snapshot.row
            version = row.get("arm_command_version")
            if arm_version is not None and version is not None and version > arm_version:
                raise RuntimeError("Frame arm intent was superseded before binding")
            return (
                version == arm_version
                and row["decision_anchor_ns"] >= anchor_ns
                and row["send_status"] == "submitted"
                and not row["safety_override"]
            )

        return self._wait(matches, timeout_s)

    def wait_for_phase(self, phase, timeout_s=1.0):
        """Wait for a recorded phase transition without polling or sleeping in the owner."""
        return self._wait(lambda snapshot: snapshot.row["phase"] == str(phase), timeout_s)

    def wait_for_command(self, version, timeout_s=1.0):
        """Wait for an explicit event acknowledgement, rejecting a superseded event."""

        def matches(snapshot):
            observed = snapshot.row.get("command_version")
            if observed is not None and observed > version:
                raise RuntimeError("Owner event was superseded")
            return observed == version

        return self._wait(matches, timeout_s, allow_fault=self._reset_version == version)


class CachedRobotProxy:
    """Recorder-only position cache; conventional send_action can never bypass the owner."""

    name = "force_owner_cache"
    robot_type = "force_owner_cache"
    id = "force-owner"
    cameras = {}

    def __init__(self, service: OwnerService):
        """Retain the service, not its hardware adapter."""
        self.service = service
        self.last_owner_snapshot = None
        self.last_joint_snapshot = None

    @property
    def observation_features(self):
        """Robot state includes joint positions only; tactile routing belongs to SensorizedRobot."""
        return {f"{name}.pos": float for name in self.service.runtime.config.joint_names}

    @property
    def action_features(self):
        """Define degree position keys matching explicit owner partial submissions."""
        return self.observation_features

    @property
    def is_connected(self):
        """Reflect the owner lifecycle without touching the transport."""
        return self.service.is_running

    @property
    def is_calibrated(self):
        """No calibration is performed by a cache view."""
        return True

    def get_observation(self):
        """Return measured values with preserved real receive provenance, never cache-read time."""
        snapshot = self.service.snapshot()
        joints = snapshot.joint_snapshot
        if snapshot.row.get("phase") == str(ControlPhase.FAULT) or joints is None or not joints.is_valid:
            raise RuntimeError("Owner feedback cache is invalid or faulted")
        now = time.perf_counter_ns()
        max_age = self.service.runtime.safety.config.joint_feedback_max_age_ms * 1e6
        if any(
            item.receive_timestamp_ns is None or not 0 <= now - item.receive_timestamp_ns <= max_age
            for item in joints.joints
        ):
            raise RuntimeError("Owner feedback cache is stale")
        self.last_owner_snapshot, self.last_joint_snapshot = snapshot, joints
        return {f"{item.name}.pos": item.position_deg for item in joints.joints}

    def send_action(self, action):
        """Reject the legacy recorder write path, which cannot represent asynchronous ownership."""
        raise RuntimeError("Use timestamped OwnerService.submit_arm_intent and await its receipt")

    def connect(self, calibrate=False):
        """Require explicit owner startup; never connect through the legacy robot entry point."""
        if not self.service.is_running:
            raise RuntimeError("Start the owner explicitly before exposing its cache")

    def disconnect(self):
        """Request owner teardown; all actual bus cleanup remains in the owner worker."""
        self.service.stop()

    def calibrate(self):
        """Forbid calibration side effects through the recorder view."""
        raise RuntimeError("A cache proxy cannot calibrate hardware")

    def configure(self):
        """Forbid control-mode changes through the recorder view."""
        raise RuntimeError("A cache proxy cannot configure hardware")
