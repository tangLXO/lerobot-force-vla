"""B601 control contracts using fake handles only; no CAN/serial access."""

import math
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.force_control.hardware_types import JointFeedback, JointSnapshot
from lerobot.robots.rebot_b601_follower import rebot_b601_follower as driver
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerRobotConfig
from lerobot.robots.rebot_b601_follower.control_io import B601BusLease, audit_motorbridge_capabilities
from lerobot.robots.sensorized_robot import SensorizedRobot
from lerobot.sensors import SensorDataUnavailableError


class FakeMotor:
    def __init__(self):
        self.calls = []
        self.fail = False

    def send_mit(self, *args):
        self.calls.append(("mit", args))
        if self.fail:
            raise OSError("submission uncertain")

    def send_force_pos(self, *args):
        self.calls.append(("force_pos", args))
        if self.fail:
            raise OSError("submission uncertain")

    def send_pos_vel(self, *args):
        self.calls.append(("pos_vel", args))

    def request_feedback(self):
        self.calls.append(("request", ()))

    def close(self):
        self.calls.append(("close", ()))


class FakeBus:
    def __init__(self):
        self.calls = []
        self.motors = []

    def add_damiao_motor(self, *args):
        self.calls.append(("add", args))
        motor = FakeMotor()
        self.motors.append(motor)
        return motor

    def poll_feedback_once(self):
        self.calls.append(("poll", ()))
        raise AssertionError("unbounded SDK poll must not be used by the Teacher path")

    def close(self):
        self.calls.append(("close", ()))


class FakeBoundedSource:
    def __init__(self, clock):
        self.clock = clock
        self.sequence = 0
        self.cached = None
        self.reuse_cache = False
        self.omit = set()
        self.delay_ns = 0
        self.position = -10.0

    def read_snapshot(self, names, *, deadline_ns):
        if self.reuse_cache:
            return self.cached
        self.sequence += 1
        self.clock.now += self.delay_ns
        joints = tuple(
            JointFeedback(
                name, self.position, True, self.clock.now, self.sequence, 0, "host_receive_perf_counter_ns"
            )
            for name in names
            if name not in self.omit
        )
        self.cached = JointSnapshot(joints, self.clock.now, deadline_ns)
        return self.cached


@pytest.fixture
def fixture_robot(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "require_package", lambda *args, **kwargs: None)
    clock = SimpleNamespace(now=1_000_000_000)
    monkeypatch.setattr(driver.time, "perf_counter_ns", lambda: clock.now)
    robot = driver.RebotB601Follower(
        RebotB601FollowerRobotConfig(port=str(tmp_path / "fake-b601"), calibration_dir=tmp_path)
    )
    source = FakeBoundedSource(clock)
    robot.install_joint_feedback_source(source, max_age_ns=50_000_000)
    robot.bus = FakeBus()
    robot.motors = {name: FakeMotor() for name in robot.motor_names}
    return robot, source, clock


def snapshot(robot, clock, names=("gripper",)):
    return robot.read_joint_snapshot(names, deadline_ns=clock.now + 10_000_000)


def test_gripper_only_never_synthesizes_wrist_yaw(fixture_robot):
    robot, _, clock = fixture_robot
    receipt = robot.send_gripper_position(-12.0, feedback_snapshot=snapshot(robot, clock))
    assert receipt.all_submitted
    assert receipt.get("gripper").submitted_deg == -12.0
    assert receipt.get("gripper").hardware_ack == "unknown"
    assert robot.motors["gripper"].calls[0][1][0] == pytest.approx(math.radians(-12.0))
    assert all(not motor.calls for name, motor in robot.motors.items() if name != "gripper")
    assert not robot.bus.calls


def test_relative_and_soft_limits_reuse_current_snapshot(fixture_robot):
    robot, _, clock = fixture_robot
    robot.config.max_relative_target = 2.0
    receipt = robot.send_gripper_position(100.0, feedback_snapshot=snapshot(robot, clock))
    result = receipt.get("gripper")
    assert result.candidate_deg == 100.0
    assert result.clipped_deg == result.submitted_deg == -8.0
    assert not robot.bus.calls


def test_partial_limits_never_reexpand_a_soft_limit(fixture_robot):
    robot, source, clock = fixture_robot
    robot.config.max_relative_target = 2.0
    source.position = 1.0  # Relative interval [-1, 3] intersects soft interval [-270, 0].
    receipt = robot.send_gripper_position(100.0, feedback_snapshot=snapshot(robot, clock))
    assert receipt.get("gripper").submitted_deg == 0.0
    source.position = 4.0  # Relative interval [2, 6] has no legal soft-limit intersection.
    with pytest.raises(ValueError, match="no intersection"):
        robot.send_gripper_position(100.0, feedback_snapshot=snapshot(robot, clock))
    assert len(robot.motors["gripper"].calls) == 1


def test_all_targets_validate_before_any_submission(fixture_robot):
    robot, _, clock = fixture_robot
    feedback = snapshot(robot, clock, ("gripper", "wrist_yaw"))
    with pytest.raises(ValueError):
        robot.send_joint_positions_partial(
            {"gripper": -12, "wrist_yaw": float("nan")}, feedback_snapshot=feedback
        )
    assert all(not motor.calls for motor in robot.motors.values())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "10 deg"])
def test_invalid_degree_values_rejected(fixture_robot, bad):
    robot, _, clock = fixture_robot
    with pytest.raises(ValueError):
        robot.send_gripper_position(bad, feedback_snapshot=snapshot(robot, clock))


def test_unknown_joint_and_invalid_limits_rejected(fixture_robot):
    robot, _, clock = fixture_robot
    feedback = snapshot(robot, clock)
    with pytest.raises(ValueError):
        robot.send_joint_positions_partial({"gripper.pos": -1}, feedback_snapshot=feedback)
    robot.config.joint_limits["gripper"] = (0, -1)
    with pytest.raises(ValueError):
        robot.send_gripper_position(-1, feedback_snapshot=feedback)


def test_partial_submission_reports_unknown_and_not_attempted(fixture_robot):
    robot, _, clock = fixture_robot
    robot.motors["gripper"].fail = True
    names = ("wrist_yaw", "gripper", "wrist_roll")
    receipt = robot.send_joint_positions_partial(
        dict.fromkeys(names, -11), feedback_snapshot=snapshot(robot, clock, names)
    )
    assert [entry.status for entry in receipt.results] == ["success", "unknown", "not_attempted"]
    assert [entry.submitted_deg for entry in receipt.results] == [-11, None, None]
    assert not receipt.all_submitted
    assert not robot.motors["wrist_roll"].calls
    with pytest.raises(RuntimeError, match="latched"):
        robot.send_action({"gripper.pos": -10})


def test_no_sdk_receive_capability_does_not_poll_or_fill_zero(fixture_robot):
    robot, _, clock = fixture_robot
    robot._joint_feedback_source = None
    feedback = snapshot(robot, clock)
    assert not feedback.is_valid
    assert feedback.get("gripper").position_deg is None
    assert feedback.get("gripper").receive_timestamp_ns is None
    assert not robot.bus.calls
    with pytest.raises(ValueError, match="Fresh valid"):
        robot.send_gripper_position(-12, feedback_snapshot=feedback)


def test_reading_cache_again_never_updates_receive_time(fixture_robot):
    robot, source, clock = fixture_robot
    first = snapshot(robot, clock)
    source.reuse_cache = True
    clock.now += 1_000_000
    repeated = snapshot(robot, clock)
    assert not repeated.is_valid
    assert repeated.get("gripper").receive_timestamp_ns == first.get("gripper").receive_timestamp_ns
    assert repeated.get("gripper").age_ns == 1_000_000
    assert repeated.get("gripper").error == "receive_sequence_not_advanced"


def test_missing_joint_in_one_poll_is_not_complete(fixture_robot):
    robot, source, clock = fixture_robot
    source.omit.add("wrist_yaw")
    feedback = snapshot(robot, clock, ("gripper", "wrist_yaw"))
    assert not feedback.is_valid
    assert feedback.get("wrist_yaw").position_deg is None
    assert not feedback.get("wrist_yaw").is_valid


@pytest.mark.parametrize("position", [None, float("nan"), float("inf"), True, "10 deg"])
def test_invalid_feedback_never_reaches_submission(fixture_robot, position):
    robot, source, clock = fixture_robot
    source.position = position
    feedback = snapshot(robot, clock)
    assert not feedback.is_valid
    with pytest.raises(ValueError, match="Fresh valid"):
        robot.send_gripper_position(-12, feedback_snapshot=feedback)
    assert not robot.motors["gripper"].calls


def test_late_read_and_expired_snapshot_refuse_write(fixture_robot):
    robot, source, clock = fixture_robot
    source.delay_ns = 11_000_000
    feedback = snapshot(robot, clock)
    assert not feedback.is_valid and feedback.error == "feedback_deadline_exceeded"
    source.delay_ns = 0
    feedback = snapshot(robot, clock)
    clock.now += 51_000_000
    with pytest.raises(ValueError, match="expired"):
        robot.send_gripper_position(-12, feedback_snapshot=feedback)


def test_foreign_snapshot_is_rejected(fixture_robot):
    robot, _, clock = fixture_robot
    feedback = replace(snapshot(robot, clock))
    with pytest.raises(ValueError, match="latest snapshot"):
        robot.send_gripper_position(-12, feedback_snapshot=feedback)


@pytest.mark.parametrize(
    "method", ["snapshot", "send", "old_send", "old_read", "configure", "calibrate", "disconnect", "fault"]
)
def test_competing_owner_rejected_for_all_entrypoints(fixture_robot, method):
    robot, _, clock = fixture_robot
    feedback = snapshot(robot, clock)
    callbacks = {
        "snapshot": lambda: snapshot(robot, clock),
        "send": lambda: robot.send_gripper_position(-12, feedback_snapshot=feedback),
        "old_send": lambda: robot.send_action({"gripper.pos": -12}),
        "old_read": robot.get_observation,
        "configure": robot.configure,
        "calibrate": robot.calibrate,
        "disconnect": robot.disconnect,
        "fault": lambda: robot.fault_stop_gripper(reason="test"),
    }
    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(RuntimeError, match="different thread"):
        pool.submit(callbacks[method]).result()
    assert all(not motor.calls for motor in robot.motors.values())


def test_forked_owner_identity_is_rejected_before_any_io(fixture_robot, monkeypatch):
    robot, _, _ = fixture_robot
    original_pid = robot._io_owner_pid
    monkeypatch.setattr(driver.os, "getpid", lambda: original_pid + 1)
    with pytest.raises(RuntimeError, match="different process"):
        robot._present_pos()
    assert all(not motor.calls for motor in robot.motors.values())


def test_observe_only_never_enables_zeros_configures_or_disables(fixture_robot, monkeypatch):
    robot, _, _ = fixture_robot
    robot.bus = None
    robot.motors = {}
    bus = FakeBus()
    monkeypatch.setattr(driver, "MotorBridgeController", SimpleNamespace(from_dm_serial=lambda **kwargs: bus))
    robot.connect_observe_only()
    assert [call[0] for call in bus.calls] == ["add"] * len(robot.motor_names)
    assert all(not motor.calls for motor in bus.motors)
    with pytest.raises(RuntimeError, match="observe_only"):
        robot.configure()
    robot.disconnect()
    assert [call[0] for call in bus.calls][-1] == "close"
    assert all(motor.calls == [("close", ())] for motor in bus.motors)


def test_sensorized_narrow_write_health_gate_and_fault_channel(fixture_robot):
    robot, _, clock = fixture_robot
    wrapper = SensorizedRobot(robot, {})
    feedback = snapshot(robot, clock)
    wrapper.latch_fatal_error(RuntimeError("sensor failed"))
    with pytest.raises(SensorDataUnavailableError):
        wrapper.send_gripper_position(-12, feedback_snapshot=feedback)
    with pytest.raises(SensorDataUnavailableError):
        wrapper.send_joint_positions_partial({"gripper": -12}, feedback_snapshot=feedback)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        wrapper.fault_stop_gripper(reason="test stop")
    assert wrapper.has_fatal_error
    assert all(not motor.calls for motor in robot.motors.values())
    with pytest.raises(AttributeError, match="explicit"):
        wrapper._send_joint_position("gripper", -12)


def test_installed_039_audit_refuses_without_opening_native_abi(monkeypatch):
    from lerobot.robots.rebot_b601_follower import control_io

    monkeypatch.setattr(control_io, "version", lambda _: "0.3.9")
    capabilities = audit_motorbridge_capabilities()
    assert not capabilities.teacher_ready
    assert not capabilities.per_joint_receive_evidence
    assert not capabilities.bounded_poll
    with pytest.raises(RuntimeError, match="preflight refused"):
        capabilities.require_teacher_ready()


def test_programmatic_control_preflight_refuses_before_any_owner_or_transport(monkeypatch, tmp_path):
    from lerobot.robots.rebot_b601_follower import control_io

    monkeypatch.setattr(driver, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(control_io, "version", lambda _: "0.3.9")

    def unexpected_open(**kwargs):
        raise AssertionError("preflight must not open hardware")

    monkeypatch.setattr(driver, "MotorBridgeController", SimpleNamespace(from_dm_serial=unexpected_open))
    robot = driver.RebotB601Follower(
        RebotB601FollowerRobotConfig(port=str(tmp_path / "never-open"), calibration_dir=tmp_path)
    )
    with pytest.raises(RuntimeError, match="preflight refused"):
        robot.preflight_control()
    assert robot.bus is None
    assert robot.motors == {}
    assert robot._bus_lease is None
    assert robot._io_owner_id is None


def test_same_bus_instances_exclude_each_other_before_sdk_open(fixture_robot, monkeypatch, tmp_path):
    first, _, _ = fixture_robot
    first.bus = None
    first.motors = {}
    opened = []

    def open_fake(**kwargs):
        bus = FakeBus()
        opened.append(bus)
        return bus

    monkeypatch.setattr(driver, "MotorBridgeController", SimpleNamespace(from_dm_serial=open_fake))
    second = driver.RebotB601Follower(
        RebotB601FollowerRobotConfig(port=first.config.port, calibration_dir=tmp_path)
    )
    first.connect_observe_only()
    try:
        with pytest.raises(RuntimeError, match="already has an owner"):
            second.connect_observe_only()
        assert len(opened) == 1
    finally:
        first.close_observe_only()
    second.connect_observe_only()
    second.close_observe_only()
    assert len(opened) == 2


def test_failed_sdk_open_releases_bus_lease(fixture_robot, monkeypatch):
    robot, _, _ = fixture_robot
    robot.bus = None

    def fail_open(**kwargs):
        raise OSError("fake open failure")

    monkeypatch.setattr(driver, "MotorBridgeController", SimpleNamespace(from_dm_serial=fail_open))
    with pytest.raises(OSError, match="fake open failure"):
        robot.connect_observe_only()
    assert robot._bus_lease is None
    lease = B601BusLease(robot.config.can_adapter, robot.config.port)
    lease.acquire()
    lease.release()


def test_failed_close_retains_bus_lease(fixture_robot, monkeypatch):
    robot, _, _ = fixture_robot
    robot.bus = None
    robot.motors = {}
    bus = FakeBus()
    monkeypatch.setattr(driver, "MotorBridgeController", SimpleNamespace(from_dm_serial=lambda **kwargs: bus))
    robot.connect_observe_only()

    def fail_close():
        raise OSError("fake close failure")

    monkeypatch.setattr(bus, "close", fail_close)
    with pytest.raises(OSError, match="fake close failure"):
        robot.close_observe_only()
    other = B601BusLease(robot.config.can_adapter, robot.config.port)
    with pytest.raises(RuntimeError, match="already has an owner"):
        other.acquire()
    monkeypatch.setattr(bus, "close", lambda: None)
    robot.close_observe_only()
    other.acquire()
    other.release()


def _lease_child(lock_root, connection):
    """Spawn target only locks a file; it cannot construct hardware objects."""
    lease = B601BusLease("damiao", "COM241", lock_root=Path(lock_root))
    try:
        lease.acquire()
        connection.send("acquired")
        connection.recv()
    finally:
        lease.release()
        connection.close()


@pytest.mark.parametrize("terminate", [False, True])
def test_bus_lease_excludes_another_process_and_releases_on_exit(tmp_path, terminate):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_lease_child, args=(str(tmp_path), child))
    process.start()
    child.close()
    try:
        assert parent.poll(10), "mock lease child did not start"
        assert parent.recv() == "acquired"
        # Normalize Windows serial device aliases independently of the host OS.
        lease = B601BusLease("damiao", "\\\\.\\com0241", lock_root=tmp_path)
        with pytest.raises(RuntimeError, match="already has an owner"):
            lease.acquire()
        if terminate:
            process.terminate()
        else:
            parent.send("release")
        process.join(10)
        assert not process.is_alive()
        lease.acquire()
        lease.release()
        assert lease.path.exists()  # Never unlink an inode another owner may have opened.
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
