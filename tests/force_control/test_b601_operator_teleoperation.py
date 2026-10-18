"""Normal operator lifecycle with a fake bus; no physical device or historical trial files."""

import math
import struct
from dataclasses import asdict, replace

import pytest

from lerobot.robots.rebot_b601_follower.control_backend import B601ControlBackend
from lerobot.robots.rebot_b601_follower.control_protocol import (
    encode_disable,
    encode_enable,
    encode_mode_write,
)
from lerobot.robots.rebot_b601_follower.full_arm_control_types import JOINT_NAMES
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from tests.force_control.operator_teleoperation_helpers import (
    close_operator_scope,
    make_operator_files,
    make_operator_scope,
)


@pytest.fixture
def scope(tmp_path, monkeypatch):
    value = make_operator_scope(tmp_path, monkeypatch)
    yield value
    close_operator_scope(value)


def start(scope, guard=lambda: None):
    return scope.device.start_operator_teleoperation(guard=guard, leader_sample=scope.leader)


def make_logged_timeout_scope(tmp_path, monkeypatch, *, connect=True):
    """Reproduce recorded six-arm mode2/TIMEOUT0 and gripper mode4/TIMEOUT2000."""
    files = make_operator_files(tmp_path)
    files.operator = files.configuration = replace(files.operator, watchdog_ticks=2000)
    files.control = replace(files.control, watchdog_ticks=2000)
    value = make_operator_scope(tmp_path, monkeypatch, files=files, connect=False)
    value.serial.watchdog_readback = 0
    value.serial.watchdog_overrides[7] = 2000
    if connect:
        value.device.connect()
    return value


@pytest.fixture
def logged_timeout_scope(tmp_path, monkeypatch):
    value = make_logged_timeout_scope(tmp_path, monkeypatch)
    yield value
    close_operator_scope(value)


def movement_packets(serial):
    return [packet for packet in serial.writes if int.from_bytes(packet[13:17], "little") != 0x7FF]


def test_query_only_connection_and_explicit_all_joint_start(scope):
    assert not movement_packets(scope.serial)
    assert scope.device.operator_teleoperation_config is scope.files.operator
    assert scope.device.full_arm_validation_config is None
    with pytest.raises(RuntimeError, match="validation capability"):
        scope.device.start_full_arm_validation(guard=lambda: None, leader_sample=scope.leader)
    final = start(scope)
    assert final.is_valid and len(final.joints) == 7
    packets = movement_packets(scope.serial)
    first_enable = next(index for index, packet in enumerate(packets) if packet == encode_enable(1))
    assert first_enable >= 7
    assert all(packet != encode_enable(index) for packet in packets[:7] for index in range(1, 8))
    receipt = scope.device.startup_receipt
    assert receipt["kind"] == "operator_teleoperation" and receipt["complete"]
    assert "target_force_N" not in receipt
    assert not receipt["production_arm_authorized"] and not receipt["watchdog_semantics_verified"]
    assert all(row["tolerance_deg"] == 12 for row in receipt["alignment_checks"][0]["joints"])
    assert all(status == 1 for status in scope.serial.statuses.values())
    assert [bytes.fromhex(item["packet_hex"]) for item in receipt["mode_writes"]] == [
        encode_mode_write(index, "mit") for index in range(1, 7)
    ]
    assert scope.serial.mode_readback == {index: 4 if index == 7 else 1 for index in range(1, 8)}


def test_recorded_zero_arm_timeouts_connect_read_only_and_prepare_selected_values_on_start(
    logged_timeout_scope,
):
    scope = logged_timeout_scope
    original_gripper = scope.device.config.profiles[-1]
    assert not movement_packets(scope.serial)
    assert all(packet[23] in (0x33, 0xCC) for packet in scope.serial.writes)
    state = scope.device._full_validation_session.connection_state
    values = {(row.name, row.register_id): row.value for row in state.registers}
    assert all(values[(name, 10)] == 2 and values[(name, 9)] == 0 for name in JOINT_NAMES[:-1])
    assert values[("gripper", 10)] == 4 and values[("gripper", 9)] == 2000

    start(scope)
    receipt = scope.device.startup_receipt
    assert [item["name"] for item in receipt["watchdog_writes"]] == list(JOINT_NAMES[:-1])
    assert all(
        item["register_id"] == 9 and item["requested_value"] == 2000 for item in receipt["watchdog_writes"]
    )
    assert [item["name"] for item in receipt["mode_writes"]] == list(JOINT_NAMES[:-1])
    assert [
        (row["name"], row["register_id"], row["value"])
        for row in receipt["connection_control_state"]["registers"]
    ] == [
        (row["name"], row["register_id"], row["value"])
        for row in receipt["original_control_state"]["registers"]
    ]
    prepared = {
        (row["name"], row["register_id"]): row["value"]
        for row in receipt["prepared_control_state"]["registers"]
    }
    assert all(prepared[(name, 9)] == 2000 for name in JOINT_NAMES)
    assert all(prepared[(name, 10)] == (4 if name == "gripper" else 1) for name in JOINT_NAMES)
    first_enable = next(
        index for index, packet in enumerate(scope.serial.writes) if packet == encode_enable(1)
    )
    writes = [
        (index, packet)
        for index, packet in enumerate(scope.serial.writes)
        if int.from_bytes(packet[13:17], "little") == 0x7FF and packet[23] == 0x55
    ]
    assert len(writes) == 12
    assert all(index < first_enable and packet[21] != 7 and packet[24] in (9, 10) for index, packet in writes)
    assert not any(packet[23] == 0xAA for packet in scope.serial.writes)
    assert scope.device.config.profiles[-1] == original_gripper
    assert scope.device.config.watchdog_ticks == scope.files.operator.watchdog_ticks == 2000
    assert receipt["complete"] and not receipt["watchdog_semantics_verified"]


@pytest.mark.parametrize("gripper_mode", [1, 2, 3])
def test_all_zero_timeouts_and_reset_gripper_mode_prepare_existing_profiles_unchanged(
    tmp_path, monkeypatch, gripper_mode
):
    scope = make_logged_timeout_scope(tmp_path, monkeypatch, connect=False)
    scope.serial.mode_readback[7] = gripper_mode
    scope.serial.watchdog_overrides.clear()
    original_profiles = [asdict(profile) for profile in scope.device.config.profiles]
    try:
        scope.device.connect()
        assert all(packet[23] in (0x33, 0xCC) for packet in scope.serial.writes)
        start(scope)
        receipt = scope.device.startup_receipt
        assert len(receipt["watchdog_writes"]) == len(receipt["mode_writes"]) == 7
        gripper = receipt["mode_writes"][-1]
        assert gripper["name"] == "gripper" and gripper["requested_value"] == 4
        assert bytes.fromhex(gripper["packet_hex"]) == encode_mode_write(7, "force_pos")
        assert [asdict(profile) for profile in scope.device.config.profiles] == original_profiles
        prepared = {
            (row["name"], row["register_id"]): row["value"]
            for row in receipt["prepared_control_state"]["registers"]
        }
        assert all(prepared[(name, 9)] == 2000 for name in JOINT_NAMES)
        assert all(prepared[(name, 10)] == (4 if name == "gripper" else 1) for name in JOINT_NAMES)
        assert receipt["complete"] and all(status == 1 for status in scope.serial.statuses.values())
    finally:
        close_operator_scope(scope)


@pytest.mark.parametrize("echo", [False, True])
def test_stop_after_gripper_force_pos_selection_preserves_generic_parameter_cleanup(
    tmp_path, monkeypatch, echo
):
    scope = make_logged_timeout_scope(tmp_path, monkeypatch, connect=False)
    scope.serial.mode_readback[7] = 2
    scope.serial.mode_write_echo = echo

    def guard():
        if scope.serial.mode_readback[7] == 4:
            raise RuntimeError("stop_after_gripper_mode_write")

    try:
        scope.device.connect()
        with pytest.raises(RuntimeError, match="stop_after_gripper_mode_write"):
            start(scope, guard)
        receipt = scope.device.fault_receipt
        pending = receipt["pending_parameter_write_before_disable"]
        assert pending["name"] == "gripper" and pending["requested_value"] == 4
        assert pending["submission"] == "complete"
        assert receipt["all_joints_disabled_observed"] and receipt["outstanding_query_accounted"]
        assert not any(
            packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8)
        )
    finally:
        close_operator_scope(scope)


def test_timeout_change_after_connect_is_refreshed_before_start(scope):
    scope.serial.watchdog_overrides[3] = 0
    start(scope)
    writes = scope.device.startup_receipt["watchdog_writes"]
    assert len(writes) == 1 and writes[0]["name"] == "elbow_flex"
    assert writes[0]["requested_value"] == scope.device.config.watchdog_ticks == 9876


def test_timeout_readback_failure_never_enables_and_still_disables_all(logged_timeout_scope):
    scope = logged_timeout_scope
    scope.serial.ignored_watchdog_writes.add(3)
    with pytest.raises(RuntimeError, match="TIMEOUT readback failed"):
        start(scope)
    assert not scope.device.startup_receipt["mode_writes"]
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


@pytest.mark.parametrize("echo,feedback", [(False, False), (True, False), (False, True), (True, True)])
def test_stop_after_timeout_write_accounts_optional_replies_and_disables_all(
    logged_timeout_scope, echo, feedback
):
    scope = logged_timeout_scope
    scope.serial.mode_write_echo, scope.serial.mode_write_feedback = echo, feedback

    def guard():
        if scope.serial.watchdog_overrides.get(1) == 2000:
            raise RuntimeError("stop_after_timeout_write")

    with pytest.raises(RuntimeError, match="stop_after_timeout_write"):
        start(scope, guard)
    receipt = scope.device.fault_receipt
    assert receipt["pending_parameter_write_before_disable"]["register_id"] == 9
    assert receipt["pending_parameter_write_before_disable"]["submission"] == "complete"
    assert receipt["all_joints_disabled_observed"] and receipt["outstanding_query_accounted"]
    assert not scope.device.startup_receipt["mode_writes"]


def test_partial_timeout_write_preserves_unknown_evidence_after_all_disable_attempts(logged_timeout_scope):
    scope = logged_timeout_scope
    scope.serial.partial_watchdog_write = True
    with pytest.raises(OSError, match="partial"):
        start(scope)
    receipt = scope.device.fault_receipt
    assert receipt["pending_parameter_write_before_disable"]["register_id"] == 9
    assert receipt["pending_parameter_write_before_disable"]["submission"] == "entered_unknown"
    assert receipt["all_joints_disabled_observed"] and not receipt["outstanding_query_accounted"]
    assert receipt["fresh_disabled_verification_complete"]
    assert receipt["unresolved_transactions"][0]["evidence"]["submission"] == "entered_unknown"
    assert all(item["status"] == "submitted" for item in receipt["per_joint_results"])


@pytest.mark.parametrize("echo", [False, True])
def test_complete_timeout_write_returning_late_keeps_submitted_evidence_for_cleanup(
    logged_timeout_scope, echo
):
    scope = logged_timeout_scope
    scope.serial.mode_write_echo = echo
    original = scope.serial.write

    def late_write(packet):
        result = original(packet)
        if int.from_bytes(packet[13:17], "little") == 0x7FF and packet[24] == 9 and packet[23] == 0x55:
            scope.clock.now += int(scope.device.config.send_budget_s * 1e9)
        return result

    scope.serial.write = late_write
    with pytest.raises(TimeoutError, match="deadline"):
        start(scope)
    receipt = scope.device.fault_receipt
    assert receipt["pending_parameter_write_before_disable"]["register_id"] == 9
    assert receipt["pending_parameter_write_before_disable"]["submission"] == "complete"
    assert scope.device.startup_receipt["watchdog_writes"][0]["status"] == "submitted"
    assert receipt["all_joints_disabled_observed"] and receipt["outstanding_query_accounted"]


def test_final_all_joint_readback_catches_a_changed_gripper_timeout_before_enable(logged_timeout_scope):
    scope = logged_timeout_scope
    original = scope.serial.write

    def write(packet):
        result = original(packet)
        if int.from_bytes(packet[13:17], "little") == 0x7FF and packet[21:25] == bytes((6, 0, 0x55, 10)):
            scope.serial.watchdog_overrides[7] = 0
        return result

    scope.serial.write = write
    with pytest.raises(RuntimeError, match="prepared mode/TIMEOUT readback failed"):
        start(scope)
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


def test_operator_start_keeps_existing_mit_modes_without_register_writes(tmp_path, monkeypatch):
    scope = make_operator_scope(tmp_path, monkeypatch, connect=False)
    scope.serial.mode_readback.update(dict.fromkeys(range(1, 7), 1))
    try:
        scope.device.connect()
        start(scope)
        assert scope.device.startup_receipt["mode_writes"] == []
        assert not any(
            packet[23] == 0x55
            for packet in scope.serial.writes
            if int.from_bytes(packet[13:17], "little") == 0x7FF
        )
    finally:
        close_operator_scope(scope)


def test_operator_mode_selection_accounts_optional_echo_and_verifies_readback(scope):
    scope.serial.mode_write_echo = True
    start(scope)
    assert all(item["readback"]["error"] is None for item in scope.device.startup_receipt["mode_writes"])
    assert scope.device.startup_receipt["complete"]


def test_operator_mit_readback_failure_disables_all_without_enabling(scope):
    scope.serial.ignored_mode_writes.add(3)
    with pytest.raises(RuntimeError, match="MIT mode readback failed"):
        start(scope)
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


@pytest.mark.parametrize("echo,feedback", [(False, False), (True, False), (False, True), (True, True)])
def test_stop_after_complete_mode_write_accounts_optional_echo_separately_from_disable(scope, echo, feedback):
    scope.serial.mode_write_echo = echo
    scope.serial.mode_write_feedback = feedback

    def guard():
        if scope.serial.mode_readback[1] == 1:
            raise RuntimeError("stop_after_first_mode_write")

    with pytest.raises(RuntimeError, match="stop_after_first_mode_write"):
        start(scope, guard)
    receipt = scope.device.fault_receipt
    assert receipt["pending_command_before_disable"] is None
    assert receipt["pending_parameter_write_before_disable"]["submission"] == "complete"
    assert receipt["outstanding_query_accounted"]
    assert receipt["all_joints_disabled_observed"]
    assert receipt["hardware_ack"] == "unknown"
    assert [item["packet_hex"] for item in receipt["per_joint_results"]] == [
        encode_disable(index).hex() for index in range(1, 8)
    ]
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))


@pytest.mark.parametrize("feedback", [False, True])
def test_stop_during_fragmented_parameter_rx_preserves_historical_frame(scope, monkeypatch, feedback):
    scope.serial.mode_write_echo = not feedback
    scope.serial.mode_write_feedback = feedback
    original = scope.device._read_rx_chunk
    interrupted = False

    def stop_after_rx(context, deadline_ns, receive):
        nonlocal interrupted
        if context["source"] == "parameter_write_unattributed_rx" and not interrupted:
            scope.serial.fragment_size = 3
            original(context, deadline_ns, receive)
            interrupted = True
            scope.serial.fragment_size = 999
            raise RuntimeError("stop_during_mode_rx")
        original(context, deadline_ns, receive)

    monkeypatch.setattr(scope.device, "_read_rx_chunk", stop_after_rx)
    with pytest.raises(RuntimeError, match="stop_during_mode_rx"):
        start(scope)
    receipt = scope.device.fault_receipt
    assert receipt["all_joints_disabled_observed"]
    assert receipt["outstanding_query_accounted"]
    assert len(receipt["historical_responses"]) == 1
    assert receipt["historical_responses"][0]["historical_only"]
    assert receipt["historical_responses"][0]["pending_parameter_write"]["submission"] == "complete"


def test_partial_mode_write_keeps_unknown_submission_evidence_and_attempts_all_disables(scope):
    scope.serial.partial_mode_write = True
    with pytest.raises(OSError, match="partial"):
        start(scope)
    receipt = scope.device.fault_receipt
    assert receipt["pending_parameter_write_before_disable"]["submission"] == "entered_unknown"
    assert not receipt["outstanding_query_accounted"]
    assert receipt["all_joints_disabled_observed"] and receipt["fresh_disabled_verification_complete"]
    assert receipt["unresolved_transactions"][0]["evidence"]["submission"] == "entered_unknown"
    assert all(item["status"] == "submitted" for item in receipt["per_joint_results"])
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))


@pytest.mark.parametrize("echo", [False, True])
def test_complete_mode_write_returning_after_deadline_keeps_known_submission_for_cleanup(scope, echo):
    scope.serial.mode_write_echo = echo
    original = scope.serial.write

    def late_mode_write(packet):
        count = original(packet)
        if int.from_bytes(packet[13:17], "little") == 0x7FF and packet[23] == 0x55:
            scope.clock.now += int(scope.device.config.send_budget_s * 1e9)
        return count

    scope.serial.write = late_mode_write
    with pytest.raises(TimeoutError, match="deadline"):
        start(scope)
    receipt = scope.device.fault_receipt
    assert scope.device.startup_receipt["mode_writes"][0]["status"] == "submitted"
    assert receipt["pending_parameter_write_before_disable"]["submission"] == "complete"
    assert receipt["outstanding_query_accounted"]
    assert receipt["all_joints_disabled_observed"]
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))


@pytest.mark.parametrize("position_low_byte", [0x33, 0x55])
def test_parameter_capture_accepts_disabled_feedback_with_opcode_like_position_byte(scope, position_low_byte):
    scope.serial.positions[1] = math.degrees((0x8000 | position_low_byte) * 16 / 65535 - 8)
    scope.serial.mode_write_feedback = True
    start(scope)
    assert scope.device.startup_receipt["complete"]


def test_operator_frame_uses_native_mit_and_preserves_force_pos_gripper(scope):
    snapshot = start(scope)
    before = len(movement_packets(scope.serial))
    receipt = scope.device.send_joint_positions_partial(
        dict.fromkeys(JOINT_NAMES, 80), feedback_snapshot=snapshot
    )
    assert receipt.all_submitted
    packets = movement_packets(scope.serial)[before:]
    for item, packet in zip(receipt.results, packets, strict=True):
        assert item.clipped_deg == pytest.approx(snapshot.get(item.name).position_deg + 10)
        if item.name == "gripper":
            assert int.from_bytes(packet[13:17], "little") == 0x307
            assert item.submitted_deg == pytest.approx(math.degrees(struct.unpack("<f", packet[21:25])[0]))
            assert struct.unpack("<HH", packet[25:29]) == (1000, 200)
        else:
            motor_id = JOINT_NAMES.index(item.name) + 1
            assert int.from_bytes(packet[13:17], "little") == motor_id
            p = int.from_bytes(packet[21:23], "big")
            assert item.submitted_deg == pytest.approx(math.degrees(p * 16 / 65535 - 8))
            assert packet[23] << 4 | packet[24] >> 4 == 2047  # native v_des=0
            assert (packet[27] & 15) << 8 | packet[28] == 2047  # native tau_ff=0
    gains = scope.device.startup_receipt["arm_mit_gains"]
    assert gains["shoulder_pan"]["requested"] == {"kp": 45, "kd": 12}
    assert gains["shoulder_pan"]["wire_effective"]["kd"] == 5
    assert gains["wrist_flex"]["wire_effective"]["kd"] == 1


def test_native_optional_defaults_allow_arm_following_without_new_thresholds(tmp_path, monkeypatch):
    files = make_operator_files(tmp_path)
    files.operator = files.configuration = replace(
        files.operator,
        alignment_tolerance_deg=None,
        preload_tolerance_deg=None,
        max_command_gap_s=None,
    )
    files.control = replace(
        files.control,
        profiles=tuple(replace(profile, max_relative_deg=None) for profile in files.control.profiles[:-1])
        + files.control.profiles[-1:],
    )
    scope = make_operator_scope(tmp_path, monkeypatch, files=files)
    scope.positions.update(dict.fromkeys(JOINT_NAMES, 80.0))
    stop = False

    def guard():
        if stop:
            raise RuntimeError("raw_force_or_stop")

    try:
        snapshot = start(scope, guard)
        checks = scope.device.startup_receipt["alignment_checks"]
        assert checks and all(not item["alignment_gate_enabled"] for item in checks)
        assert all(row["aligned"] is None for item in checks for row in item["joints"])
        receipt = scope.device.send_joint_positions_partial(
            dict.fromkeys(JOINT_NAMES, 80.0), feedback_snapshot=snapshot
        )
        assert receipt.all_submitted
        assert all(item.clipped_deg == 80 for item in receipt.results[:-1])
        assert receipt.results[-1].clipped_deg == pytest.approx(snapshot.get("gripper").position_deg + 10)
        assert scope.device.config.profiles[-1] == files.control.profiles[-1]
        scope.device.take_io_evidence()

        # Fresh observations remain required after a pause, while the optional
        # extra command-gap threshold and arm relative clip are disabled.
        scope.clock.now += 300_000_000
        snapshot = scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 100_000_000)
        receipt = scope.device.send_joint_positions_partial(
            dict.fromkeys(JOINT_NAMES, -80.0), feedback_snapshot=snapshot
        )
        assert receipt.all_submitted
        assert all(item.clipped_deg == -80 for item in receipt.results[:-1])
        assert scope.device.startup_receipt["reference_bounds_deg"]["shoulder_pan"][1] == pytest.approx(
            scope.device.startup_receipt["reference_deg"]["shoulder_pan"] + 100
        )

        session = scope.device._full_validation_session
        session.hold_targets["shoulder_pan"] += 5
        session.check_preloaded_target("shoulder_pan", scope.device._latest_snapshot)

        stop = True
        with pytest.raises(RuntimeError, match="raw_force_or_stop"):
            scope.device.send_joint_positions_partial(
                dict.fromkeys(JOINT_NAMES, 0.0), feedback_snapshot=scope.device._latest_snapshot
            )
        assert scope.device.fault_receipt["all_joints_disabled_observed"]
    finally:
        close_operator_scope(scope)


def test_normal_operator_has_no_30_second_or_micro_total_travel_expiry(scope):
    start(scope)
    initial = scope.clock.now
    for index in range(320):
        scope.clock.now += 100_000_000
        snapshot = scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 100_000_000)
        goal = 20 if index % 8 < 4 else -20
        receipt = scope.device.send_joint_positions_partial(
            dict.fromkeys(JOINT_NAMES, goal), feedback_snapshot=snapshot
        )
        assert receipt.all_submitted
        scope.device.take_io_evidence()  # the real owner drains bounded evidence each control frame
    assert scope.clock.now - initial > 30_000_000_000
    assert all(status == 1 for status in scope.serial.statuses.values())
    assert scope.device.fault_receipt is None


def test_arm_reference_envelope_clips_at_100_without_micro_magnitude_cap(scope):
    start(scope)
    for _ in range(9):
        snapshot = scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 100_000_000)
        receipt = scope.device.send_joint_positions_partial(
            dict.fromkeys(JOINT_NAMES, 150), feedback_snapshot=snapshot
        )
        assert receipt.all_submitted
        scope.device.take_io_evidence()
    reference = scope.device.startup_receipt["reference_deg"]
    # MIT's inward integer quantization can accumulate a fraction of a degree
    # over these frames. Place the simulated plant within one frame of every
    # arm endpoint before testing semantic clipping against that envelope.
    scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 100_000_000)
    for motor_id, name in enumerate(JOINT_NAMES, 1):
        scope.serial.positions[motor_id] = reference[name] + 99
    snapshot = scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 100_000_000)
    receipt = scope.device.send_joint_positions_partial(
        dict.fromkeys(JOINT_NAMES, 150), feedback_snapshot=snapshot
    )
    for result in receipt.results:
        expected = (
            snapshot.get("gripper").position_deg + 10
            if result.name == "gripper"
            else reference[result.name] + 100
        )
        assert result.clipped_deg == pytest.approx(expected)
    assert receipt.all_submitted
    assert scope.device.fault_receipt is None


def test_raw_force_stop_guard_fault_still_attempts_all_seven_disables(scope):
    active = [False]

    def guard():
        if active[0]:
            raise RuntimeError("raw_force_hard_limit")

    snapshot = start(scope, guard)
    active[0] = True
    before = len(scope.serial.calls)
    with pytest.raises(RuntimeError, match="raw_force"):
        scope.device.send_joint_positions_partial(dict.fromkeys(JOINT_NAMES, 0), feedback_snapshot=snapshot)
    operations = scope.serial.calls[before:]
    first_rx = next(index for index, call in enumerate(operations) if call[0] == "read")
    assert sum(call[0] == "write" for call in operations[:first_rx]) == 7
    receipt = scope.device.fault_receipt
    assert receipt["action"] == "disable_all_operator_teleoperation"
    assert receipt["all_joints_disabled_observed"] and receipt["status"] == "submitted"


def test_changed_operator_config_cannot_block_pinned_all_joint_cleanup(scope):
    snapshot = start(scope)
    scope.device._operator_teleoperation = replace(scope.files.operator, preload_tolerance_deg=3)
    scope.device.config = replace(scope.device.config, send_budget_s=1)
    before = len(movement_packets(scope.serial))
    with pytest.raises(RuntimeError, match="replaced|configuration"):
        scope.device.send_joint_positions_partial(dict.fromkeys(JOINT_NAMES, 0), feedback_snapshot=snapshot)
    assert movement_packets(scope.serial)[before:] == [encode_disable(index) for index in range(1, 8)]
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


def test_changed_calibration_blocks_motion_and_still_proves_disabled(scope):
    snapshot = start(scope)
    scope.files.calibration_path.write_text("corrupted", encoding="utf-8")
    before = len(movement_packets(scope.serial))
    with pytest.raises(ValueError, match="changed"):
        scope.device.send_joint_positions_partial(dict.fromkeys(JOINT_NAMES, 0), feedback_snapshot=snapshot)
    assert movement_packets(scope.serial)[before:] == [encode_disable(index) for index in range(1, 8)]
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


def test_operator_disconnect_disables_every_joint_and_closes(scope):
    start(scope)
    scope.device.disconnect()
    assert not scope.device.has_resources
    assert all(status == 0 for status in scope.serial.statuses.values())
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


@pytest.mark.parametrize(
    "field,value",
    [("alignment_tolerance_deg", 0), ("preload_tolerance_deg", -1), ("max_command_gap_s", float("nan"))],
)
def test_missing_operator_parameter_refuses_before_transport_creation(tmp_path, monkeypatch, field, value):
    pytest.importorskip("serial")
    files = make_operator_files(tmp_path)
    opened = []
    from lerobot.robots.rebot_b601_follower import readonly_transport

    monkeypatch.setattr(readonly_transport.serial, "Serial", lambda **kwargs: opened.append(kwargs))
    with pytest.raises(ValueError):
        B601ControlBackend(
            files.operator.port,
            DEFAULT_B601_JOINTS,
            files.control,
            operator_teleoperation=replace(files.operator, **{field: value}),
        )
    assert not opened
