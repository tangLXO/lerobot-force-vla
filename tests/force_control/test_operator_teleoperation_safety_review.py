"""Independent normal-operator admission and failure tests; fake serial only."""

from dataclasses import asdict, replace

import pytest

from lerobot.robots.rebot_b601_follower import full_arm_validation as diagnostic
from lerobot.robots.rebot_b601_follower.control_protocol import encode_disable, encode_enable
from lerobot.robots.rebot_b601_follower.operator_teleoperation import (
    OperatorLeaderSample,
    OperatorTeleoperationConfig,
)
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from tests.force_control.operator_teleoperation_helpers import (
    JOINT_NAMES,
    close_operator_scope,
    make_operator_files,
    make_operator_scope,
)


@pytest.fixture
def scope(tmp_path, monkeypatch):
    value = make_operator_scope(tmp_path, monkeypatch)
    yield value
    close_operator_scope(value)


def start(scope, *, guard=lambda: None, leader=None):
    return scope.device.start_operator_teleoperation(guard=guard, leader_sample=leader or scope.leader)


def test_admission_uses_only_current_calibration_and_no_diagnostic_history(tmp_path, monkeypatch):
    files = make_operator_files(tmp_path)

    def historical_gate(*_args, **_kwargs):
        pytest.fail("Normal operator admission must not read or require micro-trial history")

    monkeypatch.setattr(diagnostic, "_micro_trial", historical_gate)
    monkeypatch.setattr(diagnostic.FullArmValidationConfig, "validate_binding", historical_gate)
    admission = files.operator.validate_binding(DEFAULT_B601_JOINTS, files.control, port=files.operator.port)
    assert tuple(item.path for item in admission.file_identities) == (str(files.calibration_path.resolve()),)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["native_calibration.json"]
    assert not hasattr(files.operator, "micro_trials")
    assert not hasattr(files.operator, "max_duration_s")
    assert not hasattr(files.operator, "target_force_N")
    assert all(
        profile.mode == "mit" and profile.velocity_deg_s is None for profile in files.control.profiles[:-1]
    )
    assert all(profile.max_relative_deg == 10 for profile in files.control.profiles)


@pytest.mark.parametrize(
    "field,value",
    [
        ("leader_max_age_s", None),
        ("watchdog_ticks", None),
        ("supported_arm", False),
        ("reference_envelope_deg", 0),
    ],
)
def test_unknown_required_protection_is_not_filled_from_diagnostic_defaults(tmp_path, field, value):
    files = make_operator_files(tmp_path)
    incomplete = replace(files.operator, **{field: value})
    with pytest.raises((ValueError, TypeError)):
        incomplete.validate_binding(DEFAULT_B601_JOINTS, files.control, port=incomplete.port)


def test_native_operator_defaults_leave_additional_constraints_unselected(tmp_path):
    files = make_operator_files(tmp_path)
    fields = asdict(files.operator)
    for name in ("alignment_tolerance_deg", "preload_tolerance_deg", "max_command_gap_s"):
        fields.pop(name)
    configuration = OperatorTeleoperationConfig(**fields)
    control = replace(
        files.control,
        profiles=tuple(replace(profile, max_relative_deg=None) for profile in files.control.profiles[:-1])
        + files.control.profiles[-1:],
    )
    configuration.validate_binding(DEFAULT_B601_JOINTS, control, port=configuration.port)
    assert configuration.alignment_tolerance_deg is None
    assert configuration.preload_tolerance_deg is None
    assert configuration.max_command_gap_s is None
    assert configuration.reference_envelope_deg == 100
    assert control.profiles[-1] == files.control.profiles[-1]


@pytest.mark.parametrize("field", ["alignment_tolerance_deg", "preload_tolerance_deg", "max_command_gap_s"])
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_selected_optional_constraints_still_require_valid_positive_values(tmp_path, field, value):
    files = make_operator_files(tmp_path)
    with pytest.raises(ValueError, match="positive when selected"):
        replace(files.operator, **{field: value}).validate()


def test_current_calibration_still_bounds_even_explicit_operator_profiles(tmp_path):
    files = make_operator_files(tmp_path)
    outside = replace(files.control.profiles[0], minimum_deg=-201)
    control = replace(files.control, profiles=(outside, *files.control.profiles[1:]))
    with pytest.raises(ValueError, match="calibration"):
        files.operator.validate_binding(DEFAULT_B601_JOINTS, control, port=files.operator.port)


def test_operator_leader_freshness_uses_explicit_value_not_old_150ms(tmp_path):
    files = make_operator_files(tmp_path)
    sample = OperatorLeaderSample(
        tuple(dict.fromkeys(JOINT_NAMES, 0.0).items()),
        1_000_000_000,
        1_001_000_000,
        1,
        files.operator.leader_config_sha256,
    )
    sample.validate(files.operator, now_ns=1_200_000_000)
    with pytest.raises(ValueError, match="stale|fresh"):
        sample.validate(files.operator, now_ns=1_260_000_000)


@pytest.mark.parametrize("fault", ["future", "stale", "wrong_source", "missing_joint"])
def test_invalid_leader_before_start_can_never_enable_any_joint(scope, fault):
    original = scope.leader

    def invalid():
        sample = original()
        if fault == "future":
            return replace(sample, captured_ns=scope.clock.now + 1, response_completed_ns=scope.clock.now + 1)
        if fault == "stale":
            return replace(sample, captured_ns=scope.clock.now - 300_000_000)
        if fault == "wrong_source":
            return replace(sample, source_config_sha256="b" * 64)
        return replace(sample, positions_deg=sample.positions_deg[:-1])

    with pytest.raises((ValueError, RuntimeError)):
        start(scope, leader=invalid)
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))
    assert all(status == 0 for status in scope.serial.statuses.values())


@pytest.mark.parametrize("wrong", ["unknown_gripper_mode", "unknown_arm_mode"])
def test_unsupported_native_modes_refuse_during_query_only_connect(tmp_path, monkeypatch, wrong):
    value = make_operator_scope(tmp_path, monkeypatch, connect=False)
    if wrong == "unknown_gripper_mode":
        value.serial.mode_readback[7] = 0
    else:
        value.serial.mode_readback[3] = 5
    try:
        with pytest.raises(RuntimeError, match="mode|TIMEOUT"):
            value.device.connect()
        assert not any(
            packet == encode_enable(index) for packet in value.serial.writes for index in range(1, 8)
        )
        # The double asserts if any register write is attempted.
        assert all(int.from_bytes(packet[13:17], "little") == 0x7FF for packet in value.serial.writes)
    finally:
        close_operator_scope(value)


@pytest.mark.parametrize("fault", ["config", "diagnostics", "capability", "calibration", "force_or_stop"])
def test_faults_cannot_prevent_the_pinned_all_joint_disable_attempts(scope, fault):
    guard_fault = False

    def guard():
        if guard_fault:
            raise RuntimeError("Injected raw-force or operator Stop protection")

    snapshot = start(scope, guard=guard)
    if fault == "config":
        scope.device.config = replace(scope.device.config, profiles=())
    elif fault == "diagnostics":
        scope.device.config = replace(scope.device.config, io_event_capacity=None)
    elif fault == "capability":
        scope.device._operator_teleoperation = replace(scope.files.operator, reference_envelope_deg=101)
    elif fault == "calibration":
        scope.files.calibration_path.write_text("invalid calibration", encoding="utf-8")
    else:
        guard_fault = True
    before = len(scope.serial.calls)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        scope.device.send_joint_positions_partial(dict.fromkeys(JOINT_NAMES, 1.0), feedback_snapshot=snapshot)
    operations = scope.serial.calls[before:]
    assert [item[0] for item in operations[:7]] == ["write"] * 7
    writes = [item for item in scope.serial.writes if item in [encode_disable(i) for i in range(1, 8)]]
    assert writes == [encode_disable(index) for index in range(1, 8)]
    assert all(status == 0 for status in scope.serial.statuses.values())
    assert scope.device.fault_receipt["hardware_ack"] == "unknown"
