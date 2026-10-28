"""Pinned full-arm scope avoids repeated validation but checks bytes before every write."""

import builtins
import os
from dataclasses import replace

import pytest

from lerobot.robots.rebot_b601_follower.control_protocol import encode_enable
from lerobot.robots.rebot_b601_follower.full_arm_control_types import JOINT_NAMES, EvidenceFileIdentity
from tests.force_control.operator_teleoperation_helpers import close_operator_scope, make_operator_scope


@pytest.fixture
def scope(tmp_path, monkeypatch):
    value = make_operator_scope(tmp_path, monkeypatch)
    yield value
    close_operator_scope(value)


def test_pinned_scope_does_not_repeat_dataclass_validation(scope, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unchanged admitted dataclasses must not be revalidated in the write path")

    device = scope.device
    session = device._full_validation_session
    monkeypatch.setattr(type(device.config), "validate", forbidden)
    monkeypatch.setattr(type(session.configuration), "validate", forbidden)
    device._require_scope()


@pytest.mark.parametrize("change", ["control", "capability", "profile_map", "spec_map"])
def test_pinned_scope_rejects_every_configuration_drift(scope, change):
    device = scope.device
    if change == "control":
        device.config = replace(device.config, send_budget_s=device.config.send_budget_s * 2)
    elif change == "capability":
        device._operator_teleoperation = replace(device._operator_teleoperation)
    elif change == "profile_map":
        name = JOINT_NAMES[0]
        device._profiles[name] = replace(device._profiles[name], kp=1)
    else:
        device._specs = {name: spec for name, spec in device._specs.items() if name != JOINT_NAMES[0]}
    with pytest.raises((RuntimeError, ValueError), match="changed|replaced"):
        device._require_scope()


def test_calibration_identity_is_checked_before_and_after_each_content_read(scope, monkeypatch):
    session = scope.device._full_validation_session
    original = EvidenceFileIdentity.require_unchanged
    paths = []

    def observe(pin):
        paths.append(pin.path)
        original(pin)

    monkeypatch.setattr(EvidenceFileIdentity, "require_unchanged", observe)
    session.require_binding()
    assert paths.count(session.calibration_identity.path) == 2


def test_same_stat_calibration_edit_between_axis_writes_blocks_the_next_motion(scope, monkeypatch):
    device, serial = scope.device, scope.serial
    snapshot = device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)
    session = device._full_validation_session
    pin = session.calibration_identity
    original_write = serial.write
    changed = False
    attempted_motion = []

    def edit_after_first_target(packet):
        nonlocal changed
        identifier = int.from_bytes(packet[13:17], "little")
        # Only active target packets count; all-seven fault disables stay available.
        if identifier != 0x7FF and packet[21:29] not in (b"\xff" * 7 + b"\xfd", b"\xff" * 7 + b"\xfc"):
            attempted_motion.append(identifier)
        result = original_write(packet)
        if not changed and attempted_motion:
            changed = True
            stat = os.stat(pin.path)
            with open(pin.path, "rb") as handle:
                original = handle.read()
            updated = original.replace(b"200.0", b"199.0", 1)
            assert updated != original and len(updated) == len(original)
            with open(pin.path, "wb") as handle:
                handle.write(updated)
            os.utime(pin.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            pin.require_unchanged()  # A metadata-only shortcut would miss this edit.
        return result

    monkeypatch.setattr(serial, "write", edit_after_first_target)
    targets = {name: snapshot.get(name).position_deg for name in JOINT_NAMES}
    receipt = device.send_joint_positions_partial(targets, feedback_snapshot=snapshot)
    assert not receipt.all_submitted
    assert attempted_motion == [1]
    assert "contents changed" in receipt.results[1].error
    assert device.fault_receipt["all_joints_disabled_observed"]
    assert all(packet != encode_enable(index) for packet in serial.writes[-7:] for index in range(1, 8))


def test_normal_transaction_never_reopens_calibration(scope, monkeypatch):
    """An expensive file-open operation must stay outside the normal send budget."""
    device = scope.device
    snapshot = device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)
    original = builtins.open
    reopened = []

    def forbid_reopen(path, *args, **kwargs):
        if str(path) == device._full_validation_session.calibration_identity.path:
            reopened.append(str(path))
            raise OSError("injected calibration open delay/failure in normal send")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", forbid_reopen)
    receipt = device.send_joint_positions_partial(
        {name: snapshot.get(name).position_deg for name in JOINT_NAMES},
        feedback_snapshot=snapshot,
    )
    assert receipt.all_submitted and not reopened


@pytest.mark.parametrize("change", ["replace", "remove", "truncate"])
def test_calibration_path_mutation_between_axes_preserves_all_motor_cleanup(scope, monkeypatch, change):
    from pathlib import Path

    device, serial = scope.device, scope.serial
    snapshot = device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)
    path = Path(device._full_validation_session.calibration_identity.path)
    original_write, motion, mutations = serial.write, [], []

    def mutate(packet):
        identifier = int.from_bytes(packet[13:17], "little")
        active = identifier != 0x7FF and packet[21:29] not in (b"\xff" * 7 + b"\xfd", b"\xff" * 7 + b"\xfc")
        result = original_write(packet)
        if active:
            motion.append(identifier)
            if len(motion) == 1:
                if change == "replace":
                    replacement = path.with_suffix(".replacement")
                    replacement.write_bytes(path.read_bytes())
                    stamp = path.stat()
                    os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    path.rename(path.with_suffix(".old"))
                    replacement.rename(path)
                elif change == "remove":
                    path.unlink()
                else:
                    path.write_bytes(b"")
                mutations.append(change)
        return result

    monkeypatch.setattr(serial, "write", mutate)
    receipt = device.send_joint_positions_partial(
        {name: snapshot.get(name).position_deg for name in JOINT_NAMES}, feedback_snapshot=snapshot
    )
    assert not receipt.all_submitted and motion == [1]
    assert mutations == [change]  # A denied OS mutation is not a guard test.
    assert device.fault_receipt["all_joints_disabled_observed"]
    assert len(device.fault_receipt["per_joint_results"]) == 7


def test_calibration_handle_closes_on_serial_open_failure(tmp_path, monkeypatch):
    value = make_operator_scope(tmp_path, monkeypatch, connect=False)
    monkeypatch.setattr(value.serial, "open", lambda: (_ for _ in ()).throw(OSError("serial open failed")))
    with pytest.raises(OSError, match="serial open failed"):
        value.device.connect()
    assert not value.device.has_resources
    assert not value.device._full_validation_session.has_binding_resources


def test_binding_close_failure_remains_owned_and_can_be_retried_after_serial_closed(scope, monkeypatch):
    reader = scope.device._full_validation_session._calibration_reader
    original = reader._stream
    calls = []

    class CloseFailure:
        def close(self):
            calls.append(True)
            if len(calls) == 1:
                raise OSError("binding close failed")
            original.close()

    monkeypatch.setattr(reader, "_stream", CloseFailure())
    with pytest.raises(OSError, match="binding close failed"):
        scope.device.disconnect()
    assert scope.device._serial is None and scope.device.has_resources
    scope.device.disconnect()
    assert not scope.device.has_resources and original.closed and len(calls) == 2
