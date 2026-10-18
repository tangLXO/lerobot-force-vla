"""The active I/O guard must retain a stop's real stage and raw submission status."""

from tests.force_control.test_operator_timeout_latency import latency_scope


def test_stop_during_active_reply_is_user_stop_not_startup_or_generic_send_failure(tmp_path, monkeypatch):
    with latency_scope(tmp_path, monkeypatch, fast_timeout=True) as rig:
        rig.tick("arm")
        rig.tick("start")
        original = rig.serial.read
        stopped = False

        def stop_while_receiving(size):
            nonlocal stopped
            if not stopped:
                stopped = True
                rig.runtime.request_stop()
            return original(size)

        rig.serial.read = stop_while_receiving
        row = rig.tick()
        assert row["phase"] == "FAULT" and row["failure_reason"] == "user_stop"
        assert row["send_status"] == "unknown" and not row["label_valid"]
        assert rig.runtime.controller.fault_reason == "user_stop"
        assert rig.scope.device.fault_receipt["reason"] == "user_stop"
        assert rig.scope.device.fault_receipt["all_joints_disabled_observed"]


def test_stop_during_actual_startup_keeps_startup_attribution(tmp_path, monkeypatch):
    with latency_scope(tmp_path, monkeypatch, fast_timeout=True) as rig:
        rig.tick("arm")
        original = rig.serial.write

        def stop_after_preload(packet):
            result = original(packet)
            if int.from_bytes(packet[13:17], "little") == 1:
                rig.runtime.request_stop()
            return result

        rig.serial.write = stop_after_preload
        row = rig.tick("start")
        assert row["phase"] == "FAULT"
        assert "user_stop_during_startup" in row["failure_reason"]
        assert rig.scope.device.fault_receipt["all_joints_disabled_observed"]
