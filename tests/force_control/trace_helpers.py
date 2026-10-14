"""Shared sensor and causal control-row fixtures for evidence tests."""

from lerobot.sensors import Sensor, SensorConfig, SensorFeature


class TraceSensor(Sensor):
    def __init__(self):
        super().__init__(SensorConfig(sample_rate_hz=400))
        self.connected = True

    @property
    def features(self):
        return {
            "left.normal_force": SensorFeature("float64", "N"),
            "right.normal_force": SensorFeature("float64", "N"),
        }

    @property
    def native_features(self):
        return {"register_left": SensorFeature("int16", "count")}

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def publish(self, timestamp_ns=1_000_000_000, arrival_timestamp_ns=None, *, is_valid=True):
        return self._publish_sample(
            {"left.normal_force": 1.0, "right.normal_force": 1.2} if is_valid else None,
            timestamp_ns=timestamp_ns,
            arrival_timestamp_ns=timestamp_ns if arrival_timestamp_ns is None else arrival_timestamp_ns,
            native_values={"register_left": 42},
            native_payload=b"\x00\xff",
            is_valid=is_valid,
            status="ok" if is_valid else "timeout",
            error=None if is_valid else "injected timeout",
        )


def control_row(tick=0, **changes):
    anchor = 1_001_000_000 + tick * 20_000_000
    row = {
        "tick_id": tick,
        "segment_id": "segment-0",
        "epoch": 0,
        "decision_anchor_ns": anchor,
        "dt_control_s": 0.02,
        "phase": "FORCE_TRACK",
        "mode": "TEACHER",
        "teacher_active": True,
        "label_valid": tick > 0,
        "safety_override": False,
        "deadline_miss": False,
        "feedback_valid": True,
        "feedback_sequence": tick,
        "feedback_received_ns": anchor - 500,
        "feedback_age_ns": 500,
        "feedback_time_source": "host_rx_test",
        "q_prev_sent_deg": tick * 0.1,
        "q_measured_deg": tick * 0.1,
        "q_candidate_deg": (tick + 1) * 0.1,
        "q_after_safety_deg": (tick + 1) * 0.1,
        "q_sent_deg": (tick + 1) * 0.1,
        "send_status": "submitted",
        "hardware_ack": "unknown",
        "send_start_ns": anchor + 100,
        "send_end_ns": anchor + 200,
        "dt_send_s": 0.02 if tick > 0 else None,
        "prev_successful_tick_id": tick - 1 if tick else None,
        "close_sign": 1,
        "u_label_deg_s": 5.0 if tick else None,
        "force_sequence": tick,
        "force_timestamp_ns": anchor - 1_000_000,
        "force_arrival_timestamp_ns": anchor - 1_000_000,
        "force_left_N": 1.0,
        "force_right_N": 1.2,
        "force_mean_N": 1.1,
        "force_target_effective_N": 1.0,
        "force_target_requested_N": 1.0,
    }
    row.update(changes)
    return row
