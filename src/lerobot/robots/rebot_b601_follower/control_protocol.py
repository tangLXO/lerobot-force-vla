"""Offline B601 command encoding, separate from the query-only diagnostic API.

Source: motorbridge v0.3.9 commit 23c22998db072bc0f2ee841874220ac36844a550,
motor_vendors/damiao/src/{protocol,motor}.rs and motor_core/src/dm_serial.rs.
Encoding bytes does not establish motor mode, calibration, ACK or safe stopping.
"""

import math
import struct
from dataclasses import dataclass
from typing import Literal


def _number(value: float, name: str) -> float:
    """Reject booleans, nonnumeric inputs and nonfinite command parameters."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _f32(value: float) -> float:
    """Use the SDK's float32 wire precision without silently overflowing."""
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ValueError("Command cannot be represented as float32") from exc
    if not math.isfinite(result):
        raise ValueError("Command cannot be represented as finite float32")
    return result


def _velocity_code(value: int) -> int:
    """Accept only a nonzero FORCE_POS integer code, without float quantization."""
    if type(value) is not int or not 1 <= value <= 10000:
        raise ValueError("velocity_wire_code must be an integer in 1..10000")
    return value


def _next_f32(value: float, direction: int) -> float:
    """Move one float32 value toward increasing (+1) or decreasing (-1) radians."""
    if value == 0:
        bits = 1 | (0x80000000 if direction < 0 else 0)
    else:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits += -direction if bits & 0x80000000 else direction
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def position_wire_bounds(lower_deg: float, upper_deg: float) -> tuple[float, float]:
    """Round mechanical endpoints inward to representable float32 radians."""
    lower = _number(lower_deg, "lower_deg")
    upper = _number(upper_deg, "upper_deg")
    if lower >= upper:
        raise ValueError("Position minimum must be below maximum")
    bounds = []
    for bound, direction in ((lower, 1), (upper, -1)):
        position = _f32(math.radians(bound))
        if direction * (math.degrees(position) - bound) < 0:
            position = _next_f32(position, direction)
        bounds.append(math.degrees(position))
    if bounds[0] >= bounds[1]:
        raise ValueError("Position range has no distinct float32 endpoints")
    return bounds[0], bounds[1]


@dataclass(frozen=True)
class B601JointCommandProfile:
    """Explicit per-joint bounds and drive parameters; no hardware defaults."""

    name: str
    mode: Literal["mit", "pos_vel", "force_pos"]
    minimum_deg: float
    maximum_deg: float
    max_relative_deg: float | None = None
    velocity_deg_s: float | None = None
    torque_ratio: float | None = None
    kp: float | None = None
    kd: float | None = None
    velocity_wire_code: int | None = None
    mit_velocity_deg_s: float = 0.0
    mit_torque_Nm: float = 0.0  # noqa: N815 - preserve the explicit SI torque unit.

    @property
    def velocity_limit_deg_s(self) -> float | None:
        """Express the selected velocity input in degrees per second for display."""
        if self.velocity_wire_code is None:
            return self.velocity_deg_s
        if self.velocity_deg_s is not None:
            raise ValueError("velocity_deg_s and velocity_wire_code are mutually exclusive")
        return math.degrees(_velocity_code(self.velocity_wire_code) / 100)

    def validate(self) -> None:
        """Validate explicit settings, retaining the vendor's MIT gain saturation."""
        if not isinstance(self.name, str) or not self.name or "." in self.name:
            raise ValueError("A bare joint name is required")
        if self.mode not in ("mit", "pos_vel", "force_pos"):
            raise ValueError("Unsupported B601 command mode")
        if _number(self.minimum_deg, "minimum_deg") >= _number(self.maximum_deg, "maximum_deg"):
            raise ValueError("Joint minimum must be below maximum")
        if self.max_relative_deg is None and self.mode != "mit":
            raise ValueError("Position modes require an explicit positive max_relative_deg")
        if self.max_relative_deg is not None and _number(self.max_relative_deg, "max_relative_deg") <= 0:
            raise ValueError("max_relative_deg must be positive")
        _number(self.mit_velocity_deg_s, "mit_velocity_deg_s")
        _number(self.mit_torque_Nm, "mit_torque_Nm")
        if self.mode == "mit":
            if _number(self.kp, "kp") < 0 or _number(self.kd, "kd") < 0:
                raise ValueError("MIT gains must be nonnegative")
            if (
                self.velocity_deg_s is not None
                or self.velocity_wire_code is not None
                or self.torque_ratio is not None
            ):
                raise ValueError("MIT profile does not accept position-mode parameters")
        else:
            if self.mit_velocity_deg_s != 0 or self.mit_torque_Nm != 0:
                raise ValueError("Position-mode profile does not accept MIT velocity/torque references")
            if self.kp is not None or self.kd is not None:
                raise ValueError("Position-mode profile does not accept MIT gains")
            if self.mode == "force_pos":
                if self.name != "gripper":
                    raise ValueError("FORCE_POS is restricted to gripper")
                force_pos_wire_codes(
                    self.velocity_deg_s, self.torque_ratio, velocity_wire_code=self.velocity_wire_code
                )
            else:
                if self.velocity_wire_code is not None:
                    raise ValueError("velocity_wire_code is restricted to FORCE_POS")
                velocity = _number(self.velocity_deg_s, "velocity_deg_s")
                if velocity <= 0:
                    raise ValueError("velocity_deg_s must be positive")
                if self.torque_ratio is not None:
                    raise ValueError("POS_VEL does not accept a torque ratio")


@dataclass(frozen=True)
class B601EncodedPosition:
    """Representable target and packet, before any submission has occurred."""

    packet: bytes
    clipped_deg: float
    wire_position_deg: float


def _bridge_packet(arbitration_id: int, payload: bytes) -> bytes:
    """Encode one standard DLC-8 frame without retransmit or parameter storage."""
    raw = bytearray(30)
    raw[:4] = b"\x55\xaa\x1e\x03"
    raw[4:8] = struct.pack("<I", 1)
    raw[8:12] = struct.pack("<I", 10)
    raw[13:17] = struct.pack("<I", arbitration_id)
    raw[18] = 8
    raw[21:29] = payload
    return bytes(raw)


def _motor_id(motor_id: int) -> None:
    """Restrict command IDs to the same decodable set as feedback IDs."""
    if type(motor_id) is not int or not 1 <= motor_id <= 15:
        raise ValueError("motor_id must be an integer in 1..15")


def encode_mode_query(motor_id: int) -> bytes:
    """Read CTRL_MODE (10); never write or silently repair device configuration."""
    _motor_id(motor_id)
    return _bridge_packet(0x7FF, bytes((motor_id, 0, 0x33, 10, 0, 0, 0, 0)))


def encode_timeout_query(motor_id: int) -> bytes:
    """Read TIMEOUT (9), without assuming firmware-specific timeout behavior."""
    _motor_id(motor_id)
    return _bridge_packet(0x7FF, bytes((motor_id, 0, 0x33, 9, 0, 0, 0, 0)))


def encode_timeout_write(motor_id: int, ticks: int) -> bytes:
    """Encode a nonzero TIMEOUT register value, with no store or enable command.

    The caller must verify the firmware's tick unit and physical disposition.
    A positive readback alone does not prove a working communication watchdog.
    """
    _motor_id(motor_id)
    if type(ticks) is not int or not 1 <= ticks <= 0xFFFFFFFF:
        raise ValueError("Timeout ticks must be a nonzero uint32")
    return _bridge_packet(0x7FF, bytes((motor_id, 0, 0x55, 9)) + struct.pack("<I", ticks))


def encode_mode_write(motor_id: int, mode: Literal["mit", "pos_vel", "force_pos"]) -> bytes:
    """Encode only CTRL_MODE, without a parameter-store, enable or zero command.

    This is an offline primitive. A caller must separately establish disabled
    state, authorize the selected motor, and verify mode readback before enable.
    It does not make the gated production backend ready for hardware control.
    """
    _motor_id(motor_id)
    modes = {"mit": 1, "pos_vel": 2, "force_pos": 4}
    if not isinstance(mode, str) or mode not in modes:
        raise ValueError("Only MIT, POS_VEL and FORCE_POS modes can be encoded")
    return _bridge_packet(0x7FF, bytes((motor_id, 0, 0x55, 10)) + struct.pack("<I", modes[mode]))


def encode_enable(motor_id: int) -> bytes:
    """Encode one addressed enable, without an implicit target or any other joint.

    Byte encoding alone does not validate activation or target initialization.
    No hardware entry point calls this before lifecycle acceptance.
    """
    _motor_id(motor_id)
    return _bridge_packet(motor_id, b"\xff" * 7 + b"\xfc")


def encode_disable(motor_id: int) -> bytes:
    """Encode a disable candidate, not proof of a safe physical stop."""
    _motor_id(motor_id)
    return _bridge_packet(motor_id, b"\xff" * 7 + b"\xfd")


def _uint(value: float, low: float, high: float, bits: int) -> int:
    """Match the SDK's clamp, float32 arithmetic and truncation for MIT fields."""
    value = max(low, min(high, value))
    return int(_f32(_f32(_f32(value - low) * ((1 << bits) - 1)) / _f32(high - low)))


def mit_gain_wire_codes(kp: float, kd: float) -> tuple[int, int]:
    """Encode requested gains exactly as motorbridge 0.3.9, including its clamp.

    The native B601 profile requests kd=12 for its first three joints. The
    vendor protocol saturates those requests to its kd=5 wire maximum.
    """
    kp, kd = _number(kp, "kp"), _number(kd, "kd")
    if kp < 0 or kd < 0:
        raise ValueError("MIT gains must be nonnegative")
    return _uint(_f32(min(kp, 500)), 0, 500, 12), _uint(_f32(min(kd, 5)), 0, 5, 12)


def force_pos_wire_codes(
    velocity_deg_s: float | None = None,
    torque_ratio: float | None = None,
    *,
    velocity_wire_code: int | None = None,
) -> tuple[int, int]:
    """Preserve SDK degrees/s quantization or use one explicit integer speed code."""
    torque = _number(torque_ratio, "torque_ratio")
    if not 0 < torque <= 1:
        raise ValueError("FORCE_POS velocity/torque exceeds the protocol range")
    if velocity_wire_code is not None:
        if velocity_deg_s is not None:
            raise ValueError("velocity_deg_s and velocity_wire_code are mutually exclusive")
        v = _velocity_code(velocity_wire_code)
    else:
        velocity = _number(velocity_deg_s, "velocity_deg_s")
        if not 0 < math.radians(velocity) <= 100:
            raise ValueError("FORCE_POS velocity/torque exceeds the protocol range")
        v = int(_f32(_f32(math.radians(velocity)) * 100))
    t = int(_f32(_f32(torque) * 10000))
    if not 1 <= v <= 10000 or not 1 <= t <= 10000:
        raise ValueError("FORCE_POS velocity/torque is outside nonzero wire resolution")
    return v, t


def encode_position(
    motor_id: int,
    candidate_deg: float,
    measured_deg: float,
    profile: B601JointCommandProfile,
    mapping: tuple[float, float, float],
) -> B601EncodedPosition:
    """Clip against explicit and measured-relative bounds, then report wire precision."""
    _motor_id(motor_id)
    profile.validate()
    candidate = _number(candidate_deg, "candidate_deg")
    measured = _number(measured_deg, "measured_deg")
    if len(mapping) != 3 or any(_number(value, "mapping") <= 0 for value in mapping):
        raise ValueError("Three positive session mapping registers are required")
    pmax, vmax, tmax = mapping
    low = max(profile.minimum_deg, -math.degrees(pmax))
    high = min(profile.maximum_deg, math.degrees(pmax))
    if profile.max_relative_deg is not None:
        low = max(low, measured - profile.max_relative_deg)
        high = min(high, measured + profile.max_relative_deg)
    if low > high:
        raise ValueError("Joint, measured-relative and device ranges have no intersection")
    clipped = max(low, min(high, candidate))
    pos = _f32(math.radians(clipped))
    if profile.mode == "mit":
        p = _uint(pos, -pmax, pmax, 16)
        wire_position = math.degrees(p * (2 * pmax) / 65535 - pmax)
        # The SDK truncates MIT position to a 16-bit code. At an interval
        # endpoint that code can fall just outside the existing bounds. Move
        # inward on the same grid, retaining the semantic clipped target.
        while wire_position < low and p < 65535:
            p += 1
            wire_position = math.degrees(p * (2 * pmax) / 65535 - pmax)
        while wire_position > high and p > 0:
            p -= 1
            wire_position = math.degrees(p * (2 * pmax) / 65535 - pmax)
        velocity = math.radians(profile.mit_velocity_deg_s)
        if abs(velocity) > vmax or abs(profile.mit_torque_Nm) > tmax:
            raise ValueError("MIT velocity/torque reference exceeds verified device mapping")
        v = _uint(velocity, -vmax, vmax, 12)
        t = _uint(profile.mit_torque_Nm, -tmax, tmax, 12)
        kp, kd = mit_gain_wire_codes(profile.kp, profile.kd)
        payload = bytes(
            (
                p >> 8,
                p & 255,
                v >> 4,
                (v & 15) << 4 | kp >> 8,
                kp & 255,
                kd >> 4,
                (kd & 15) << 4 | t >> 8,
                t & 255,
            )
        )
        arbitration_id = motor_id
        if payload[:7] == b"\xff" * 7:
            raise ValueError("MIT command collides with a reserved motor command")
    else:
        # Nearest float32 rounding can put a valid endpoint just outside the
        # calibrated/relative/device intersection (e.g. -270 degrees). Move
        # one representable step inward; never widen the allowed interval.
        wire_position = math.degrees(pos)
        if wire_position < low:
            pos = _next_f32(pos, 1)
        elif wire_position > high:
            pos = _next_f32(pos, -1)
        if profile.mode == "force_pos":
            v, torque = force_pos_wire_codes(
                profile.velocity_deg_s, profile.torque_ratio, velocity_wire_code=profile.velocity_wire_code
            )
            if v / 100 > vmax:
                raise ValueError("Velocity exceeds the actual device mapping")
            payload = struct.pack("<fHH", pos, v, torque)
            arbitration_id = 0x300 + motor_id
        else:
            velocity = _f32(math.radians(profile.velocity_deg_s))
            if velocity > vmax:
                raise ValueError("Velocity exceeds the actual device mapping")
            if velocity <= 0:
                raise ValueError("POS_VEL velocity is below wire resolution")
            payload = struct.pack("<ff", pos, velocity)
            arbitration_id = 0x100 + motor_id
        wire_position = math.degrees(pos)
    if not low <= wire_position <= high:
        raise ValueError("Wire position quantization leaves the allowed interval")
    return B601EncodedPosition(_bridge_packet(arbitration_id, payload), clipped, wire_position)
