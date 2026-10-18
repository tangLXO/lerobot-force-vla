"""Persist only editable arm choices; calibration, references and activation stay separate."""

import math

from lerobot.robots.rebot_b601_follower.full_arm_control_types import ARM_NAMES

ARM_OPERATOR_FIELDS = (
    "reference_envelope_deg",
    "alignment_tolerance_deg",
    "preload_tolerance_deg",
    "max_command_gap_s",
    "handover_tolerance_deg",
)
ARM_PROFILE_FIELDS = (
    "kp",
    "kd",
    "minimum_deg",
    "maximum_deg",
    "max_relative_deg",
    "mit_velocity_deg_s",
    "mit_torque_Nm",
)


def migrate_arm_settings(settings):
    """Preserve v7 drafts and the historical explicit zero MIT references."""
    if isinstance(settings, dict) and isinstance(settings.get("joints"), dict):
        settings = {
            **settings,
            "joints": {
                name: {"mit_velocity_deg_s": "0", "mit_torque_Nm": "0", **profile}
                if isinstance(profile, dict)
                else profile
                for name, profile in settings["joints"].items()
            },
        }
    return settings


def checked_arm_settings(settings: dict | None) -> dict:
    """Accept empty legacy defaults or a complete detached string-only draft."""
    if settings is None or settings == {}:
        return {}
    if not isinstance(settings, dict) or set(settings) != {"operator", "joints"}:
        raise ValueError("全臂参数字段不匹配")
    operator, joints = settings["operator"], settings["joints"]
    if (
        not isinstance(operator, dict)
        or set(operator) != set(ARM_OPERATOR_FIELDS)
        or any(type(value) is not str for value in operator.values())
        or not isinstance(joints, dict)
        or set(joints) != set(ARM_NAMES)
    ):
        raise ValueError("全臂参数字段或类型不匹配")
    for profile in joints.values():
        if (
            not isinstance(profile, dict)
            or set(profile) != set(ARM_PROFILE_FIELDS)
            or any(type(value) is not str for value in profile.values())
        ):
            raise ValueError("全臂关节参数字段或类型不匹配")
    return {"operator": dict(operator), "joints": {name: dict(profile) for name, profile in joints.items()}}


def apply_arm_settings(document: dict, settings: dict | None) -> None:
    """Apply saved explicit numbers to a disconnected profile; never select new gains."""
    settings = checked_arm_settings(settings)
    if not settings:
        return

    def number(text, label, *, optional=False, signed=False, zero=False):
        if optional and not text.strip():
            return None
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"请填写全臂 {label}") from exc
        if not math.isfinite(value) or (not signed and (value < 0 if zero else value <= 0)):
            raise ValueError(f"全臂 {label} 必须为有效有限值")
        return value

    operator = document["b601"]["operator_teleoperation"]
    for key, text in settings["operator"].items():
        value = number(text, key, optional=key != "reference_envelope_deg")
        if key == "handover_tolerance_deg":
            document["runtime"][key] = value
        else:
            operator[key] = value
    for profile in document["b601"]["control"]["profiles"]:
        if profile["name"] not in settings["joints"]:
            continue
        for key, text in settings["joints"][profile["name"]].items():
            profile[key] = number(
                text,
                f"{profile['name']} {key}",
                optional=key == "max_relative_deg",
                signed=key in ("minimum_deg", "maximum_deg", "mit_velocity_deg_s", "mit_torque_Nm"),
                zero=key in ("kp", "kd"),
            )
