"""Shared offline force-control fixtures; no test-module imports or hardware I/O."""

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.runtime import RuntimeConfig
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.types import ControlInputs, ForceObservation
from lerobot.scripts import lerobot_force_teacher as runner


def inputs(force=0.5, *, now=1_000_000_000, dt=0.02, q=0.0, right=None, measured=None):
    """Construct one fresh synthetic raw-force decision anchor."""
    return ControlInputs(
        now,
        dt,
        q if measured is None else measured,
        q,
        ForceObservation(force, force if right is None else right, now, now, now),
        now,
        now,
    )


def tracking_teacher(close_sign=1, **overrides):
    """Start the unique I-P law without a physical device."""
    teacher = ForceTeacher(TeacherConfig.fake(**overrides), close_sign)
    teacher.arm()
    teacher.start(0, 0.0)
    return teacher


def submit(teacher, proposal, q=None):
    """Commit a synthetic final wire target for pure arithmetic tests."""
    teacher.commit(proposal, q_sent_deg=proposal.q_candidate_deg if q is None else q, software_success=True)


def run_config(tmp_path, *, armed=False):
    """Use relaxed timing budgets so CI validates logic instead of a scheduler benchmark."""
    return runner.ForceTeacherRunConfig(
        explicit_arm=armed,
        close_sign=1,
        teacher=TeacherConfig.fake(
            release_travel_deg=0.03,
            max_release_time_s=0.8,
        ),
        safety=SafetyConfig.fake(
            max_close_speed_deg_s=25.0,
            acceleration_limit_deg_s2=1000.0,
            force_max_age_ms=500.0,
            joint_feedback_max_age_ms=500.0,
            decision_max_age_ms=200.0,
            max_dt_s=0.5,
        ),
        runtime=RuntimeConfig(feedback_budget_ms=100.0, tick_deadline_ms=200.0, command_ttl_ms=1000.0),
        trace_root=tmp_path / "run",
        duration_s=0.3 if armed else 0.06,
        fake_response_time_s=0.001,
    )
