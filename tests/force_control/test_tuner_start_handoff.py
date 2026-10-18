"""Preview-to-owner handoff uses completed cleanup evidence without real I/O."""

import tkinter as tk
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def handoff_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture
def handoff(panel, calibration, settings, monkeypatch, tmp_path, handoff_root):
    def forbidden(*args, **kwargs):
        pytest.fail("Handoff test accessed hardware or started a control process")

    for name in ("ReadonlyB601Transport", "X518Sensor", "run_in_process"):
        monkeypatch.setattr(panel, name, forbidden)
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path / "sessions")
    monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [])
    app = panel.Tuner(handoff_root, calibration_path=calibration.path)
    values, enabled = settings
    for key, value in values.items():
        app.fields[key].set(value)
    for key, value in enabled.items():
        app.toggles[key].set(value)
    for timer in handoff_root.tk.call("after", "info"):
        handoff_root.after_cancel(timer)

    launched = []

    def launch(function):
        launched.append(function)
        app.worker = SimpleNamespace(is_alive=lambda: True)

    monkeypatch.setattr(app, "launch", launch)

    class Live:
        def __init__(self, **kwargs):
            self.stop = panel.threading.Event()
            self.latest = self.startup = self.applied = None
            self.recording = self.preview = None
            self.closed = False

        def receive(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(panel, "ProcessLive", Live)
    app.read()
    assert len(launched) == 1 and app.monitoring, app.status.get()
    app.preview_latest = {
        "monitor_session": app.monitor_session,
        "read_only": True,
        "joint_valid": True,
        "force_valid": True,
        "joint_status_code": 0,
        "phase": "DISABLED",
        "q_measured_deg": 0.0,
        "joint_received_ns": panel.time.perf_counter_ns(),
        "force_timestamp_ns": panel.time.perf_counter_ns(),
    }
    app.confirmed.set(True)

    def complete_preview(*, transport_closed=True, sensor_closed=True, publish_cleanup=True, error=None):
        def monitor(stop, publish, values, loaded_calibration, cleanup_publish):
            if publish_cleanup:
                cleanup_publish(
                    {
                        "read_only": True,
                        "motion_attempted": False,
                        "transport_closed": transport_closed,
                        "sensor_closed": sensor_closed,
                    }
                )
            if error is not None:
                raise RuntimeError(error)

        monkeypatch.setattr(panel, "monitor", monitor)
        try:
            value = launched[0]()
            app.messages.put(("done", value))
        except BaseException as exc:
            app.messages.put(("error", f"{type(exc).__name__}: {exc}"))
            app.messages.put(("done", None))

    try:
        yield SimpleNamespace(app=app, launched=launched, complete_preview=complete_preview)
    finally:
        for timer in handoff_root.tk.call("after", "info"):
            handoff_root.after_cancel(timer)
        for widget in handoff_root.winfo_children():
            widget.destroy()


@pytest.mark.parametrize("preview_finishes_during_review", [False, True])
def test_successful_cleanup_dispatches_pending_start_from_poll_only(
    handoff, panel, monkeypatch, preview_finishes_during_review
):
    app = handoff.app
    if preview_finishes_during_review:
        review = panel.run

        def review_then_preview_exit(*args, **kwargs):
            result = review(*args, **kwargs)
            handoff.complete_preview()
            return result

        monkeypatch.setattr(panel, "run", review_then_preview_exit)
    app.start()
    assert callable(app.next_run) and app.running
    if not preview_finishes_during_review:
        handoff.complete_preview()
    assert len(handoff.launched) == 1
    app.poll()
    assert len(handoff.launched) == 2
    assert app.next_run is None
    assert app.running and app.worker is not None
    assert not app.live.stop.is_set()


@pytest.mark.parametrize(
    "cleanup,error,exit_confirmed",
    [
        ({"transport_closed": False, "sensor_closed": True}, None, False),
        ({"transport_closed": True, "sensor_closed": False}, None, False),
        ({"publish_cleanup": False}, "preview failed before cleanup evidence", False),
        ({"transport_closed": True, "sensor_closed": True}, "preview failed during handoff", True),
    ],
)
def test_failed_or_incomplete_preview_cancels_pending_start_without_dispatch(
    handoff, panel, monkeypatch, cleanup, error, exit_confirmed
):
    app = handoff.app
    review = panel.run

    def review_then_preview_exit(*args, **kwargs):
        result = review(*args, **kwargs)
        handoff.complete_preview(**cleanup, error=error)
        return result

    monkeypatch.setattr(panel, "run", review_then_preview_exit)
    app.start()
    assert callable(app.next_run)
    app.poll()
    assert len(handoff.launched) == 1
    assert app.next_run is None and app.pending_cleanup is None
    assert not app.running and app.worker is None
    assert app.live.closed
    assert app.exit_confirmed is exit_confirmed


def test_stop_during_preview_handoff_cancels_without_starting_owner(handoff):
    app = handoff.app
    app.start()
    assert callable(app.next_run)
    app.stop()
    handoff.complete_preview()
    app.poll()
    assert len(handoff.launched) == 1
    assert app.next_run is None and app.pending_cleanup is None
    assert not app.running and app.worker is None
    assert app.live.stop.is_set() and app.live.closed
    assert app.exit_confirmed


@pytest.mark.parametrize("identified", [False, True])
def test_unknown_or_previous_monitor_cleanup_cannot_start_current_owner(handoff, identified):
    app = handoff.app
    app.start()
    evidence = {
        "read_only": True,
        "motion_attempted": False,
        "transport_closed": True,
        "sensor_closed": True,
    }
    if identified:
        evidence["monitor_session"] = f"{app.monitor_session}-previous"
    app.messages.put(("preview_cleanup", evidence))
    handoff.complete_preview(publish_cleanup=False)
    app.poll()
    assert len(handoff.launched) == 1
    assert app.next_run is None and app.pending_cleanup is None
    assert not app.running and app.worker is None
    assert app.live.closed
    assert not app.exit_confirmed
