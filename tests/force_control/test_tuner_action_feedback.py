"""Start refusals stay visible beside the action without connecting any device."""

import json

import pytest

from tests.force_control import test_default_mit_entry

default_tuner = test_default_mit_entry.default_tuner
tk_root = test_default_mit_entry.tk_root


@pytest.mark.parametrize("scaling", [1.25, 1.5])
def test_teleoperation_refusal_is_visible_at_minimum_window_size(default_tuner, panel, monkeypatch, scaling):
    app, _, _, _ = default_tuner
    root = app.root
    original_scaling = root.tk.call("tk", "scaling")
    failure = "六臂启动未完成：shoulder_pan TIMEOUT=0，当前配置要求 TIMEOUT=2000。"
    attempts = []
    callback_errors = []

    def refuse_offline_review(*args, **kwargs):
        # Tuner.run() is invoked for review with execute=False. Reject here
        # before ProcessLive or a hardware handoff can ever be constructed.
        assert kwargs.get("execute", False) is False
        attempts.append((args, kwargs))
        raise RuntimeError(failure)

    monkeypatch.setattr(panel, "run", refuse_offline_review)
    monkeypatch.setattr(
        root,
        "report_callback_exception",
        lambda kind, error, traceback: callback_errors.append((kind, str(error))),
    )
    try:
        root.tk.call("tk", "scaling", scaling)
        # Mapping outside the desktop resolves Windows layout without showing
        # an interactive window or inspecting any existing application.
        root.geometry("1000x740+10000+10000")
        root.deiconify()
        root.update()
        app.monitoring = True
        app.monitor_port = app.fields["port"].get()
        app.confirmed.set(True)
        app.preview_latest = {
            "joint_received_ns": panel.time.perf_counter_ns(),
            "phase": "DISABLED",
        }
        assert str(app.hybrid_button.cget("state")) == "normal"
        app.hybrid_button.invoke()
        root.update_idletasks()

        assert len(attempts) == 1 and not callback_errors
        assert app.status.get() == failure
        assert "shoulder_pan TIMEOUT=0" in app.action_status.get()
        label = app.action_status_label
        assert label.winfo_ismapped()
        for parent in (root, label.master):
            x = label.winfo_rootx() - parent.winfo_rootx()
            y = label.winfo_rooty() - parent.winfo_rooty()
            evidence = {
                "scaling": scaling,
                "status": app.action_status.get(),
                "label": (x, y, label.winfo_width(), label.winfo_height()),
                "parent": (parent.winfo_width(), parent.winfo_height()),
            }
            assert 0 <= x < x + label.winfo_width() <= parent.winfo_width(), evidence
            assert 0 <= y < y + label.winfo_height() <= parent.winfo_height(), evidence
        assert label.winfo_height() >= label.winfo_reqheight()
        assert not app.running and app.next_run is None and app.live is None
        assert app.worker is None and not app.monitor_stop.is_set()
    finally:
        app.monitoring = False
        root.withdraw()
        root.tk.call("tk", "scaling", original_scaling)


def test_compact_action_feedback_retains_the_complete_status(default_tuner):
    app, _, _, _ = default_tuner
    detail = "启动未通过：" + "完整诊断信息；" * 45 + "原始详情末尾。"
    app.status.set(detail)
    app.root.update_idletasks()
    assert app.status.get() == detail
    assert app.action_status.get().startswith("启动未通过：")
    assert 0 < len(app.action_status.get()) <= 153
    assert len(app.action_status.get()) < len(detail)


def test_ipc_initialization_failure_is_visible_and_consumes_review_without_handoff(
    default_tuner, panel, monkeypatch
):
    app, _, _, _ = default_tuner
    root = app.root
    callback_errors = []
    attempts = []

    def fail_ipc(**kwargs):
        attempts.append(kwargs)
        raise OSError("injected IPC allocation failure")

    monkeypatch.setattr(panel, "ProcessLive", fail_ipc)
    monkeypatch.setattr(
        root,
        "report_callback_exception",
        lambda kind, error, traceback: callback_errors.append((kind, str(error))),
    )
    try:
        root.geometry("1000x740+10000+10000")
        root.deiconify()
        root.update()
        app.monitoring = True
        app.monitor_port = app.fields["port"].get()
        app.confirmed.set(True)
        app.preview_latest = {
            "joint_received_ns": panel.time.perf_counter_ns(),
            "phase": "DISABLED",
        }
        app.hybrid_button.invoke()
        root.update_idletasks()

        assert len(attempts) == 1 and not callback_errors
        assert app.status.get() == "遥操未启动：控制进程通信初始化失败：injected IPC allocation failure"
        assert "injected IPC allocation failure" in app.action_status.get()
        label = app.action_status_label
        assert label.winfo_ismapped()
        top = label.winfo_rooty() - root.winfo_rooty()
        assert 0 <= top < top + label.winfo_height() <= root.winfo_height()
        assert label.winfo_height() >= label.winfo_reqheight()
        assert not app.running and app.next_run is None and app.live is None
        assert app.worker is None and app.pending_cleanup is None
        assert app.monitoring and not app.monitor_stop.is_set() and app.exit_confirmed
        reviews = list(panel.OUTPUTS.glob("*/reviewed.json"))
        assert len(reviews) == 1
        review = json.loads(reviews[0].read_text(encoding="utf-8"))
        assert review["consumed"] is True and review["executable"] is False
    finally:
        app.monitoring = False
        root.withdraw()
