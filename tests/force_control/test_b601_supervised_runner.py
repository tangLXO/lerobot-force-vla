"""Thin supervised runner invokes only the common runtime and saves exit evidence."""

import importlib
import json
from dataclasses import asdict
from pathlib import Path

import draccus
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    return importlib.import_module("b601_supervised_bench")


def reviewed(panel, calibration, settings, tmp_path):
    config = panel.build_config(*settings, calibration, tmp_path / "trace")
    path = tmp_path / "reviewed.json"
    path.write_text(
        json.dumps(
            {
                "config": draccus.encode(config),
                "calibration": asdict(calibration),
                "executable": True,
                "consumed": False,
            }
        )
    )
    return path


def test_offline_review_checks_calibration_and_encodings_without_opening(
    runner, panel, calibration, settings, tmp_path, monkeypatch
):
    path = reviewed(panel, calibration, settings, tmp_path)
    monkeypatch.setattr(runner, "run_runtime", lambda *a, **k: pytest.fail("offline review executed"))
    result = runner.run(tmp_path / "run", reviewed_config=path)
    assert result["transport_opened"] is False
    assert result["velocity_wire_code"] == 1 and result["drive_wire_code"] == 200
    assert not (tmp_path / "run").exists()
    Path(calibration.path).write_text("{}")
    with pytest.raises(ValueError):
        runner.run(tmp_path / "run", reviewed_config=path)


@pytest.mark.parametrize("failure", [False, True])
def test_runner_preserves_canonical_runtime_result_and_failure_shutdown(
    runner, panel, calibration, settings, tmp_path, monkeypatch, failure
):
    path = reviewed(panel, calibration, settings, tmp_path)
    called = []
    shutdown = {
        "transport_closed": True,
        "fault_disposition": {"disabled_feedback_observed": True, "status": "submitted", "error": None},
    }

    def execute(config, live):
        called.append((config, live))
        config.trace_root.mkdir(parents=True)
        (config.trace_root / "manifest.json").write_text(
            json.dumps({"metadata": {"shutdown_evidence": shutdown}})
        )
        if failure:
            raise RuntimeError("injected_send_failure")
        return {"valid": True, "shutdown_evidence": shutdown}

    monkeypatch.setattr(runner, "run_runtime", execute)
    live = object()
    result = runner.run(tmp_path / "run", execute=True, reviewed_config=path, live=live)
    assert len(called) == 1 and called[0][1] is live
    assert called[0][0].trace_root == tmp_path / "run/stream"
    assert result["shutdown_evidence"] == shutdown
    assert bool(result["errors"]) == failure
    saved = json.loads((tmp_path / "run/result.json").read_text())
    assert saved["hardware_ack"] == "unknown" and saved["calibration"] == asdict(calibration)
    assert panel.exit_confirmation(saved)


def test_consumed_profile_cannot_execute(runner, panel, calibration, settings, tmp_path, monkeypatch):
    path = reviewed(panel, calibration, settings, tmp_path)
    data = json.loads(path.read_text())
    data["consumed"] = True
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "run_runtime", lambda *a, **k: pytest.fail("replayed used config"))
    with pytest.raises(ValueError, match="未使用"):
        runner.run(tmp_path / "run", execute=True, reviewed_config=path, live=object())


def test_canonical_trace_errors_are_preserved_in_receipt(
    runner, panel, calibration, settings, tmp_path, monkeypatch
):
    path = reviewed(panel, calibration, settings, tmp_path)
    monkeypatch.setattr(
        runner, "run_runtime", lambda config, live: {"valid": False, "errors": ["raw_source_mismatch"]}
    )
    result = runner.run(tmp_path / "run", execute=True, reviewed_config=path, live=object())
    assert result["valid"] is False and result["errors"] == ["raw_source_mismatch"]
    assert "核验失败" in panel.stop_report(result, tmp_path)["summary"]


def test_grouped_failures_show_feedback_details_and_cleanup_error(
    runner, panel, calibration, settings, tmp_path, monkeypatch
):
    path = reviewed(panel, calibration, settings, tmp_path)
    feedback_error = RuntimeError("invalid_joint_feedback")
    feedback_error.add_note("B601 joint feedback: TimeoutError: B601 read-only deadline expired")

    def execute(config, live):
        raise ExceptionGroup(
            "I-P session failed",
            [
                feedback_error,
                ExceptionGroup("Owner cleanup failed", [RuntimeError("disable feedback unconfirmed")]),
            ],
        )

    monkeypatch.setattr(runner, "run_runtime", execute)
    output = tmp_path / "run"
    result = runner.run(output, execute=True, reviewed_config=path, live=object())
    assert result["errors"] == [
        "RuntimeError: invalid_joint_feedback; B601 joint feedback: "
        "TimeoutError: B601 read-only deadline expired",
        "RuntimeError: disable feedback unconfirmed",
    ]
    assert json.loads((output / "result.json").read_text())["errors"] == result["errors"]
    assert "deadline expired" in panel.stop_report(result, output)["summary"]
    assert not panel.exit_confirmation(result)


def test_owner_group_feedback_note_is_preserved_with_individual_errors(runner):
    group = ExceptionGroup("Owner cleanup failed", [RuntimeError("close failed")])
    group.add_note("joint_error: TimeoutError: B601 no response before deadline")
    assert runner.exception_messages(group) == [
        "Owner cleanup failed: joint_error: TimeoutError: B601 no response before deadline",
        "RuntimeError: close failed",
    ]
