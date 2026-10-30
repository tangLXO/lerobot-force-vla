"""Offline startup precedence and validation for both desktop entry points."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def options(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    monkeypatch.delenv("MOTORBRIDGE_TX_GAP_US", raising=False)
    return importlib.import_module("communication_options")


@pytest.mark.parametrize(
    "explicit,environment,expected,source",
    [
        (None, None, 120, "官方默认"),
        (None, "1000", 1000, "环境变量 MOTORBRIDGE_TX_GAP_US"),
        (800, "1000", 800, "命令行"),
        (0, "invalid", 0, "命令行"),
        (None, "0", 0, "环境变量 MOTORBRIDGE_TX_GAP_US"),
    ],
)
def test_precedence(options, monkeypatch, explicit, environment, expected, source):
    if environment is not None:
        monkeypatch.setenv("MOTORBRIDGE_TX_GAP_US", environment)
    assert options.resolve_command_tx_gap(explicit) == (expected, source)


@pytest.mark.parametrize("invalid", ["", "bad", "-1", "nan", "inf", "-inf"])
@pytest.mark.parametrize("from_environment", [False, True])
def test_invalid_interval(options, monkeypatch, invalid, from_environment):
    if from_environment:
        monkeypatch.setenv("MOTORBRIDGE_TX_GAP_US", invalid)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        options.resolve_command_tx_gap(None if from_environment else invalid)


@pytest.mark.parametrize("entry", ["b601_desktop", "b601_tuner"])
def test_invalid_environment_exits_before_ui_or_connection(options, monkeypatch, entry):
    monkeypatch.setenv("MOTORBRIDGE_TX_GAP_US", "nan")
    with pytest.raises(SystemExit) as error:
        importlib.import_module(entry).main([])
    assert error.value.code == 2


def test_sequential_and_gripper_description(options):
    for mode, gripper_only in [("sequential", False), ("cached", True)]:
        assert "不适用" in options.communication_description(mode, 120, "官方默认", gripper_only=gripper_only)
