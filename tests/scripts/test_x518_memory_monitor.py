# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Diagnostic memory probes must not enumerate processes during acquisition."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def smoke_module():
    for dependency in ("psutil", "datasets", "av", "cv2", "serial", "deepdiff"):
        pytest.importorskip(dependency)
    path = Path(__file__).resolve().parents[2] / "examples/x518/x518_collection_hardware_smoke_test.py"
    spec = importlib.util.spec_from_file_location("x518_collection_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_probe_reads_only_known_process_handles(smoke_module):
    parent = Mock()
    parent.children.side_effect = AssertionError("No process table scan during capture")
    parent.memory_info.return_value = SimpleNamespace(rss=100)
    children = [Mock(), Mock()]
    children[0].memory_info.return_value = SimpleNamespace(rss=20)
    children[1].memory_info.return_value = SimpleNamespace(rss=30)
    for _ in range(3):
        assert smoke_module.sample_process_memory(parent, children) == {
            "parent_rss_bytes": 100,
            "children_rss_bytes": 50,
        }
    parent.children.assert_not_called()
    assert all(child.memory_info.call_count == 3 for child in children)


def test_memory_probe_tolerates_exited_or_inaccessible_children(smoke_module):
    parent = Mock()
    parent.memory_info.return_value = SimpleNamespace(rss=100)
    children = [Mock(), Mock(), Mock()]
    children[0].memory_info.side_effect = smoke_module.psutil.NoSuchProcess(123)
    children[1].memory_info.side_effect = smoke_module.psutil.AccessDenied(456)
    children[2].memory_info.return_value = SimpleNamespace(rss=20)
    assert smoke_module.sample_process_memory(parent, children)["children_rss_bytes"] == 20
