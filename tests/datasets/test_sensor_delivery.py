#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from lerobot.datasets.sensor_window import SensorStreamReader
from lerobot.utils.constants import OBS_STATE, OBS_TACTILE

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name", ["SENSOR_FRAMEWORK.md", "SENSOR_DATASET_FORMAT.md", "SENSOR_INTEGRATION_GUIDE.md"]
)
def test_sensor_document_links_and_python_examples(name):
    path = ROOT / name
    content = path.read_text(encoding="utf-8")
    assert "process-crash recoverable + replayable on-disk state" in content
    assert "observation.tactile" in content
    assert "frame_features" in content
    for target in re.findall(r"\]\(([^)]+)\)", content):
        if "://" not in target and not target.startswith("#"):
            assert (path.parent / target.split("#")[0]).exists(), target
    for example in re.findall(r"```python\n(.*?)\n```", content, flags=re.DOTALL):
        ast.parse(example)


def test_sensor_benchmark_smoke_measures_real_capture_and_structural_batch_io(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "sensor_pipeline_benchmark", ROOT / "benchmarks/sensor_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "benchmark"
    report = module.run(root, duration_s=1, read_frames=12, batch_size=4, cache_mb=0)
    assert json.loads((root / "benchmark_results.json").read_text()) == report
    assert report["capture"]["frames"] == 30
    assert report["capture"]["raw_rows_per_stream"] == 1000
    assert report["capture"]["state_shape"] == [6]
    assert report["capture"]["tactile_shape"] == [2]
    assert report["capture"]["current_tactile_uses_sync_selection"] is True
    assert report["capture"]["diagnostics"]["sync"]["rows"] == 30
    for stream in report["capture"]["diagnostics"]["streams"].values():
        assert stream["rows"] == 1000
        assert stream["peak_buffer_rows"] <= 4096
        assert stream["overflow_count"] == 0
    for mode in ("sequential", "random", "batch"):
        assert report[mode]["frames"] == 12
        assert report[mode]["resident_decoded_bytes"] == 0
        assert report[mode]["row_groups_decoded"] > 0
    assert report["batch"]["valid_points"] == report["random"]["valid_points"]
    assert report["batch"]["row_groups_decoded"] < report["random"]["row_groups_decoded"]

    dataset = module.LeRobotDataset("benchmark/sensors", root=root)
    assert tuple(dataset.features[OBS_STATE]["shape"]) == (6,)
    assert tuple(dataset.features[OBS_TACTILE]["shape"]) == (2,)
    assert set(dataset.features[OBS_STATE]["names"]).isdisjoint(dataset.features[OBS_TACTILE]["names"])
    reader = SensorStreamReader(root, verify="full", cache_mb=0)
    sync = reader.read_sync(episode_index=0)
    for frame_index, row in enumerate(sync):
        item = dataset[frame_index]
        for tactile_index, instance in enumerate(("left", "right")):
            window = reader.get_window(
                row["frame_anchor_ns"],
                5,
                target_hz=1000,
                max_age_ms=5,
                instance=instance,
                episode_index=0,
            )
            assert bool(window.valid_mask[-1])
            assert int(window.sequence[-1]) == row["sensors"][instance]["sequence"]
            assert float(item[OBS_TACTILE][tactile_index]) == pytest.approx(float(window.values[-1, 0]))
