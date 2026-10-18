"""Independent adversarial evidence admission tests; all artifacts are synthetic."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lerobot.robots.rebot_b601_follower.arm_acceptance import (
    command_profile_digest,
)
from lerobot.robots.rebot_b601_follower.control_protocol import B601JointCommandProfile
from lerobot.robots.rebot_b601_follower.full_arm_validation import (
    JOINT_NAMES,
    ValidationLeaderSample,
)
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from tests.force_control.full_arm_validation_helpers import digest, make_full_validation_files, write_json


@pytest.fixture
def validation_evidence(tmp_path):
    return make_full_validation_files(tmp_path)


def admit(evidence):
    config, control = evidence
    return config.validate_binding(DEFAULT_B601_JOINTS, control, port="FAKE_VALIDATION")


def rewrite_result(evidence, mutate):
    config, control = evidence
    binding = config.micro_trials[0]
    path = Path(binding.result_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    updated = replace(binding, result_sha256=write_json(path, value))
    return replace(config, micro_trials=(updated, *config.micro_trials[1:])), control


def test_admission_stays_validation_only_and_does_not_mutate_evidence(validation_evidence):
    config, _ = validation_evidence
    paths = [Path(binding.result_path) for binding in config.micro_trials]
    before = [(path.stat().st_mtime_ns, digest(path)) for path in paths]
    admission = admit(validation_evidence)
    assert admission.production_arm_authorized is False
    assert len(admission.micro_limits) == 6
    assert before == [(path.stat().st_mtime_ns, digest(path)) for path in paths]


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", "true"),
        ("executed", 1),
        ("production_arm_authorized", True),
        ("cleanup_pending", True),
        ("sensor_closed", False),
    ],
)
def test_summary_flags_cannot_promote_partial_or_micro_evidence(validation_evidence, field, value):
    with pytest.raises(ValueError):
        admit(rewrite_result(validation_evidence, lambda row: row.update({field: value})))


def test_reviewed_config_digest_must_match_the_executed_result_binding(validation_evidence):
    changed = rewrite_result(validation_evidence, lambda row: row.update(config_sha256="0" * 64))
    with pytest.raises(ValueError):
        admit(changed)


def test_one_self_asserted_result_row_is_not_a_real_micro_timeline(validation_evidence):
    config, _ = validation_evidence
    binding = config.micro_trials[0]
    path = Path(binding.result_path).with_name("events.jsonl")
    final = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    path.write_text(json.dumps(final) + "\n", encoding="utf-8")
    changed = rewrite_result(validation_evidence, lambda row: row.update(evidence_rows=1))
    with pytest.raises(ValueError):
        admit(changed)


def test_source_manifest_cannot_drop_all_critical_executed_modules(validation_evidence):
    config, _ = validation_evidence
    path = Path(config.micro_trials[0].result_path).parent / "source_snapshot/manifest.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["files"] = source["files"][:1]
    write_json(path, source)
    with pytest.raises(ValueError):
        admit(validation_evidence)


def test_source_manifest_cannot_reference_a_file_outside_snapshot(validation_evidence):
    config, _ = validation_evidence
    path = Path(config.micro_trials[0].result_path).parent / "source_snapshot/manifest.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["files"][0]["path"] = "../result.json"
    write_json(path, source)
    with pytest.raises(ValueError):
        admit(validation_evidence)


def test_pinned_artifact_identity_rejects_later_content_changes(validation_evidence):
    config, _ = validation_evidence
    admission = admit(validation_evidence)
    path = Path(config.micro_trials[0].result_path)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        admission.require_unchanged()


def test_micro_profile_cannot_expand_the_actual_single_joint_acceptance_interface(validation_evidence):
    config, control = validation_evidence
    binding = config.micro_trials[0]
    config_path = Path(binding.reviewed_config_path)
    reviewed = json.loads(config_path.read_text(encoding="utf-8"))
    reviewed["control"]["profiles"][0]["max_relative_deg"] = 2.0
    profile = B601JointCommandProfile(**reviewed["control"]["profiles"][0])
    profile_sha = command_profile_digest(profile)
    reviewed["acceptance"]["profile_sha256"] = profile_sha
    config_sha = write_json(config_path, reviewed)
    result_path = Path(binding.result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["config_sha256"] = config_sha
    result["backend_result"]["request"]["profile_sha256"] = profile_sha
    events_path = result_path.with_name("events.jsonl")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[-1]["request"]["profile_sha256"] = profile_sha
    events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    updated = replace(
        binding, reviewed_config_sha256=config_sha, result_sha256=write_json(result_path, result)
    )
    with pytest.raises(ValueError):
        admit((replace(config, micro_trials=(updated, *config.micro_trials[1:])), control))


def test_reviewed_authorization_must_match_the_recorded_backend_request(validation_evidence):
    config, control = validation_evidence
    binding = config.micro_trials[0]
    config_path = Path(binding.reviewed_config_path)
    reviewed = json.loads(config_path.read_text(encoding="utf-8"))
    reviewed["acceptance"]["watchdog_ticks"] = 1999
    config_sha = write_json(config_path, reviewed)
    result_path = Path(binding.result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["config_sha256"] = config_sha
    updated = replace(
        binding, reviewed_config_sha256=config_sha, result_sha256=write_json(result_path, result)
    )
    with pytest.raises(ValueError):
        admit((replace(config, micro_trials=(updated, *config.micro_trials[1:])), control))


def test_admission_pins_native_calibration_in_addition_to_result_and_source_files(validation_evidence):
    config, _ = validation_evidence
    admission = admit(validation_evidence)
    path = Path(config.calibration_file)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        admission.require_unchanged()


@pytest.mark.parametrize(
    "change",
    [
        {"source_config_sha256": "0" * 64},
        {"captured_ns": 0},
        {"response_completed_ns": 2_000_000_001},
        {"sequence": True},
        {"captured_ns": 1_000_000_000},
    ],
)
def test_leader_sample_rejects_wrong_mapping_future_stale_or_false_sequence(validation_evidence, change):
    config, _ = validation_evidence
    sample = ValidationLeaderSample(
        tuple((name, 0.0) for name in JOINT_NAMES),
        1_990_000_000,
        1_995_000_000,
        1,
        config.leader_config_sha256,
    )
    with pytest.raises(ValueError):
        replace(sample, **change).validate(config, 2_000_000_000)
