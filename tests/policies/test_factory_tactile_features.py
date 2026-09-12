# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import lerobot.policies.factory as policy_factory
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_TACTILE


def _dataset_meta():
    return SimpleNamespace(
        features={
            OBS_STATE: {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"joint_{index}" for index in range(6)],
            },
            OBS_TACTILE: {
                "dtype": "float32",
                "shape": (2,),
                "names": ["force_left", "force_right"],
            },
            ACTION: {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"joint_{index}" for index in range(6)],
            },
        },
        stats={},
    )


def _policy_config(input_features):
    return SimpleNamespace(
        type="mock",
        device="cpu",
        pretrained_path=None,
        use_peft=False,
        input_features=input_features,
        output_features={},
    )


def _patch_policy_construction(monkeypatch):
    policy = torch.nn.Linear(1, 1)
    policy_class = MagicMock(return_value=policy)
    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: policy_class)
    monkeypatch.setattr(policy_factory, "validate_visual_features_consistency", lambda *args: None)
    return policy


@pytest.mark.parametrize("input_features", [None, {}])
def test_make_policy_default_inputs_exclude_tactile(monkeypatch, input_features):
    cfg = _policy_config(input_features)
    expected_policy = _patch_policy_construction(monkeypatch)

    policy = policy_factory.make_policy(cfg, ds_meta=_dataset_meta())

    assert policy is expected_policy
    assert set(cfg.input_features) == {OBS_STATE}
    assert cfg.output_features[ACTION].type is FeatureType.ACTION


def test_make_policy_preserves_explicit_tactile_input(monkeypatch):
    tactile_feature = PolicyFeature(type=FeatureType.STATE, shape=(2,))
    cfg = _policy_config({OBS_TACTILE: tactile_feature})
    _patch_policy_construction(monkeypatch)

    policy_factory.make_policy(cfg, ds_meta=_dataset_meta())

    assert cfg.input_features == {OBS_TACTILE: tactile_feature}


@pytest.mark.parametrize(
    "rename_map",
    [
        {OBS_TACTILE: OBS_STATE},
        {"observation.force": OBS_TACTILE},
    ],
)
def test_make_policy_rejects_tactile_boundary_renames(rename_map):
    cfg = SimpleNamespace(type="mock")

    with pytest.raises(ValueError, match="tactile modality boundary"):
        policy_factory.make_policy(cfg, ds_meta=SimpleNamespace(), rename_map=rename_map)
