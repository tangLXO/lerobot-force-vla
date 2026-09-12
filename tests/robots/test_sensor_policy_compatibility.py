#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE, OBS_TACTILE

ROBOT_STATE_DIM = 1
ACTION_DIM = 2


def state_and_action_features():
    return (
        {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(ROBOT_STATE_DIM,))},
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )


def test_act_baseline_training_batch_uses_robot_only_observation_state() -> None:
    input_features, output_features = state_and_action_features()
    input_features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(1,))
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    policy = ACTPolicy(config)
    batch = {
        OBS_STATE: torch.randn(2, ROBOT_STATE_DIM),
        OBS_ENV_STATE: torch.randn(2, 1),
        ACTION: torch.randn(2, config.chunk_size, ACTION_DIM),
        "action_is_pad": torch.zeros(2, config.chunk_size, dtype=torch.bool),
    }

    loss, loss_dict = policy(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict


def test_pi0_pi05_and_smolvla_keep_robot_only_state_schema_shape() -> None:
    for config_cls in (PI0Config, PI05Config, SmolVLAConfig):
        input_features, output_features = state_and_action_features()
        config = config_cls(input_features=input_features, output_features=output_features)
        assert config.robot_state_feature == PolicyFeature(type=FeatureType.STATE, shape=(ROBOT_STATE_DIM,))
        assert config.action_feature == PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
        assert OBS_TACTILE not in config.input_features
