# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations

import abc
import contextlib
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

from lerobot.datasets.sensor_stream import (
    SensorQueueOverflowError,
    SensorRecorderError,
)
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.robots import SensorizedRobot
from lerobot.sensors import SensorDataUnavailableError
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)

SENSOR_FATAL_ERRORS = (
    SensorDataUnavailableError,
    SensorQueueOverflowError,
    SensorRecorderError,
)


def process_robot_observation(processors: ProcessorContext, observation: dict) -> dict:
    """Run observation processing plus the optional sensor modality invariant."""
    process = getattr(processors, "process_observation", None)
    if callable(process):
        return process(observation)
    processed = processors.robot_observation_processor(observation)
    validator = getattr(processors, "observation_validator", None)
    if validator is not None:
        validator(observation, processed)
    return processed


def current_sensor_capture_metadata(ctx: RolloutContext) -> dict | None:
    """Return an immutable snapshot paired with the latest raw observation."""
    robot = ctx.hardware.robot_wrapper.inner
    if not isinstance(robot, SensorizedRobot):
        return None
    metadata = getattr(ctx.data, "current_capture_metadata", None)
    if metadata is None:
        raise SensorRecorderError("A sensorized observation is missing its capture metadata snapshot.")
    return deepcopy(metadata)


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.  This is also the extension point
    for third-party strategies: subclass it next to a registered
    :class:`RolloutStrategyConfig` and ``lerobot-rollout --strategy.type=<name>``
    drives it with no edit to LeRobot (see "Bring your own strategy" in
    ``docs/source/inference.mdx``).

    Lifecycle: ``setup()`` once, then ``run()``, then ``teardown()`` once.
    A strategy whose config declares ``supports_interactive = True`` is also
    driven by ``--interactive=true``, which calls ``run()`` once per
    start/stop segment.  Such a strategy must keep ``run()`` restartable:

    - never finalize the dataset in ``run()`` — that belongs in ``teardown()``;
      at most save a partial tail episode when a segment ends;
    - keep state that must survive a segment on the instance, not in ``run()``
      locals (the ``CycleTimer`` is deliberately the other way round, see ``run()``);
    - never bind keyboard/terminal listeners — stdin belongs to the command prompt;
    - call ``engine.pump_query(obs_processed)`` once at the end of every tick, see
      ``run()``.

    One-shot strategies (``supports_interactive = False``, the default) are
    free to finalize on ``run()`` exit, e.g. via ``VideoEncodingManager``.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None
        self._cached_capture_metadata: dict | None = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        ctx.runtime.active_strategy = self
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self.reset_control_state()
        self._engine.start()
        self._warmup_flushed = False
        logger.info("Inference engine started")

    def reset_control_state(self) -> None:
        """Clear episode-scoped control state so a paused session can restart cleanly.

        Resets the inference engine (policy hidden state, action queues), the action
        interpolator and the cached processed observation; pacing state is untouched.
        ``RolloutController`` calls it on its serve thread before each run segment.
        Only call while the control loop is not running: these resets are not synchronized
        against a live loop.  A caller that resets control state while a loop runs — or a
        strategy that hoists its timer onto the instance — must also call ``timer.restart()``.
        """
        if self._engine is not None:
            self._engine.reset()
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None
        self._cached_capture_metadata = None

    def _process_observation_and_notify(self, ctx: RolloutContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = process_robot_observation(ctx.processors, obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
            self._cached_capture_metadata = current_sensor_capture_metadata(ctx)
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, timer: CycleTimer) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  Warmup ticks are paced through *timer* so the loop cadence
        stays anchored.  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            timer.wait()
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            timer.restart()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _handle_sensor_failure(self, ctx: RolloutContext, exc: BaseException) -> None:
        """Latch safe-stop state, drop pending actions, and abort recording."""
        logger.error("Fatal required-sensor failure; stopping rollout: %s", exc)
        if self._engine is not None:
            try:
                self._engine.pause()
            except Exception:
                logger.exception("Inference engine pause failed during sensor safe-stop")
            try:
                self._engine.reset()
            except Exception:
                logger.exception("Inference engine reset failed during sensor safe-stop")
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None
        self._cached_capture_metadata = None
        robot = ctx.hardware.robot_wrapper.inner
        if isinstance(robot, SensorizedRobot):
            robot.latch_fatal_error(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
        recorder = getattr(ctx.data, "sensor_recorder", None)
        dataset = ctx.data.dataset
        if dataset is not None:
            try:
                dataset.clear_episode_buffer()
            except Exception:
                logger.exception("Failed to clear main episode buffer during sensor safe-stop")
        if recorder is not None:
            try:
                if recorder.is_active:
                    recorder.abort_episode(str(exc))
                elif recorder.has_prepared_episode:
                    recorder.abort_prepared(str(exc))
            except Exception:
                logger.exception("Failed to abort sensor sidecar after fatal sensor error")
        ctx.runtime.shutdown_event.set()

    def _teardown_hardware(
        self,
        hw: HardwareContext,
        return_to_initial_position: bool = True,
        sensor_recorder=None,
    ) -> None:
        """Stop the inference engine, optionally return robot to initial position, and disconnect hardware."""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        if sensor_recorder is not None:
            try:
                sensor_recorder.close()
            except Exception:
                logger.exception("Sensor recorder cleanup failed during hardware teardown")
        robot = hw.robot_wrapper.inner
        sensor_fatal = isinstance(robot, SensorizedRobot) and robot.has_fatal_error
        robot_hardware_connected = (
            robot.inner.is_connected if isinstance(robot, SensorizedRobot) else robot.is_connected
        )
        if robot_hardware_connected:
            if return_to_initial_position and hw.initial_position and not sensor_fatal:
                logger.info("Returning robot to initial position before shutdown...")
                self.return_to_initial_position(hw)
            elif sensor_fatal:
                logger.error("Skipping return-to-initial-position after fatal sensor failure")
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> bool:
        """Smoothly interpolate the robot back to its initial position.

        Returns ``True`` when the interpolation completed, ``False`` when it failed
        partway — the robot is then at an arbitrary pose, so callers must not report
        a completed reset on ``False``.
        """
        robot = hw.robot_wrapper
        target = hw.initial_position
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)
            return False
        return True

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to the visualization backend if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_visualization_data(
            cfg.display_mode,
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.).

        The default only attaches and starts the inference engine; an override must
        call ``self._init_engine(ctx)`` (or ``super().setup(ctx)``) first.
        """
        self._init_engine(ctx)

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires.

        Implementations must call ``engine.resume()`` before entering their loop
        (async backends start paused, and the interactive controller pauses again at
        the end of every segment), and ``engine.pump_query(obs_processed)`` at the end
        of every tick — the text-query channel only advances through it, and a
        multi-second generation must not sit inside the action path.

        Each ``run()`` call builds its own ``CycleTimer`` and reports it through
        ``timer.log_run_summary()`` from its ``finally``: a fresh timer's start-up
        exemption is what absorbs the interpolator that ``reset_control_state()``
        re-primes at every ``/start``, and each segment gets its own cadence report.
        """

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: finalize dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
    timer: CycleTimer | None = None,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    When *timer* is given, the engine pull and the robot send are timed as the
    ``infer`` and ``send`` steps of its cadence summary, and a tick with no action
    to send is counted there.  Note that on async backends ``infer`` is only a
    queue pull — inference runs off-thread, so its latency surfaces as starved
    ticks rather than as loop-body time.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys
    # ``nullcontext`` accepts (and ignores) the section name, so it stands in for
    # ``timer.section`` verbatim when no timer was passed.
    section = timer.section if timer is not None else contextlib.nullcontext

    try:
        ctx.hardware.robot_wrapper.check_health()
        recorder = getattr(ctx.data, "sensor_recorder", None)
        if recorder is not None:
            recorder.check_health()
    except SENSOR_FATAL_ERRORS as exc:
        strategy = getattr(ctx.runtime, "active_strategy", None)
        if strategy is not None:
            strategy._handle_sensor_failure(ctx, exc)
        raise

    if interpolator.needs_new_action():
        with section("infer"):
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        if timer is not None:
            timer.note_starved_tick()
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    with section("send"):
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        send_sensor_safe_action(ctx, processed)
    return action_dict


def sensor_safe_observation(ctx: RolloutContext) -> dict:
    """Start sidecar capture before observation and safe-stop on sensor failure."""
    recorder = getattr(ctx.data, "sensor_recorder", None)
    dataset = ctx.data.dataset
    if (
        recorder is not None
        and getattr(ctx.data, "sensor_recording_enabled", True)
        and not recorder.is_active
        and not recorder.has_prepared_episode
    ):
        try:
            recorder.start_episode(dataset.num_episodes, dataset=dataset)
            recorder.wait_until_ready()
        except SENSOR_FATAL_ERRORS as exc:
            strategy = getattr(ctx.runtime, "active_strategy", None)
            if strategy is not None:
                strategy._handle_sensor_failure(ctx, exc)
            raise
    check_sensor_health(ctx)
    try:
        observation = ctx.hardware.robot_wrapper.get_observation()
        robot = ctx.hardware.robot_wrapper.inner
        ctx.data.current_capture_metadata = (
            deepcopy(robot.last_capture_metadata) if isinstance(robot, SensorizedRobot) else None
        )
        return observation
    except SENSOR_FATAL_ERRORS as exc:
        strategy = getattr(ctx.runtime, "active_strategy", None)
        if strategy is not None:
            strategy._handle_sensor_failure(ctx, exc)
        raise


def check_sensor_health(ctx: RolloutContext) -> None:
    """Run robot and recorder fatal checks before any direct action path."""
    try:
        ctx.hardware.robot_wrapper.check_health()
        recorder = getattr(ctx.data, "sensor_recorder", None)
        if recorder is not None and recorder.is_active:
            recorder.check_health()
    except SENSOR_FATAL_ERRORS as exc:
        strategy = getattr(ctx.runtime, "active_strategy", None)
        if strategy is not None:
            strategy._handle_sensor_failure(ctx, exc)
        raise


def send_sensor_safe_action(ctx: RolloutContext, action) -> None:
    """Check and send one action, applying the full sensor safe-stop on failure."""
    check_sensor_health(ctx)
    try:
        ctx.hardware.robot_wrapper.send_action(action)
    except SENSOR_FATAL_ERRORS as exc:
        strategy = getattr(ctx.runtime, "active_strategy", None)
        if strategy is not None:
            strategy._handle_sensor_failure(ctx, exc)
        raise


def add_dataset_frame(ctx: RolloutContext, frame: dict, capture_metadata: dict | None = None) -> None:
    """Add one main frame and its matching Sync row."""
    dataset = ctx.data.dataset
    recorder = getattr(ctx.data, "sensor_recorder", None)
    if recorder is not None and capture_metadata is None:
        raise SensorRecorderError("A sensorized Dataset frame is missing its capture metadata snapshot.")

    dataset.add_frame(frame)
    if recorder is None:
        return
    try:
        recorder.record_sync(None, capture_metadata)
    except SENSOR_FATAL_ERRORS as exc:
        strategy = getattr(ctx.runtime, "active_strategy", None)
        if strategy is not None:
            strategy._handle_sensor_failure(ctx, exc)
        raise


def save_dataset_episode(ctx: RolloutContext) -> None:
    """Commit the current main episode and sensor sidecars atomically."""
    recorder = getattr(ctx.data, "sensor_recorder", None)
    if recorder is None:
        ctx.data.dataset.save_episode()
        return
    if recorder.is_active:
        recorder.prepare_episode(task_info=[ctx.runtime.cfg.dataset.single_task])
    recorder.commit_prepared(ctx.data.dataset)


def discard_dataset_episode(ctx: RolloutContext, reason: str) -> None:
    """Clear the main buffer and persist an ABORTED sensor transaction."""
    ctx.data.dataset.clear_episode_buffer()
    recorder = getattr(ctx.data, "sensor_recorder", None)
    if recorder is None:
        return
    if recorder.is_active:
        recorder.abort_episode(reason)
    elif recorder.has_prepared_episode:
        recorder.abort_prepared(reason)
