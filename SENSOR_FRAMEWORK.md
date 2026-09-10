# Sensor Framework Naming and Layering

Status: **Locked**

This document is the source of truth for sensor-framework work in this repository. Follow it unless the user explicitly asks to revise the convention.

## Package and public API

Use this structure for the first implementation:

```text
src/lerobot/sensors/
├── __init__.py
├── sensor.py                 # Sensor, SensorFeature, SensorSample, subscriptions
├── buffer.py                 # bounded-duration online HistoryBuffer
├── synchronization.py        # shared causal latest-before/window selection
├── configs.py                # SensorConfig
├── utils.py                  # make_sensors_from_configs
└── x518/
    ├── __init__.py
    ├── configuration_x518.py # X518SensorConfig, X518ChannelConfig
    └── sensor_x518.py        # X518Sensor
```

The generic top-level API is:

```python
from lerobot.sensors import Sensor, SensorConfig, SensorFeature, SensorSample, make_sensors_from_configs
```

Import hardware-specific classes from their subpackage, for example:

```python
from lerobot.sensors.x518 import X518SensorConfig
```

Use these exact names:

- Base class: `Sensor`, not `BaseSensor`, `GenericSensor`, or `PhysicalSensor`.
- Base config: `SensorConfig`.
- Sample value object: `SensorSample`.
- Factory: `make_sensors_from_configs()`.
- Robot config/runtime collection: `sensors` / `robot.sensors`.
- X518 driver and config: `X518Sensor` / `X518SensorConfig`.
- X518 registration choice: `@SensorConfig.register_subclass("x518")`; the CLI type is `x518`, not `x518_sensor`.
- Sampling-rate field: `sample_rate_hz`.
- Sample timing fields: `timestamp_ns`, `arrival_timestamp_ns`, and `sequence`.
- Boolean validity field: `is_valid`, not `valid`.

The common sensor interface uses `features`, `is_connected`, `connect()`, `read()`, `async_read()`, `read_latest()`, and `disconnect()`. `connect()` owns background-reader startup and `disconnect()` owns shutdown. Add `read_window()` only when buffered history is actually implemented; do not introduce public `BufferedSensor`, `start_streaming()`, or `stop_streaming()` abstractions merely for naming symmetry.

## Semantic feature keys and qualification

`Sensor.features` exposes hardware-independent paths relative to one configured stream:

```text
<location_path>.<quantity>
```

`location_path` may contain multiple dot-separated components; it is not restricted to one token. The locked X518 two-finger feature paths are:

```text
left.normal_force
right.normal_force
```

Runtime and Dataset state names are qualified exactly once by the configured stream instance:

```text
sensor.<instance>.<feature_path>
```

For `instance: gripper_force`, the names above become
`sensor.gripper_force.left.normal_force` and
`sensor.gripper_force.right.normal_force`. For a bimanual robot, use stable instance roles
such as `left_gripper_force` and `right_gripper_force`; do not add hardware identity.

Feature rules:

- Use lowercase `snake_case` tokens.
- Name physical meaning, not the vendor, transport, register, or ADC channel.
- Use `normal_force` for contact-normal force; do not mislabel it `force_z` unless it is truly the z component in a documented coordinate frame.
- Spell out axis quantities as `force_x`, `force_y`, `force_z`, `torque_x`, etc.; do not use `fx`, `fy`, `fz`, `tx`, etc.
- Do not put units in feature keys (`force_n`, `torque_nm`, and `distance_mm` are forbidden). Semantic sensor outputs use SI units, documented and tested at the driver boundary.
- Define locations from stable robot/URDF component and coordinate-frame names, never from camera or operator viewpoint.
- Do not put `x518`, `channel_0`, `ch1`, serial ports, or similar hardware details in `RobotObservation` feature keys.

At the current integration stage, selected scalar Sensor features join the existing
`observation.state` vector and retain their fully qualified `sensor.*` names in its `names`
metadata. Do not create a separate `observation.tactile` Dataset field until a policy with
an independent temporal/tactile encoder is implemented.

## Instance, channel, and provenance names

Name entries in the `sensors` config mapping by stable installation/function rather than hardware model or an arbitrary ordinal. For the current two-finger installation, prefer:

```python
sensors = {"gripper_force": X518SensorConfig(...)}
```

Use `channel` for the device channel field and obey the X518 protocol's native numbering. If code must distinguish an array offset from a device identifier, use `channel_index` and `channel_id` respectively; do not silently convert a documented 1-based hardware identifier into a 0-based identifier.

Hardware identity must disappear from semantic feature names, but it must remain in configuration and provenance metadata (device type, firmware, calibration, raw capture metadata) for reproducibility.

`state_features` contains relative feature paths. `None` selects all semantic features, an
explicit list selects and orders a subset, and `[]` makes the stream raw-only. State order is
the original Robot state, then Sensor config order, then `state_features` order. A stream that
contributes state must be required.

## Sample, history, and causal-time contract

`SensorFeature` declares one scalar (`shape=()`) with an explicit NumPy/Arrow `dtype` and
unit. `SensorSample.values` contains calibrated semantic/SI values; optional
`native_values` and `native_payload` preserve device-native data. Valid samples must exactly
match the semantic schema. Invalid acquisition attempts retain status/error metadata and still
consume a framework sequence number.

All online selection uses one host-monotonic clock domain. A sample is eligible for target
`t` only when all of the following hold:

```text
sample.timestamp_ns <= t
sample.arrival_timestamp_ns <= t
sample.is_valid
t - sample.timestamp_ns <= resolved_max_age
```

The first timestamp is measurement time; the second is when the framework made the sample
available. This same rule is shared by current-value and every temporal-window grid point, so
late-arriving or future data can never leak into an earlier observation.

`HistoryBuffer` is a short online cache used only by `read_latest()`,
`read_latest_before()`, and `read_window()`. Native-rate recording uses a separate bounded
subscriber queue per recorder/episode. Publishing never blocks on a subscriber: overflow is
latched and invalidates the episode. Recorder completeness must never depend on history
duration.

## Robot and Dataset boundary

Sensor attachment is composition through `SensorizedRobot`; do not add Sensor fields to the
central `RobotConfig` or change `make_robot_from_config()`. For each observation the wrapper
records start time, obtains the inner Robot observation, immediately fixes one shared frame
anchor, performs all causal Sensor selections against that anchor, merges selected state, and
then records completion time.

The main LeRobot Dataset remains fixed-FPS and compatible with existing state-consuming
policies. Native-rate Raw data, per-frame Sync metadata, dynamic provenance, and transaction
journals live in the Sensor Sidecar described by [`SENSOR_DATASET_FORMAT.md`](./SENSOR_DATASET_FORMAT.md).
Sliding windows are reconstructed by a Reader/Adapter and are never duplicated persistently.

## Layer boundaries

`lerobot.sensors` performs connection, acquisition, timestamping, validation, calibration/unit conversion, buffering, and semantic output. It must not contain robot-specific control, PID/admittance/impedance control, policy code, or Force-VLA experiment logic.

## Production lifecycle and recovery contract

Arrival timestamps must be non-decreasing, while measurement timestamps may arrive out of order.
Default arrival is generated under the publication lock. `read()` and `async_read()` inspect the
complete unseen sequence interval, return its newest valid sequence and advance the shared cursor
through invalid trailing attempts; invalid-only intervals preserve waiting/timeout/disconnect behavior.

`HistoryBuffer.clear()` resets history and its arrival monotonic state, not Sensor framework
sequence. Drivers must use `_reset_framework_state()` to reset sequence/cursors, which is legal
only without subscribers or recorder leases. Leases cover startup through draining and transaction
cleanup. Any sequence-reset attempt during ownership latches an episode error. Required streams
that need reconnection during an episode latch failure and stop that episode; X518 may reconnect
outside recordings, but may not transparently continue the same required recorded episode.

Recorder startup waits for all required streams, including raw-only streams. Raw and Sync capture
use bounded disk spools, independently of online History. Shutdown timeout retains workers, files
and leases until workers stop. Unknown future windows preserve Raw; explicit finite-age windows
can permit safe trimming, including late measurements during final merge.

The persistence guarantee is **process-crash recoverable + replayable on-disk state**, without
power-loss durability or concurrent Reader/Writer isolation. Sidecar v1 final schemas/layout stay
fixed; transaction journal format v1 uses a fixed active pointer, registered artifact locators and
episode-local logical evidence. Only Writer/Recovery holding the writer lock may recover. Readers
are strictly read-only, refuse live Writers and validate selected shared artifact ranges. See
[`SENSOR_DATASET_FORMAT.md`](./SENSOR_DATASET_FORMAT.md) for verification modes,
active-transaction recovery, Hub subset localization, batch/cache behavior and runtime diagnostics.
Earlier prototype journals are not part of the supported format and have no migration path.

Create `x518/calibration.py` only when calibration has substantial independent behavior. Create `x518/protocol.py` only when register maps, packet parsing, CRC, or transport protocol code warrants a separate module. Do not create empty abstraction files or speculative sensor-category directories.
