# Sensor Backend Integration Guide

Status: **Implementation Guide — Sidecar v2**

> 本文是新增非视觉传感器 backend 的实施指南。
> [`SENSOR_FRAMEWORK.md`](./SENSOR_FRAMEWORK.md) 是锁定的最高规范；两者发生冲突时，
> 必须以 `SENSOR_FRAMEWORK.md` 为准。修改公共架构、命名或数据分层前，应先显式修订
> Framework，而不是只修改本指南。

## 1. 本指南解决什么问题

当需要接入新的力、力矩、触觉、磁触觉、电容接近或其他非视觉物理传感器时，
本指南用于回答“具体怎样实现一个合格的 backend”。完成接入后，backend 应做到：

- 能通过 `SensorConfig` 配置和 Draccus 注册；
- 能通过 `make_sensors_from_configs()` 构造，但构造时不访问硬件；
- 能在断连状态下描述 `features`；
- 能把原始硬件数据转换为带时间戳、序号和有效性的 `SensorSample`；
- 正确区分 `read()`、`async_read()` 和 `read_latest()`；
- 正确管理连接、后台采样、异常、重连和资源释放；
- 单元测试不依赖真实硬件；
- 真机验证放在 `examples/`，不混入自动化测试。

本层只负责：

```text
硬件连接与协议
        ↓
原始读数解析
        ↓
校准和 SI 单位转换
        ↓
语义 feature 映射
        ↓
SensorSample
```

本层不负责 Robot、Controller、Policy、PID、力控算法或 Dataset 写入。

## 2. 开始编码前

先完成以下准备，不要根据旧版 LeRobot 或其他项目猜测接口：

1. 完整阅读 [`SENSOR_FRAMEWORK.md`](./SENSOR_FRAMEWORK.md)。
2. 阅读当前 Core：
   - `src/lerobot/sensors/configs.py`
   - `src/lerobot/sensors/sensor.py`
   - `src/lerobot/sensors/utils.py`
3. 阅读一个当前 backend；X518 可作为网络型、高频传感器的参考。
4. 阅读设备官方协议，确认：
   - 传输方式；
   - 设备地址或通道编号是否从 0 或 1 开始；
   - 原始数据类型、符号位、字节序和 word swap；
   - 采样率、单位、小数点或 scale；
   - 请求超时与断线表现；
   - 哪些操作是只读，哪些操作会改变设备状态。
5. 在实现前写清楚每个硬件通道对应的物理含义、坐标系和最终 SI 单位。

如果协议资料不足以确定通道、单位或字节序，应停止实现并确认设备事实，不能依靠试猜。

## 3. 目录和命名

一个普通 backend 使用以下最小结构：

```text
src/lerobot/sensors/<device_name>/
├── __init__.py
├── configuration_<device_name>.py
└── sensor_<device_name>.py
```

配套测试与真机检查放在：

```text
tests/sensors/test_<device_name>.py
examples/<device_name>/<device_name>_hardware_smoke_test.py
examples/<device_name>/run_<device_name>_hardware_smoke_test.bat  # Windows 可选
```

只在确有独立职责时增加：

- `protocol.py`：寄存器表、报文编码/解码、CRC、socket/serial 传输已形成独立逻辑；
- `calibration.py`：校准流程或数学变换足够复杂，需要独立测试和复用。

不要创建空文件、空抽象层或按 `force/`、`tactile/`、`proximity/` 预先分类的目录。

命名示例：

```text
DeviceSensorConfig
DeviceSensor
DeviceChannelConfig       # 只有设备确实存在通道配置时才需要
@SensorConfig.register_subclass("device")
```

禁止用 `BaseSensor`、`GenericSensor`、`SensorManager` 或项目名称替代公共 `Sensor`。

## 4. 编写 SensorConfig

配置类继承 `SensorConfig`，使用当前 Draccus `ChoiceRegistry`，不要另建 registry：

```python
from dataclasses import dataclass

from ..configs import SensorConfig


@SensorConfig.register_subclass("device")
@dataclass(kw_only=True)
class DeviceSensorConfig(SensorConfig):
    host: str
    request_timeout_s: float = 0.05
```

配置层负责描述和验证“用户想连接什么设备”，但不能在 `__post_init__()` 中访问硬件。

公共字段 `frame_features` 决定哪些 semantic scalar 会参与固定 FPS 的 current-frame view：

- `None`：按 backend `features` 顺序选择全部 semantic features；
- 非空列表：选择并固定给定相对 feature 的顺序；
- `[]`：只录 Raw/Sync，不创建 current-frame Dataset feature。

`state_features` 是历史 Sidecar v1 字段，在 v2 中不是兼容 alias。只要
`frame_features` 不是 `[]`，stream 就必须为 required，并且必须能够解析有限的
`max_age_ms`。

### 必须做到

- 注册名称简短、稳定、使用 `snake_case`；
- 使用公共字段 `sample_rate_hz`，不要创建 `frequency`、`fps` 等同义字段；
- 对端口范围、地址、超时、重试次数、采样率等进行类型和边界验证；
- 拒绝 `bool` 冒充 `int`，并拒绝 `NaN`、正负无穷等无效浮点值；
- 配置错误在启动线程或打开硬件之前暴露；
- 用 `draccus.decode(SensorConfig, {"type": ...})` 测试真实解析路径。

### 通道与语义映射

硬件通道和上层语义必须通过配置显式关联。推荐让字典键表示 semantic feature，值表示硬件通道：

```python
channels = {
    "left.normal_force": DeviceChannelConfig(channel=1),
    "right.normal_force": DeviceChannelConfig(channel=2),
}
```

不要在驱动内部默认规定 `channel 1 == left_finger`。相同设备换到腕部、右臂或其他机器人后，
应该只修改配置，不修改驱动。

设备原生编号必须原样保留。若协议使用 1-based 编号，配置中的 `channel` 也使用 1-based；
只有访问 Python 数组时才在局部变量中换算为 `channel_index`。

## 5. Feature 契约与单位

backend feature 名称必须是 stream 内的相对语义路径：

```text
<location_path>.<quantity>
```

例如：

```text
left.normal_force
end_effector.force_x
end_effector.torque_z
left.distance
```

配置实例名由运行时只限定一次，形成
`sensor.<instance>.<feature_path>`。backend 不得自行加入 `sensor`、`tactile` 或实例前缀。

需要同时满足：

- 每个 token 使用小写 `snake_case`；
- 表达物理意义，不表达型号、寄存器、串口或 ADC 通道；
- 坐标轴名称必须对应已记录的坐标系；
- 单位不写入 key；
- 对外数值使用 SI：力为 N、力矩为 N·m、距离为 m、时间为 s；
- `features` 的键与每个 `SensorSample.values` 的键完全一致；
- `features` 在没有连接硬件时也能读取。

当前 Core 的 `features` 是 `dict[str, SensorFeature]`，每项显式声明 `dtype`、`unit` 和
`shape=()`。多轴传感器应拆成多个语义标量，
例如 `force_x`、`force_y`、`force_z`。不要用 tuple 冒充非视觉数组，因为当前 LeRobot 数据转换路径
仍可能把 tuple 当作图像 shape。若未来需要触觉矩阵或点云，应先显式扩展 Core 和 Dataset 契约。

硬件型号不能出现在 semantic feature 中，但必须保留在配置、设备参数、校准记录或 provenance 中，
以保证实验能够复现。

## 6. SensorSample 的生成

每次发布给上层的样本统一使用：

```python
SensorSample(
    timestamp_ns=...,
    arrival_timestamp_ns=...,
    sequence=...,
    values={...},
    is_valid=True,
)
```

推荐行为：

- `timestamp_ns` 表示完整响应被接收并可用于解析的时刻；当前主机采样推荐使用
  `time.perf_counter_ns()`，这样可与 `read_latest()` 的年龄计算处于同一单调时钟域；
- 如果使用设备自身时钟，必须明确记录时钟域以及设备时钟到主机时钟的映射方式；
- `arrival_timestamp_ns` 表示样本进入框架、可以被在线调用方使用的 host-monotonic 时刻；
- `sequence` 由框架分配，每个成功发布的 valid/invalid acquisition 都递增；校验拒绝的发布不消耗序号。
  重置必须经过 `_reset_framework_state()`，且不能存在 Recorder lease 或 subscriber；
  硬件序号只能写入 `hardware_sequence`；
- arrival 必须单调不减，measurement 可以倒序；默认 arrival 由发布锁内生成，不在锁外预取；
- `values` 应一次性包含 `features` 声明的全部键；
- `native_values` 用于 schema 化的 ADC/register 数值，`native_payload` 用于可选原始二进制；
- 协议超时、半包或校验失败应发布带状态/错误的 invalid acquisition attempt，
  随后按 backend 策略重试或重连；invalid 样本允许缺失 semantic values。

不要把 `timestamp_ns`、`sequence`、连接状态或硬件型号塞入 `values`；它们不是物理 feature。

## 7. 生命周期与资源管理

### `__init__()`

- 保存配置；
- 创建锁、事件、condition 和空缓存；
- 可以构造尚未连接的协议客户端；
- 不打开 socket/serial，不启动线程，不读取设备。

这样 factory 只负责创建对象，不会因为硬件暂时离线而失败。

### `is_connected`

应反映真实可用状态。对于后台采样型设备，通常至少要求生命周期已激活且采样线程存活；
不能只依赖一个永远不会自动更新的布尔值。

### `connect()`

推荐顺序：

1. 拒绝重复连接，抛出 `DeviceAlreadyConnectedError`；
2. 使用 `_reset_framework_state()` 检查占用并重置旧样本、framework sequence 和消费 cursor，
   然后清空停止事件；活动 Recorder 占用期间的重置尝试必须锁存错误，不能绕过保护；
3. 打开传输连接；
4. 读取并验证设备实际设置；
5. 确认单位、数据格式和采样率与配置兼容；
6. 标记生命周期有效并启动后台读取；
7. 任一步骤失败都关闭已创建资源，恢复为可再次连接的干净状态。

### 重连

只有设备需求明确时才实现自动重连。需要重连时：

- required 流在活动 episode 中需要重连时，调用 `_notify_reconnect_required()` 锁存故障并停止
  该 episode；不得透明重连后继续提交。非录制状态仍可按 backend 策略恢复；
- Recorder lease 从启动持续到排空与事务清理结束，unsubscribe 后也不能自行归零 sequence；

- 读取失败后先关闭损坏的连接；
- 用可被停止事件打断的等待代替不可中断的长时间 `sleep()`；
- 重连成功后重新读取设备设置，不能沿用断线前的单位或格式假设；
- 重连期间不发布虚构值或不完整样本；
- 对高频重复错误进行日志节流，避免淹没控制循环日志。

### `disconnect()`

推荐顺序：

1. 拒绝未连接状态，抛出 `DeviceNotConnectedError`；
2. 先使生命周期失效并设置停止事件；
3. 唤醒所有正在等待新样本的读取调用；
4. 关闭 socket/serial，以打断阻塞 I/O；
5. 等待后台线程退出，但设置有限超时；
6. 再次关闭连接，防止重连线程竞争造成句柄泄漏；
7. 清理线程、设备设置和运行时采样率。

析构函数只是安全网，正常使用必须显式 `disconnect()` 或使用 `with sensor:`。

## 8. 读取与 History 接口

三种方法都返回 `SensorSample`，但语义不同：

### `read()`

- 检查完整未消费 sequence 区间，返回其中 sequence 最新的 valid 样本；
- cursor 推进到区间末尾，包括 invalid 尾部；没有未消费 valid 样本时继续等待；
- 适合希望控制循环跟随传感器节奏的调用方；
- 断开连接时必须唤醒阻塞调用并抛出 `DeviceNotConnectedError`。

### `async_read(timeout_ms=200)`

- 与 `read()` 共享消费 cursor，检查完整未消费区间并返回其中 sequence 最新的 valid 样本；
- 若当前没有未消费样本，则等待到超时；
- 超时抛出 `TimeoutError`；
- 名称沿用 Camera API，它是普通阻塞方法，不是 `async def` 协程。

### `read_latest(max_age_ms=500)`

- 立即查看最新缓存，不等待、不消费；
- 尚未产生任何样本时抛出 `RuntimeError`；
- 样本年龄大于 `max_age_ms` 时抛出 `TimeoutError`；
- 适合 Sensor 与 Robot 主循环异频的场景。

### `read_latest_before(target_timestamp_ns, max_age_ms)`

- 从短期 History 中选择不晚于 target 的最近 valid 样本；
- 必须同时满足 measurement timestamp 和 arrival timestamp 不晚于 target；
- 最近候选 invalid、迟到或 stale 时继续回退更早候选，而不是返回未来值或填零。

### `read_window(start_timestamp_ns, end_timestamp_ns)`

- 返回 `(start, end]` 内的原始 History 样本；
- 只服务在线短期读取，不作为原频录制通道；
- Dataset temporal window 由 Sidecar Reader 重建，不持久化滑动窗口。

超时和最大年龄参数应拒绝错误类型、`NaN`、无穷值及不符合该方法约定的范围。

上述 History API 与 Recorder subscriber queue 完全独立。每个录制 episode 使用自己的有界
subscriber queue；Sensor publish 只做非阻塞 `put_nowait()`，overflow 必须锁存并使该 episode 失败。

## 9. 协议层与校准层

### 何时创建 `protocol.py`

当 backend 涉及报文、寄存器或传输状态机时，把这些内容放入私有协议层：

- 建立和关闭底层连接；
- 生成请求报文；
- 接收完整响应；
- 校验 transaction ID、设备 ID、功能码、长度、CRC 等；
- 处理符号位、字节序和 word swap；
- 解码设备报告的单位、scale 和采样率；
- 返回原始通道值或设备设置。

协议层不应知道 `left.normal_force` 等安装语义。semantic feature 映射和最终 SI 转换由
`DeviceSensor` 完成。

默认优先实现最小、安全的协议子集。若当前需求只读取数据，不要顺便提供写寄存器接口。

### 何时创建 `calibration.py`

简单的 scale、offset 或单位换算可以留在 driver。只有校准包含独立采集流程、多阶段拟合、
持久化格式或较复杂数学行为时才拆分，并为其单独编写测试。

## 10. Factory 与导出

顶层公共 API 保持硬件无关：

```python
from lerobot.sensors import Sensor, SensorConfig, SensorSample, make_sensors_from_configs
```

具体设备从自己的子包导入：

```python
from lerobot.sensors.device import DeviceSensor, DeviceSensorConfig
```

backend 的 `__init__.py` 导出配置类和实现类，使通用设备 factory 能按照
`DeviceSensorConfig -> DeviceSensor` 的命名约定找到实现。不要把具体设备导入
`lerobot.sensors.__init__`，否则顶层导入可能意外加载硬件专属依赖。

必须测试：

```python
sensors = make_sensors_from_configs({"installation_name": config})
assert isinstance(sensors["installation_name"], DeviceSensor)
assert not sensors["installation_name"].is_connected
```

factory 只构造对象，不连接硬件。映射键使用稳定安装名称或功能名称，不使用任意序号。

## 11. 单元测试

Sensor 单元测试必须在没有真实硬件时运行。推荐使用 fake socket、fake client、固定响应字节和
可控的 `threading.Event`，不要依赖局域网设备在线。

至少覆盖：

- Config 注册和 Draccus 解码；
- 必填字段和全部边界验证；
- semantic feature 与硬件通道的可配置映射；
- 协议请求字节和正常响应解码；
- transaction ID、设备 ID、功能码、长度、异常响应和半包；
- 有符号数、字节序、word swap、小数位和 SI 单位转换；
- timestamp 在响应接收后生成；
- sequence 单调递增；
- `read()`、`async_read()`、`read_latest()` 的区别；
- 连接、重复连接、断开、重复断开；
- 初始连接失败后的清理；
- 读取失败后的重连与设备设置刷新；
- disconnect 能唤醒阻塞读取并打断网络 I/O；
- 重连与 disconnect 竞争时不会泄漏连接；
- factory 构造但不连接。

测试文件使用：

```text
tests/sensors/test_<device_name>.py
```

运行示例：

```bash
uv run pytest tests/sensors/test_<device_name>.py -svv
uv run ruff check src/lerobot/sensors/<device_name> tests/sensors/test_<device_name>.py
uv run ruff format --check src/lerobot/sensors/<device_name> tests/sensors/test_<device_name>.py
```

## 12. 真机 Smoke Test

真实硬件检查放在：

```text
examples/<device_name>/<device_name>_hardware_smoke_test.py
```

smoke test 应做到：

- 通过命令行接收地址、端口、设备 ID、样本数和超时；
- 明确打印测试是否只读；
- 打印设备实际单位、采样率和关键格式设置；
- 连续读取多个样本；
- 检查 `is_valid`、sequence、timestamp 和 feature 键；
- 输出已经转换到 SI 的值；
- 使用 `finally` 可靠断开；
- 失败时返回非零退出码并给出可操作的连接排查建议。

Windows 双击启动器可以放在同一目录，命名为：

```text
run_<device_name>_hardware_smoke_test.bat
```

不要把临时 ZIP、日志、抓包、原始数据或 `__pycache__` 放在 `src/`。

## 13. 与 Robot 和 Dataset 的边界

完成 backend 不代表已经完成 Robot 集成。Sensor backend 中不要：

- import SO101、B601 或其他具体 Robot；
- 拼接 Robot 的关节状态；
- 实现 PID、导纳、阻抗或力控；
- 修改 Policy 输入；
- 写 Dataset；
- 根据某个实验决定 action。

Robot 集成使用 `SensorizedRobot` 组合并持有 `sensors`。Sidecar v2 是一次正式 Dataset
schema breaking change：

```text
v1  observation.state = float32[D_robot + selected_force]
v2  observation.state = float32[D_robot]
    observation.tactile = float32[2] = [force_left, force_right]
```

wrapper 的 `get_observation()` 先固定 `frame_anchor_ns`，然后对每个 Sensor 只调用一次
`read_latest_before()`。同一个局部 `SensorSample` 同时提供
`sensor.<instance>.<feature_path>` qualified values 和 capture metadata 中的
sequence/timestamps/status。后续 processor、routing、frame packing、Sync 写入不得再次读取
Sensor；并发到达的新 Raw sample 只能供下一帧选择。

schema routing 根据 wrapper 声明的 source 删除 state 中相应项，并创建独立
`observation.tactile`，不能靠 `sensor.*` 前缀猜测。routing helper 本身支持任意宽度；正式
Sidecar v2 current-force profile 另行要求全局恰好两路、相对路径依次为
`left.normal_force` 和 `right.normal_force`、unit 为 `N`。全部 stream 都是 raw-only 时不创建
tactile。更宽的触觉、六轴 F/T 或多指 current view 需要未来 schema/profile version。

processor 不得 rename、drop 或数值变换本帧 selected source；共享校验也禁止通过 rename map
把 tactile 移入 state，或把其他 feature 移入 tactile。state 和 tactile 不得重复保存 force。
baseline policy 的自动 input feature 推导默认排除 tactile；只有未来 policy/config 显式声明
tactile 并实现 encoder/fusion 后才可使用。当前阶段不修改 ACT、π0、π0.5、SmolVLA 或 OpenPI
模型逻辑。

采集由独立 Recorder 使用有界 Raw/Sync spool 完成；默认每 4096 行或 0.5 秒 flush，运行指标不进入
稳定 manifest。required raw-only 流也参加启动屏障，optional 不阻塞。停止时拒绝新 Sync、unsubscribe、
排空/flush/close、join 并验证清理；join 超时保留线程、文件与 lease，禁止复用。

恢复能力限定为 **process-crash recoverable + replayable on-disk state**，不承诺掉电持久性。
Reader 的 fast/full 验证都严格只读；恢复交给持写锁的 Writer 或 `TransactionRecoveryManager`。
恢复仅发现 active pointer 和未清理 staging，不扫描历史 journal；早期原型格式不受支持且不提供迁移。
Reader 可只读 v1/v2 Raw、Sync 与 window，但不会从历史 v1 state 在线拆力或合成 tactile；Writer 仅创建或
续录 v2 root，v1 数据使用新 root，不做原地迁移。带 Sensor 的 resume 必须已有 v2 manifest，不能从旧主
Dataset 中途开始 Sidecar；已有 Sidecar 的 root 也不能在未配置 Sensor 时继续追加主帧。Sidecar v2 只升级逻辑 manifest/frame-view contract，
storage layout、Raw/Sync Arrow schema、transaction journal 与 main evidence 继续使用各自 v1 格式。
Hub subset、Windows spawn、批量窗口、64 MiB worker-local 缓存与诊断使用方式见
[`SENSOR_DATASET_FORMAT.md`](./SENSOR_DATASET_FORMAT.md)。

## 14. 当前 X518 参考实例

X518 是完整 backend 示例，但不是所有传感器的强制模板：

```text
src/lerobot/sensors/x518/configuration_x518.py  # 网络参数、通道映射、配置验证
src/lerobot/sensors/x518/protocol.py            # 私有只读 Modbus-TCP 协议
src/lerobot/sensors/x518/sensor_x518.py         # 生命周期、采样、重连、SI 输出
tests/sensors/test_x518.py                       # 无硬件单元测试
examples/x518/x518_hardware_smoke_test.py        # 真机只读检查
```

最小使用方式：

```python
from lerobot.sensors.x518 import X518ChannelConfig, X518Sensor, X518SensorConfig

config = X518SensorConfig(
    host="192.168.1.100",
    frame_features=["left.normal_force", "right.normal_force"],
    channels={
        "left.normal_force": X518ChannelConfig(channel=1),
        "right.normal_force": X518ChannelConfig(channel=2),
    },
)

with X518Sensor(config) as sensor:
    sample = sensor.async_read(timeout_ms=1000)
    latest = sensor.read_latest(max_age_ms=100)
```

X518 的 Modbus 地址、双通道限制、单位码和 word swap 只属于 X518，不能提升为通用 Sensor 规则。

## 15. 完成检查清单

### 架构

- [ ] 已阅读 `SENSOR_FRAMEWORK.md`，没有修改锁定命名或分层。
- [ ] backend 不依赖 Robot、Controller、Policy 或 Dataset。
- [ ] 没有创建无实际用途的抽象或文件。
- [ ] 构造对象时不会访问硬件。

### Config 与 feature

- [ ] Config 使用 `SensorConfig.register_subclass()` 注册。
- [ ] `sample_rate_hz` 使用公共字段。
- [ ] `frame_features` 使用 `None` / ordered list / `[]` 表达 all / selected / raw-only；没有继续使用 v1 `state_features`。
- [ ] 配置在连接前完成类型、范围和有限值验证。
- [ ] 硬件通道到 semantic feature 的映射可配置。
- [ ] feature 名称不含型号、通道或单位。
- [ ] 输出已经转换为 SI，`features` 与 `values` 键完全一致。

### 采样与生命周期

- [ ] 每个样本包含 `timestamp_ns`、`arrival_timestamp_ns`、`sequence`、`values`、`is_valid`。
- [ ] invalid acquisition 也递增 framework sequence，并保留 status/error。
- [ ] current/window 同时检查 timestamp 与 arrival timestamp 的因果性。
- [ ] 每帧每个 Sensor 只选择一次；current values 和 Sync metadata 复用同一 `SensorSample`。
- [ ] 读取与 History 接口的消费、等待、区间和因果语义正确。
- [ ] `connect()` 启动后台资源，失败时完整回滚。
- [ ] `disconnect()` 能停止线程、唤醒读取者并关闭 I/O。
- [ ] 如实现重连，重连后会刷新设备设置。
- [ ] Recorder 使用独立 subscriber queue，overflow 会使 episode 失败且清理 worker/subscriber。

### 验证

- [ ] 单元测试不需要真实硬件。
- [ ] 已覆盖协议异常、单位转换、读取语义和资源竞争。
- [ ] 真机检查位于 `examples/<device_name>/` 且默认采用安全操作。
- [ ] 定向 pytest、Ruff、格式检查和 `git diff --check` 全部通过。
- [ ] `git status --short` 中没有缓存、日志、抓包或无关修改。
