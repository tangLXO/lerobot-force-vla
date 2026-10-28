# Teacher 50 Hz 采集版

此分支从 `d82f5d52` 建立，保留旧 Teacher、界面、六臂 MIT 遥操、夹爪 FORCE_POS＋I-P、人工张开接管、重新接触、主动释放、调参和冻结/恢复。控制与主数据统一为 **50 Hz**，Owner 沿用绝对节拍调度。原始力独立请求 **400 Hz**，实际频率另存。此轮只负责采集、保存、续录及文件核验。

## 启动与录制

在本分支工作目录执行，避免另一 checkout 的 editable 安装被误用：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
uv run --no-sync python examples/force_teacher/b601_tuner.py
```

也可使用 `run_b601_tuner.bat`。默认参数文件为本机 `b601_capture_parameters.json`，相机表单优先读取 `b601_camera_bindings.json`；两者不入库。可用 `--settings-file` 选择参数、`--recording-config` 预填独立录制 JSON。这些选项不会连接或使能。原有参数文件不会被覆盖。

1. 停止并断开旧会话，核对当前参数、校准、物体和相机视野。在“示范录制”页设置独立新目录，填写任务、物体和接触条件，勾选准备录制。FPS 固定 50，相机至少 50 FPS。默认 `h264_encoder=nvenc`，软件编码需显式选 `software`。
2. 沿用原界面取得新鲜只读失能参考，操作者现场确认支撑和映射，再点击“正常全臂遥操”。开始录制只提交采集请求，不使能机械臂。
3. 点击“开始录制”，等状态为 `recording` 后完成接近、接触抓取、保力及人工张开。每段人工任务结果默认 `unknown`，由操作者确认 `success` 或 `failure`。
4. 点击“结束并保存”。后台完成视频、主表、Raw、Sync 与 Teacher 附件核验，期间 Teacher 继续运行。状态为 `saved` 且文件完整后，可以开始下一段。
5. 连续完成三段后，停止控制并等候失能、串口关闭；退出界面。再次打开，选择相同目录和配置并勾选“续录”，追加下一段。续录检查已有全部完整证据及数据契约，不覆盖旧段。

勾选准备录制后，启动会话先完成相机连接、Dataset/Sidecar 初始化、编码器及预览预热，再启动主臂和力采样，最后打开电机 Owner。准备失败或期间停止时不会使能；进入遥操后点击“开始录制”复用已准备资源。首次准备可能需要数秒，等待期间不要重复启动。若准备已创建空 Dataset 后启动失败，重试同一目录也需勾选“续录”，接受已有内容核验。

启动顺序对照官方 LeRobot [`d3793b7` 的录制入口](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/scripts/lerobot_record.py#L412)：先创建或续接 Dataset，再连接 teleoperator，最后连接 robot。Teacher 将双摄及编码预热也放在主臂采样前，避免首次初始化占用主臂读取期限。官方 [`B601.disconnect`](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L398) 的 disable、clear_error、close 可作为故障维护参考；本版仍保留原始故障证据及独立只读核验，不能将清错命令返回当成失能确认。

目标力以当前参数和 Owner 生效回执为准；本机既有 2 N 参数无需为采集修改。安全保护、反馈新鲜度、Raw 超力及 Stop/Esc 保持原流程。普通节拍波动保存为统计，不单独否定文件保存；安全故障与数据丢失仍会中断采集。

准备阶段失败也保存 Owner 是否尝试启动及串口关闭事实，不声称电机已失能。历史关闭回执缺失时，空闲面板仍可请求全臂只读复核：查询传输先取得现有跨进程独占锁，再打开串口；占用或锁不可访问即失败。只有同一端口、同一校准下新观察到七轴全部 DISABLED 且查询串口关闭，才解锁后续操作。原失败记录保留，机械确认仍由现场操作者完成。

全臂会话在打开串口前建立只读、无缓冲的标定文件句柄。每次正常写入仍检查路径身份、重新读取字节、核对 SHA256；同大小且恢复原时间戳的修改也会阻止下一轴动作。会话关闭同时清理该句柄，关闭失败仍保留资源所有权。此项从近期实现单独迁入，不启用冻结标定或新的连续发送协议，不改变 20 ms 发送预算。它降低文件打开和重复参数验证开销，不能单凭软件测试宣称实机发送超时已解决。

逐轴发送中途停止时，整帧动作保持 `unknown`；成功的前缀轴、失败轴及未尝试的后续轴分别保留在回执中。夹爪尚未发送不等于整体完全未尝试，也不能让该帧成为完整动作或完整采集段。

录制进程在主臂、力和电机连接前启动并预热双摄。控制进程只转交原始 Sensor 样本和完整控制拍；图像、编码、Dataset、Raw/Sync 写盘与文件校验都在录制进程。Raw 保留原序号和两个时间戳，传输不重新发布或编号；结束边界必须证明控制拍及原始力尾部全部送达。Sensor 序号租约保留至写盘完成，队列溢出、进程退出或停止都扣留完整标记。此功能单独沿用现有 Raw 输入契约，不引入模型读取或实验分型。

## 相机与视频

每个命名 OpenCV 相机必须提供真实 `device_path`。连接时按该路径重新解析当前索引，保存实际设备路径、后端、尺寸和帧率；找不到路径或重复绑定时拒绝连接。`index_or_path` 仅为配置兼容字段，不能用旧编号代替设备身份。设备重插后仍按路径核对；换 USB 插口后需要重新确认绑定与视野。

建议本机双摄 640×480、MJPG、DSHOW、50 FPS。编码使用官方 Dataset/视频组件，NVENC 配置为 H.264、p4、GOP 2、无 B 帧。NVENC 不可用时报告错误，不自动换编码器。续录要求编码配置一致。编码线程失败、帧数缺失或视频不可解码均不能产生完整标记。

## 保存内容

根目录 `meta/teacher_capture.json` 保存 `teacher_capture_contract_v1`，每段使用 `teacher_capture_episode_v1`。Sidecar 逻辑版本仍为 **v2**，Raw/Sync Arrow、存储布局和事务日志仍为 **v1**。

| 内容       | 字段与含义                                                                                                                   |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 关节状态   | `observation.state`，7 轴，单位 °，仅 Robot 状态                                                                             |
| 实际动作   | `action`，同一控制拍全部成功提交的 7 轴目标，单位 °；不表示硬件 ACK                                                          |
| 左右当前力 | `observation.tactile`，`[left.normal_force, right.normal_force]`，N，与 Sync 引用的 Raw 样本一致                             |
| 有效目标力 | `observation.force_target`，左右平均力的有效目标，N；Dataset 特征 shape `[1]`，Parquet 存为标量                              |
| 图像       | 两路主视频；帧附件保存因果选择图像的采集/到达时间戳                                                                          |
| 任务       | 官方 task/task_index 映射，人工说明来自录制表单                                                                              |
| 全部原始力 | 独立有界订阅写入 Raw，保留接收到的有效和无效样本、序号、时间戳、有效状态；溢出或序号缺失拒绝提交                             |
| 同步       | Sidecar Sync 与 `teacher/<uid>/frames.jsonl` 关联主帧、控制 tick、Raw 序号、图像时间                                         |
| 控制与标注 | `teacher/<uid>/control.jsonl` 保存完整控制尝试；帧附件保存 Teacher 阶段、动作有效性和目标力                                  |
| 来源与结果 | `meta/teacher_episodes/<uid>.json` 保存配置、代码哈希、UTC 开始时间、实际绑定、频率、保存前后 Owner 进度、文件哈希及人工结果 |

关节顺序固定为 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper`。历史仅留在 Raw；不向 `observation.state` 复制力，不增加持久化窗口或训练接口。

真实时序使用整数 `perf_counter_ns`。图像时间是主机完成读取时间，力时间是主机收到完整响应时间，不是曝光/ADC 硬件时间。到达时间另存。官方主表 `timestamp=frame_index/50` 用于播放，不能替代真实时间戳；跨进程比较必须结合每段来源和 UTC 时间。

## 结束、中断与文件核验

| 操作              | 保存行为                           | 控制行为                         |
| ----------------- | ---------------------------------- | -------------------------------- |
| 结束并保存        | 事务提交并独立核验，之后才能标完整 | Teacher 继续                     |
| 丢弃并重录        | 保留未完成证据，重新开始           | 不追加机械动作                   |
| 释放并结束采集    | 正常释放完成后保存                 | 沿原主动释放流程张开、失能、关闭 |
| Stop / Esc / 故障 | 保留中断或失败标记，不能冒充完整段 | 优先停止                         |

文件完整、Teacher 阶段覆盖和人工任务结果分别保存。停止期间即使底层已有部分主数据提交，Teacher 完整标记仍被扣留，续录会拒绝这类不一致根目录；不得手改标记绕过检查。恢复只由持有 Writer 锁的事务恢复器处理当前未清理事务，验证器严格只读。保证范围是进程崩溃恢复，不承诺掉电持久性。

停止并关闭录制会话后运行：

```powershell
uv run --no-sync python examples/force_teacher/validate_demonstration.py "outputs/teacher_capture_50hz"
```

默认检查主表、动作证据、Raw/Sync 引用、当前力、目标力、因果时间和视频逐帧解码。`--skip-video-decode` 只用于诊断，不能作为视频验收。

## 软件与只读验证

```powershell
uv run --no-sync python examples/force_teacher/demonstration_smoke_test.py --root outputs/capture_software --episodes 3 --duration-s 3 --encoder nvenc
uv run --no-sync python examples/force_teacher/demonstration_smoke_test.py --root outputs/capture_software --episodes 1 --duration-s 3 --encoder nvenc --resume
uv run --no-sync pytest tests/force_control -q
```

软件 smoke 使用假控制和假图像，但实际写入官方 Dataset、视频、Raw/Sync 并解码核验，不能当作物理抓取验收。`recording_readonly_hardware_smoke_test.py --help` 提供查询专用七轴/双摄/力联合采集，不发送运动命令，也不产生有效 Teacher 动作示范。
