# Teacher 七轴异步通信

基线：`codex/teacher-capture` 的 `15be754e`。控制和录制保持 50 Hz。默认仍为 `sequential`；同机对照、录制完整性和接触行为通过后才考虑切换默认。

2026-10-29 验证：软件回归通过，异步实机两次出现夹爪反馈超时，采用条件未通过，继续保留串行默认。完整结果和原始证据索引见 [验证记录](PIPELINED_VALIDATION_20261029.md)。

后续 [末帧反馈诊断](GRIPPER_FEEDBACK_DIAGNOSIS_20261029.md) 在失能查询中复现了随发送顺序转移的末轴缺帧。1000 µs 显式发送间隔已通过只读对照，运动与录制验证另行记录；这不是调整通用默认值或放宽反馈期限的依据。

## 启动

从本采集工作树运行：

```powershell
.\run_force_pos_teacher.bat --command-feedback-mode sequential
.\run_force_pos_teacher.bat --command-feedback-mode pipelined --command-tx-gap-us 120
```

每次只开一个控制会话。以上命令打开面板；连接和使能仍使用既有人工操作流程。模式在会话启动时固定，不能运行中切换，不自动回退。配置文件等价字段是 `b601.control.command_feedback_mode` 和 `b601.control.command_tx_gap_us`。夹爪单独入口和诊断 capability 不支持异步模式。

根启动器显式设置本工作树的 `src` 与根目录为 `PYTHONPATH`，避免共享 `.venv` 导入桌面分支。预检和控制子进程核对实际模块路径；录制子进程核对父进程绑定的源码文件。直接运行脚本或测试时同样设置：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
uv run --no-sync pytest tests/force_control -q
```

## 通信约束

启动、模式设置、预加载和使能沿用严格串行流程。正常控制每轴写入时仍检查停止、力、主臂和校准绑定。异步模式不等待该轴反馈才提交下一轴；在发送间隙和周期等待内，由同一 Owner 有界接收。每组截止时间从发送开始计算，不在下一拍重置。

最多一组未消费命令反馈。按电机 ID 处理分片、合并和乱序；完整七轴通过校验后形成不可变快照，下一拍消费一次。缺帧、重复、未知 ID、损坏、溢出、过期或驱动异常均故障停止。协议没有请求编号，日志中的 generation 仅为主机分组，`hardware_ack` 始终为 `unknown`。

导出的每组诊断只保存一次相同的 `session_signature`，轴命令和接收上下文通过 `session_signature_ref=group.session_signature` 引用；若某项签名不同则原样保留。运行时仍逐帧核验完整上下文。这样避免诊断复制重复信息挤占 Owner 收包时间；原始字节、真实接收时间、序号和写入结果全部保留。

发送回执保留已提交、写入结果未知、未尝试。故障仍尝试七轴失能；迟到控制帧只作历史证据，失能状态通过后续独立查询取得。关闭时归档未完成组、接收残片和失能结果。

保存时，Owner 等待当前组反馈完成或故障，再发布通信尾行并冻结录制订阅。尾行不产生主数据帧、不提交动作。正常释放已关闭 Owner 时，使用已完成的释放边界。主帧保持“本拍决策观察 + 本拍实际提交动作”。双摄、原生 Raw 力、Sidecar v2 和事务格式不变。

## 对照与诊断

使用全新数据目录，交替 S1/P1/S2/P2/S3/P3，各 60 秒 FOLLOW，保持参数、设备、摄像头和录制配置相同。覆盖录制、保存及关闭后重开续录。保留每段配置、`source_binding`、运行 `control.jsonl`、录制验证结果与关闭回执。

面板操作可能让实际录制超过 60 秒。原始数据全部保留；报告默认从每段录制的首个正常 FOLLOW 提交起，固定取前 60 秒，不挑选最快窗口。报告同时保留原始时长、样本数和窗口是否完整。完整组反馈与故障证据仍扫描整个片段，避免丢掉窗口后的收尾。输入应是每段录制的 `teacher/<episode_uid>/control.jsonl`，而非包含启动等待的整场运行日志。

```powershell
uv run --no-sync python examples/force_teacher/compare_command_feedback.py `
  --run <S1-control.jsonl> --run <P1-control.jsonl> `
  --run <S2-control.jsonl> --run <P2-control.jsonl> `
  --run <S3-control.jsonl> --run <P3-control.jsonl> `
  --output <新的对照目录>/comparison.json
```

报告分别列出发送函数耗时、首轴提交窗口起点到末轴写完、整组反馈完成、拍内工作、两拍间接收工作、七轴最老反馈年龄、控制间隔和实际轴间写入间隔的平均值、P99（最近秩）和最大值。接收移到拍外不会被计为工作消失。每段分别报告，不把不同段混合成单一成绩；通信尾行不算控制拍。旧日志缺少新增字段时保持缺失，不用零填充。

操作者另做接触保力、张开接管、重新接触、冻结/恢复、释放。报告接触力绝对误差、超调、状态转换和停止结果；FOLLOW 时序不能替代接触实测。报告脚本不自动授予硬件通过资格。软件测试、合成录制、实机行为和性能对照分别记录。
