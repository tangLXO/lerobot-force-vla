# Teacher 50 Hz 采集版

当前采集功能已整理交付到 `force-vla` 主线，开发基线为 `d82f5d52`。交付包含六轴遥操＋夹爪 Teacher 的双摄、动作及高频力采集、后台保存、续录与只读完整核验。操作及数据契约见 [DEMONSTRATION_RECORDING.md](DEMONSTRATION_RECORDING.md)。没有迁入 v4/v5 实验选择、pilot、模型窗口准入或发布审核体系。

日常统一使用根目录 `run_force_pos_teacher.bat` 打开 Qt 四页工作台；本机桌面的“Force-VLA 数据采集”快捷方式指向同一脚本。安装和使用见 [DESKTOP_UI.md](DESKTOP_UI.md)。需要原 Tk 面板时，在同一个入口后添加 `--legacy-ui`，参数说明见 [TUNING.md](TUNING.md)。

Qt 和 Tk 日常七轴入口默认使用 `cached` 异步反馈读取、50 Hz 控制和官方参考的 **120 µs（0.12 ms）最小发送间隔**。实际发送间隔还包含串口与调度耗时。2026-10-31 的 5 段实机录制共 32,664 帧，完整存储与双摄解码核验通过，录制中未发现通信硬超时或发送失败；本次仍观察到保力波动、腕部画质和长等待段问题，文件完整不等于示范质量全部合格，详见 [交付记录](DELIVERY_SCOPE.md)。

间隔优先级：显式 `--command-tx-gap-us` → 环境变量 `MOTORBRIDGE_TX_GAP_US` → 默认 `120`。例如 PowerShell 中 `$env:MOTORBRIDGE_TX_GAP_US = "120"`；显式 `--command-tx-gap-us 1000` 可覆盖环境变量。非法、负数或非有限值会在连接前报错，显式 `0` 表示不额外限速。首页和启动输出显示生效值与来源。单夹爪定位继续使用 sequential，此异步间隔不适用。

已有数据根会校验通信策略：原来以 1000 µs 建立的目录，需显式指定 `--command-tx-gap-us 1000` 才能按原策略续录；采用 120 µs 时请使用新目录。普通结束仍使用“结束并保存”，保存不追加运动。

同一界面支持独立调参、Teacher 主从夹爪与正常六臂 MIT 遥操：低力时跟随 102 主爪，接触后保持目标力，主爪张开越界后接回。当前采集默认参数为本机独立文件 `b601_capture_parameters.json`。提交整理与历史回退方式见 [DELIVERY_SCOPE.md](DELIVERY_SCOPE.md)。

夹爪使用 B601 ID7 的 FORCE_POS 位置执行方式，外环使用 X518 双侧当前力均值进行增量 I-P 控制：

`delta = Ki * dt * (target_N - mean_N) - Kp * (mean_N - previous_mean_N)`。

Teacher 仅产生候选；Safety 限制候选；唯一 owner 完成硬件下发，成功后才提交控制历史。左右原始力分别进行超力保护。软件提交不代表硬件 ACK 或实际到位。

## 离线验证

```powershell
uv run --no-sync python -m lerobot.scripts.lerobot_force_teacher --config_path examples/force_teacher/fake_teacher.yaml
uv run --no-sync python -m lerobot.scripts.lerobot_force_teacher --config_path examples/force_teacher/fake_hybrid.yaml
uv run --no-sync pytest tests/force_control -q
```

`fake_teacher.yaml` 的数值仅用于仿真。`b601_teacher.template.yaml` 为禁止运动的硬件模板。正常全臂遥操使用工作台的显式启动操作及本机参数、校准和现场确认，软件测试不替代实机验收。
