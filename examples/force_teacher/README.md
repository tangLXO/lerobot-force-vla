# Teacher 50 Hz 采集版

本分支从 `d82f5d52` 重整，当前交付为六轴遥操＋夹爪 Teacher 的双摄、动作及高频力采集、后台保存和续录。操作及数据契约见 [DEMONSTRATION_RECORDING.md](DEMONSTRATION_RECORDING.md)。没有迁入 v4/v5 实验选择、pilot、模型窗口准入或发布审核体系。

调参入口：`uv run --no-sync python examples/force_teacher/b601_tuner.py`，或双击同目录的 `run_b601_tuner.bat`。具体流程见 [TUNING.md](TUNING.md)。

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

`fake_teacher.yaml` 的数值仅用于仿真。`b601_teacher.template.yaml` 为禁止运动的硬件模板。正常全臂遥操使用旧界面的 operator 入口及本机参数、校准和现场确认，软件测试不替代实机验收。
