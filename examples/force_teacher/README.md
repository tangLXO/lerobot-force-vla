# FORCE_POS 力控

夹爪使用 B601 ID7 的 FORCE_POS 位置执行方式，外环使用 X518 双侧当前力均值进行增量 I-P 控制：

`delta = Ki * dt * (target_N - mean_N) - Kp * (mean_N - previous_mean_N)`。

Teacher 仅产生候选；Safety 限制候选；唯一 owner 完成硬件下发，成功后才提交控制历史。左右原始力分别进行超力保护。软件提交不代表硬件 ACK 或实际到位。

## 离线验证

```powershell
uv run --no-sync python -m lerobot.scripts.lerobot_force_teacher --config_path examples/force_teacher/fake_teacher.yaml
uv run --no-sync pytest tests/force_control -q
```

`fake_teacher.yaml` 的数值仅用于仿真。`b601_teacher.template.yaml` 为禁止运动的硬件模板，必须由操作者填写校准、方向、目标力、增益和限制。物理控制仅开放经校准的 ID7 FORCE_POS 范围；全臂控制仍保留独立验收门禁。
