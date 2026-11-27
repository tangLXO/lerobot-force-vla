# Force-VLA 当前开发计划

2026 年 11 月 27 日更新数据位置和读取入口。近期用已有 B601 示范检验服务器上的 π 模型能否区分保力微调与主动放手。研究问题与已定分工见[研究总纲](RESEARCH_CHARTER.md)。

## 服务器所需的数据

[开发数据索引](examples/force_teacher/teacher_pilot_index.json)列出了 10 个完整 run 的原始指令、物体、目标力、帧数、动作来源区间、释放帧和双摄视频路径。每个 run 有独立的数据根，下属 episode 编号都是 0。使用 `run_uid` 和 `episode_uid` 识别示范，按完整 run 划分训练与开发验证。

本地和服务器共用同一份 Git 索引。两端的数据目录均为 `~/work/force-vla/data/b601_demos/`，Windows 上的 `~` 为用户主目录。将每条 `episodes[].root` 接在这个目录下，再从该条示范的根目录解析 `manifest` 和 `videos`。例如第一条位于 `b601_demos/teacher_bottle_transfer_20261119/`。原始视频、Parquet、Teacher 日志和传感器 Sidecar 均按原样保存。索引中的 SHA-256 对应控制记录和帧记录。`excluded_manifests` 保留历史采集位置，仅用于追溯。

在仓库根目录使用正式的 `lerobot-train`。为现有的 π0 训练命令添加以下数据参数。指定索引时，入口默认读取上述数据目录；其他部署位置可用 `--dataset.root` 指定包含这 10 条示范的目录。

```text
--dataset.repo_id=local/b601_demos
--dataset.teacher_pilot_index=examples/force_teacher/teacher_pilot_index.json
--dataset.video_backend=pyav
```

默认使用全部 10 条。复现 8 条开发训练、2 条验证的读回检查时，另传 `--dataset.held_out_run_uids='["7816e7b8-4bfe-4650-ac69-aa9c9383cb0c","9d6d8602-f8c4-4377-bbde-326fb7bc955a"]'`。每条示范保持独立的动作窗口、填充标记和模式标签，训练与验证共享训练侧的归一化统计。

## 最近的算法验证

1. **准备模型输入与标签。** 按决策时刻对齐原句、双摄、从臂状态和届时已到达的双指触觉。保留从臂动作、跟随／保力状态、重置事件、主动释放与段尾填充标记。大脑从原句获得指令力值，录制的正确目标用于输出监督和评分。[数据字段约定](SENSOR_DATASET_FORMAT.md)
2. **用已有动作验证大脑输出。** 选择一个候选基座做小规模试验，在留出的整段上检查保力动作预测和释放时机。可将保力段的预测与“沿用上一夹爪位置”比较，同时检查模型是否把微开误判为松手。是否把预测轨迹用于控制、怎样与小脑配合，按验证结果决定。
3. **只用训练数据拟合归一化。** 正式训练入口可按索引读入独立数据根、按 run 留出并按训练侧汇总统计。批次包含双指读数、七轴动作、逐步跟随／保力标签和填充标记，模型的输入、输出与损失由相应策略配置决定。[训练数据工厂](src/lerobot/datasets/factory.py)

## 已归档的排期

早期 π0 有／无触觉 A/B、24 或 36 段采集配额、16 格增益调参、重复 5 N 持瓶、本地 RTX 5070 测速及日报安排的同路线动态对照均属历史。已完成的结果按版本保留。正式数据与实物对照等算法问题明确后统一安排。
