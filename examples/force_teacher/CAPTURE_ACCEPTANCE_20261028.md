# 采集版现场记录：2026-10-28

前三段采集代码版本 `ec613e68`，重开后的第四段使用 `c34d0897`，分支 `codex/teacher-capture`，起点 `d82f5d52`。原 checkout、参数、校准和此前失败数据保留。本轮交付已验证的空载采集、连续保存和重开续录；操作者选择暂跳过实物接触验收，人工任务结果保持 `unknown`。

## 已完成的现场检查

操作者现场启动 Teacher；七轴使能反馈通过，夹爪处于 FOLLOW。双摄按稳定设备路径连接，640×480、50 Hz、MJPG，保存使用 NVENC。当前目标力 2 N。控制、发送及反馈保护预算未放宽。

在同一控制会话内连续录制并保存三段到 `outputs/teacher_capture_hardware_20261028_03/dataset`，正常停止、完全退出并重开面板后，向同一目录追加第四段：

| 段号      | 主帧 | Raw 力样本 | 实测主频 Hz | 实测力频 Hz | 保存用时 s | 保存期间提交控制拍 |
| --------- | ---- | ---------- | ----------- | ----------- | ---------- | ------------------ |
| 0         | 2198 | 17591      | 49.9998     | 399.8822    | 7.1929     | 360                |
| 1         | 2194 | 17635      | 50.0006     | 399.9000    | 7.0913     | 354                |
| 2         | 2071 | 16638      | 50.0002     | 400.0023    | 6.7740     | 339                |
| 3（重开） | 3364 | 26904      | 49.9993     | 399.6897    | 10.7715    | 539                |

四次保存期间全部控制拍为 FOLLOW，动作提交成功、控制拍无缺口、无故障；最大观测控制间隔分别为 21.2952、21.2452、21.1146、25.5930 ms。这是本次观测，普通频率波动与文件完整性仍分开判断。

停止后独立只读校验通过：

- 官方主表、Raw、Sync 和 Teacher 附件可读取；两路视频按每段范围逐帧解码通过。
- 合计 9827 主帧、78768 Raw 样本；Raw 无序号缺口，本次无无效样本，全部 Sync 引用能找到对应因果样本。
- 主表实际动作与七轴提交回执一致，状态、当前力及图像时间关联通过现有验证器。
- 校验前后数据目录内容、大小及修改时间不变。
- 两次控制会话正常停止收据均无错误，观察到七轴 DISABLED，串口关闭。第四段结束后控制和采集子进程均已退出。

完整证据：

- `outputs/teacher_capture_hardware_20261028_03/audit_three_episodes.json`
- `outputs/force_teacher_tuning/81e9d57bf6c74535bfe740ef734e8b5b/run/result.json`
- `outputs/capture_three_episodes_audit.log`
- `outputs/teacher_capture_hardware_20261028_03/audit_resumed_fourth.json`（包含全部四段）
- `outputs/force_teacher_tuning/8a4fdfd6f27a472da5e6a85cb19d1f9a/run/result.json`
- `outputs/capture_resumed_fourth_audit.log`

## 本次启动修复

录制桥接在 Owner 启动前读取首帧，真实 `OwnerService.snapshot()` 因尚无控制拍而抛错。增加显式 `allow_empty=True`，仅预热遥测使用；开始录制仍要求有效首帧，Owner 的真实故障不会被隐藏。测试改用真实 Owner 启动前状态，相关测试 37 项通过，针对变更的 pre-commit 通过。

真实 Owner API 与实际双摄、主臂和力采样的无使能预热复核通过，资源正常关闭。证据为 `process_preparation_real_owner_hardware.json`，位于上述现场输出目录。

## 续录准备故障及修复

首次重开续录在启动准备阶段报 `Capture IPC overflow`，未追加第四段。该失败记录保留于 `outputs/force_teacher_tuning/d5aecaf0384d45329706f53171e45ea8/run/result.json`。随后独立只读复核再次观察到七轴 DISABLED。

已有三段的视频校验使录制子进程的准备时间延长；准备时子进程尚不接收输入，但父进程持续发送空 Owner 状态，填满了有界队列。`c34d0897` 将周期状态发送延后至子进程准备完成，不扩大队列、不丢弃原始力或完整控制拍。同时保留启动前 Owner 未尝试连接的收据，避免延迟清理用空 runtime 收据覆盖已有关闭事实。

加入小容量队列和延迟准备的 Windows spawn 回归，以及启动前延迟清理的收据回归；相关测试 39 项及 pre-commit 通过。使用实际三段数据与真实设备再次执行无使能准备，32.031 秒后返回 ready，识别已有 3 段，双摄、主臂、力采样及关闭均正常。证据：`outputs/teacher_capture_hardware_20261028_03/process_preparation_resume_ipc_fix.json`。

随后由操作者启动新会话，完成上述第四段采集。第四段 UID 为 `dced6bfe-bc59-42fc-9d14-8f72dca6f61c`，文件完整且独立只读校验通过，续录准备队列溢出未再出现。

## 本轮暂跳过的现场验收

- 操作者尚未确认四段人工任务结果，均保留 `unknown`；文件完整不等同人工任务成功。
- 操作者明确要求先跳过实物测试。本次没有接触保力、人工张开接管、重新接触或主动释放的实物验收证据，不将空载 FOLLOW 当作这些行为通过。

此前真实保存故障目录 `outputs/teacher_capture_hardware_20261028_02` 保留，Teacher 完整标记为 false，续录拒绝不一致根目录；其只读检查在 `save_fault_integrity.json`。进程退出、保存中停止和缺失数据的自动化故障隔离测试通过，不能代替尚未完成的现场行为验收。
