# 完整示范录制

面板包含正常“全臂遥操”配置、示范录制表单、episode 操作和独立后台录制适配器。**实机连接、对齐、启动、空载遥操和录制均由用户本人操作。** 正常入口为显式 `operator_teleoperation`，不要求先完成 micro 诊断，也不继承 micro 的 0.5°、1°/s、短时长或小累计行程。可选 `full_arm_validation` 保留独立诊断语义。软件检查、历史单轴记录均不能视为正式全臂验收或完成 10 条成功示范。

## 面板流程

```powershell
uv run --no-sync python examples/force_teacher/b601_tuner.py
```

1. 停止并断开旧会话，加载校准与夹爪参数。重新加载后必须重新取得新鲜只读参考。
2. 打开“示范录制”页，填写数据目录、数据集 ID、任务描述、物体、接触条件和相机 JSON。勾选“为下次控制会话准备示范录制”。默认主数据/视频 30 Hz，目标 10 条，有界队列 128 项。
3. 现有启动入口自动加载六臂官方 MIT 和当前夹爪设置，全臂页直接显示完整参数。无需手动选择或保存草稿。正常主从启动每次从当前表单生成 `operator_teleoperation` 配置，并自动离线校验；录制配置仍从示范录制页统一提供。
4. 用户按实际安装取得新鲜只读失能反馈、确认支撑与映射后，显式点击“正常全臂遥操（用户启动 / 使能）”。后端重新读取全部七轴及主臂的新鲜参考、核对失能状态并预载当前参考，完成准备后才使能；默认入口不另设官方没有的数值对齐/预载容差。开始录制按钮只提交采集请求，不会启动或使能控制。预检通过仅代表配置相容，`production_arm_authorized=False`、`hardware_ack=unknown` 仍如实保留。
5. 点击“开始录制”，等候 `recording` 收据及相机预览后完成演示。相机预览最多四路、每路最多 256 KiB、更新不超过 5 Hz；主视频和控制独立采集。
6. 放下物体并手动张开，选择人工任务结果，点击“结束并保存”。它只结束当前 episode，控制会话继续。只有主数据、视频、Sidecar 和 Teacher 附件均核验成功，才显示文件完整。
7. 需要重做当前条目时点击“丢弃并重录”。当前条目保留不完整标记与可追溯证据，随后重新开始；不能把丢弃记录计入成功数量。
8. 下一条仍在同一控制会话内显式开始。每条新 episode 的人工结果重新置为“未确认”，文件完整与人工成功分别显示；可确认当前条目或更新最近已保存 episode 的人工结果。完成后停止会话并等候失能和关闭收据，再运行只读文件验证。

“保存参数”按钮将夹爪设置与全臂参数保存到同一调参文件，示范录制表单不写入该文件。录制配置会进入本次 reviewed 配置和 episode 来源记录。续录已有根目录时，需要显式勾选“续录”，已有 Dataset 与 Sidecar 必须通过相容性和完整性核验。

### 默认入口与高级配置

正常入口已经直接接入启动器，自动读取 `b601_ip_parameters.json`；六臂默认使用官方 Kp/Kd 和官方限位与校准交集，夹爪使用当前表单 FORCE_POS＋I-P 参数，也可加载已保存的全臂编辑值。官方未启用的额外限制显示“默认关闭”，不再把 null 当成待填草稿。六臂启动参考包络默认 q0±100°，可在全臂页显式编辑；夹爪独立沿用其工作目标范围。校准、新鲜度、原始力保护和停止流程保持。

下面的手动草稿生成器和外部配置选择仅用于高级编辑，不是默认启动步骤。

需要从本机已保存参数重新创建正常草稿时，运行 `build_operator_teleoperation_profile.py --help` 查看必填来源/输出路径；生成器只读取文件并创建新草稿和来源报告，不连接设备。历史 `a75f...` 20 秒配置属于独立受限诊断，不是日常正常全臂的推荐配置。

“全臂遥操”页显示本次请求值及来源。六个臂关节使用官方 MIT `send_mit(q, 0, Kp, Kd, 0)`，不需要 POS_VEL 速度。Kp 为 `[45,45,45,8,9,8]`，Kd 为 `[12,12,12,1,1,1]`；SDK 将前三轴 Kd=12 编码为 5，摘要同时显示请求与编码值。参数来自[官方 DM 配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/rebot_b601_follower/motor_family.py)，编码与[当前 SDK](https://github.com/motorbridge/motorbridge/blob/23c22998db072bc0f2ee841874220ac36844a550/motor_vendors/damiao/src/protocol.rs)一致。夹爪可保留来源中有效的 FORCE_POS 速度码和驱动参数，例如速度码 1000；速度码与 °/s 单位不同。仅用户主动填写 gripper °/s 时切换该输入方式，不能同时使用两种速度输入。

- 启动参考包络为用户已确认的 **q0 ±100° ∩ 校准/绝对 profile 范围**；不会自动走满，不能代替每拍 `max_relative_deg` 限制。
- 每拍 `max_relative_deg` 是可选的目标相对反馈幅度限制；六臂 MIT 按官方默认关闭，夹爪保留当前保存值。对齐、预载和命令间隔的额外数值门槛默认关闭；显式外部配置可保留用户自己的数值。
- 目标力来自配置自己的 `teacher.force_target_N`；当前来源若为 1 N 就显示 1 N，但正常 operator 不固定为 1 N。运行时“应用目标力（operator）”只发送明确 target 请求，生效值以 owner 回执为准；其他硬件参数停止并断开后修改，默认入口在下次启动时直接校验当前表单。
- 可选总时长默认禁用，`duration_s` 与 `Safety.max_run_time_s` 为 `None`；启用时同一选择同步二者。累计行程选项仅限制 **夹爪累计行程** `Safety.max_total_travel_deg`，不是全臂计程。正常模式不从旧诊断默认继承 20/30 s 或 2°。
- 主臂年龄与 TIMEOUT 同步到运行配置和底层协议字段；来源启动/写/全七轴故障清理预算、runtime 读取/节拍预算、反馈最大年龄及左右 raw 超力阈值均在摘要公开。全七轴故障清理预算不是运行总时长。校准、raw 超力、反馈过期及 Stop/Esc 不可禁用。
- `runtime.handover_tolerance_deg` 仅用于显式控制权返回/接管，不参与正常 hybrid 启动；未选择保持 `None`，不借用启动对齐容差。

显式外部 JSON 与当前校准绑定，预检固定文件 SHA256/mtime/大小；修改后需重新预检并取得新参考。默认内置配置在每次启动时从当前表单校验并写入本次 reviewed 记录，不复用旧草稿。

### 历史诊断通用模板

下面命令只创建新 JSON 草稿，不连接设备，不覆盖已有文件：

```powershell
uv run --no-sync python examples/force_teacher/build_full_arm_profile_template.py "outputs/full_arm_profile.draft.json"
```

模板的 `allow_motion`、`explicit_arm` 均为 `false`；实际目标力、增益、关节限位、速度、主臂映射、端口、通信预算和校准来源保持 `null`，X518 安装配置为空。它故意不能通过运行校验。将已验证的当前夹爪 Teacher/Safety/X518 配置逐项复制，再按实测校准填写六轴参数和完整主臂映射；不要复制测试夹具的合成数值。此历史诊断模板的六轴仍为 POS_VEL、夹爪为 FORCE_POS；正常 MIT 遥操使用上面的 operator 生成器。

未填完的草稿在“离线预检”中报告缺项；此通用模板没有正常 operator 能力，仍不能运行。模板中的只读设备规格来自当前 B601 固定 ID 映射，不替代实际接线、方向、限位及停止行为核验。正常入口已自动使用当前已保存参数与校准，不使用测试夹具或历史 micro 参数。

面板内全臂配置的 `recording` 必须保持 `null`，录制参数统一从“示范录制”页提供。配置文件带有第二份录制参数时会明确拒绝，避免表单与后端使用不同设置。完整全臂配置的 Teacher/Safety 也不会被“应用实时参数”按钮替换；需停止后修改该配置并重新预检。

### 可选诊断：短时受限联合验证

受限入口须有 typed `b601.validation: FullArmValidationConfig`，并明确配置 `startup_mode="prepare_disabled_full_arm_validation"`。能力绑定七关节 profile、当前校准、主臂端口/映射配置，以及六个关节已成功微动试验的结果、reviewed 配置、事件和源码快照证据。仅改布尔标志、选择普通配置或复制占位字段不能获得该能力。

只有该诊断路径的 capability 和 Teacher 须显式为 **1 N**，最长持续时间不得超过该 capability 限制（上限 30 秒）。这些是历史受限诊断的条件，不是正常 operator 默认值或先决条件。“全臂遥操”页需明确选择“micro 诊断（可选）”才审查诊断配置；主从对齐、新鲜度、命令间隔、原始超力、校准及停止保护持续检查。

录制帧率不能高于该会话的 `runtime.control_hz`；例如受限会话配置为 20 Hz 时，须在录制页显式填写不高于 20 的 FPS，不能直接沿用一般录制的默认 30 Hz。配置帧率和实测速率仍需分别核验。

离线预检固定配置文件的 SHA256、mtime 和大小，并对完整准入证据闭包（包括 events 与 source manifest）的 path/SHA256/大小/mtime/inode 生成独立摘要；启动前及子进程 review 会重新准入并比较。预检后修改文件或引用证据，即使仍逻辑合法，也需要停止、重新预检并获取新鲜参考。面板不会自动改写数值或追认新版本。

受限会话退出必须观察到**七关节失能且串口关闭**。仅 ID7 的只读反馈不能解除全臂退出未确认状态。`hardware_ack` 仍为 `unknown`；一次受限验证通过也不会解除普通生产全臂门禁，不能称为正式全臂验收。

## 相机配置

支持一至四路命名 OpenCV RGB 相机。下面示例使用 640×480 RGB、30 Hz、DSHOW 后端，索引仅为占位示例。每次使用前，先用采集后端按设备名称枚举并核对索引；当前安装中 `1080P USB Camera` 对应 front，`icspring camera` 对应 wrist。设备重插或换机后重新核对，不把索引当成固定设备身份。

```json
{
  "front": {
    "type": "opencv",
    "index_or_path": 1,
    "backend": 700,
    "fps": 30,
    "width": 640,
    "height": 480
  },
  "wrist": {
    "type": "opencv",
    "index_or_path": 0,
    "backend": 700,
    "fps": 30,
    "width": 640,
    "height": 480
  }
}
```

相机配置须有明确宽高，采集帧率不低于主数据帧率。后台仅选择观测时刻之前已可用且未过期的有效相机帧；预览不参与训练输入选择。打开面板、编辑 JSON 和离线预检均不连接相机。

新建示范数据集使用后台 H.264 流式编码，编码、磁盘保存及预览均不在硬件 owner 线程内。此设置仅属于新增示范录制适配器，通用 `lerobot-record` 的默认录制设置不受影响。

## 结束、释放与中断

| 操作              | episode                      | 控制与机械动作                     |
| ----------------- | ---------------------------- | ---------------------------------- |
| 结束并保存        | 保存当前条目并核验           | 控制继续，不追加动作               |
| 丢弃并重录        | 当前条目标记丢弃，再开始一条 | 不追加机械动作                     |
| 释放并结束采集    | 后端在正常受限释放后完成保存 | 执行已配置张开轨迹，失能并结束会话 |
| 停止 / Esc / 故障 | 当前条目标记中断或失败       | 优先停止，不等待正常释放           |

人工张开结束的有效 episode 不要求 `normal_release_completed=true`；这个字段仅描述是否观察到主动 RELEASE 状态机完成。保存按钮不代表任务成功，成功数量需人工确认。显示阶段拍数用于检查接近、接触交接、力控及释放/人工张开覆盖，不能据此声称抓取成功。

“结束并保存”直接使用当前下拉框的人工结果；若使用“释放并结束采集”，可先点击“确认当前 / 最近 episode 人工结果”，等候状态显示已接收该结果后再释放。未确认时以 `unknown` 保存，不自动算作成功。

## 数据与标签

- 七关节顺序固定为 `shoulder_pan`、`shoulder_lift`、`elbow_flex`、`wrist_flex`、`wrist_yaw`、`wrist_roll`、`gripper`。
- `observation.state` 仅保存上述关节状态；`observation.tactile` 为本拍因果选中的左右当前力。历史由 `sensor_windows` 读取，保持 Sensor Sidecar v2 契约与 Raw/Sync v1 格式。
- `action` 保存同一完整控制拍最终成功提交的七关节位置目标，单位 °。它不是 Teacher 候选，也不是机械臂实测位置。
- `u_label_deg_s` 来自夹爪成功提交目标的变化率；原始主臂意图、Teacher 候选、安全裁剪、标签有效性和阶段留在独立控制附件中。无效标签和故障周期保留，不能删行后当作连续示范。
- 每条使用独立 episode UID。`teacher/<uid>/control.jsonl` 保存完整控制证据，`frames.jsonl` 关联主帧、控制周期与时间戳；`meta/teacher_episodes/<uid>.json` 保存校验哈希、配置来源、阶段统计、实际频率、完整性及人工任务结果。
- 默认目标为主数据/视频 30 Hz、夹爪控制 100 Hz，X518 原始采集按当前明确配置运行。清单分别报告主帧、控制及原始力实测速率；软件提交、模式读回与文件验证均不能证明硬件 ACK 或机构物理到位。

## 只读核验

先结束采集并关闭持有数据集的录制会话。验证器拒绝活动 Writer，只读取已保存文件，不执行恢复、不连接硬件、不产生新数据。

```powershell
uv run --no-sync python examples/force_teacher/validate_demonstration.py "outputs/teacher_demonstrations"
uv run --no-sync python examples/force_teacher/validate_demonstration.py "outputs/teacher_demonstrations" --episode-uid "替换为实际 UUID"
```

默认检查 Teacher 附件、主数据、Raw/Sync 引用、当前力一致性、动作提交证据、因果时间及视频逐帧解码。`--skip-video-decode` 仅用于快速诊断；带该选项的成功结果不表示视频已解码验证。退出码 0 表示所选证据通过，1 表示错误或不完整。文件完整、阶段覆盖和人工任务结果应分别检查。

### 60 秒只读联合采集

`recording_readonly_hardware_smoke_test.py` 使用查询专用 B601 传输，采集七关节反馈、相机和 X518，保存观测数据并核验。它要求完整校准/配置和显式 `--observe-only`，不包含使能、位置或力矩命令，不产生动作标签或成功 Teacher 示范。全臂运动门禁仍保留。

先将独立 `TeacherRecordConfig` JSON 另存为 `outputs/recording.reviewed.json`，填写实际任务、物体、接触条件和相机。将示例相机编号替换为本次按设备名称枚举得到的索引：

```json
{
  "root": "outputs/teacher_demonstrations",
  "repo_id": "local/force_teacher_demo",
  "task": "填写真实任务描述",
  "object_id": "填写物体",
  "contact_condition_id": "填写接触条件",
  "fps": 30,
  "camera_max_age_ms": 100,
  "queue_capacity": 128,
  "target_episode_count": 10,
  "resume": false,
  "camera_configs": {
    "front": {
      "type": "opencv",
      "index_or_path": 1,
      "backend": 700,
      "fps": 30,
      "width": 640,
      "height": 480
    },
    "wrist": {
      "type": "opencv",
      "index_or_path": 0,
      "backend": 700,
      "fps": 30,
      "width": 640,
      "height": 480
    }
  }
}
```

在完整全臂参数与校准已填好后执行；`--root` 须为新目录，并覆盖录制 JSON 中的输出目录：

```powershell
uv run --no-sync python examples/force_teacher/recording_readonly_hardware_smoke_test.py `
  --config "outputs/full_arm_profile.reviewed.json" `
  --recording-config "outputs/recording.reviewed.json" `
  --root "outputs/readonly_combined_60s" --duration-s 60 --observe-only
```

检查新目录的 `readonly_capture_report.json`：文件/视频/Raw-Sync 核验与设备关闭证据均需成功。该工具的通过只证明本次只读联合采集；不能记入 10 条动态示范，也不能解除物理运动验收门禁。

### 使能前主从对齐检查

重查对齐使用独立查询工具，不包含 offset、home、enable、mode write 或 disable：

```powershell
uv run --no-sync python examples/force_teacher/full_arm_alignment_hardware_smoke_test.py `
  --config "outputs/full_arm_alignment.reviewed.json" `
  --output "outputs/alignment_check"
```

它显示全部七轴原始/映射主臂角、从臂角和差值，先读主臂再读从臂，保留真实时间戳。`complete` 仅表示只读采集与关闭完成；`ready_for_bounded_start` 也只覆盖对齐、模式/看门狗和失能检查，不替代 capability 准入、X518 检查或运动授权。关闭预算为 2 秒，未确认关闭时保存 `resources_retained` 并保持不完整标记。任务示范的数量和成功情况须根据实际录制结果另行核验。

## 本机资料与实机验收

运行日志、采集数据、本机参数、校准和备份保存在版本控制之外。恢复备份时先在独立目录核对文件哈希、本机配置及校准来源，再替换当前环境。

默认启动器加载完整六臂 MIT 与当前夹爪参数；由用户本人取得新参考、通过现有确认后进行全臂遥操，再进入任务录制并逐条检查示范。历史只读、逐关节 micro、短时联合诊断各自保存证据，可按需使用，但不把微动限速/行程/时长或历史诊断结果强加给正常遥操。校准与不可关闭的保护仍保留；离线通过不计作实机验收。
