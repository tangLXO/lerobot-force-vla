"""Short, evidence-based explanations for errors shown by the desktop."""


def explain_problems(messages):
    """Explain recognized error text while leaving the original messages intact."""
    entries = tuple(str(message).casefold() for message in messages)
    text = "\n".join(entries)
    explanations = []
    if "dataset already exists; explicitly enable resume" in text:
        explanations.append(
            "数据目录已经存在：若要继续使用原目录，请确认配置一致并勾选续录；否则选择新目录。"
        )
    if "camera device_path must match exactly one device" in text:
        explanations.append(
            "相机设备路径无法唯一匹配：在“设备与配置”刷新设备列表，预览并确认前景和腕部后保存绑定。"
        )
    if "measured calibration overtravel" in text:
        explanations.append("关节实测位置超出校准硬范围：检查对应关节的机械位置、零点及校准文件。")
    elif "outside calibration/reference bounds" in text or "reference-envelope overtravel" in text:
        explanations.append("关节反馈超出校准或启动参考范围：检查实际位置、零点及对应的参考范围。")
    if "left_force_hard_limit" in text:
        explanations.append("左侧原始力触发硬限保护：检查接触状态和左侧力传感器反馈。")
    if "right_force_hard_limit" in text:
        explanations.append("右侧原始力触发硬限保护：检查接触状态和右侧力传感器反馈。")
    if "stale_force" in text or "force_sensor_disconnected" in text:
        explanations.append("力传感器反馈过期或连接中断：检查 X518 连接和只读反馈。")
    if "raw producer tail was not sealed" in text:
        explanations.append(
            "高速力数据的结束边界未确认，相关录制的完整性不能确认。此报错也可能出现在清理阶段，"
            "请查看更早的原始错误和本次运行日志。"
        )
    if any("partial" in entry and ("write" in entry or "submission" in entry) for entry in entries):
        explanations.append("命令只完成部分写入或提交：设备执行状态可能未知，请查看退出证据并完成只读复核。")
    if "cached feedback expired" in text or "joint_feedback_deadline" in text:
        explanations.append("关节反馈过期或超过截止时间：检查串口连接，并查看本次运行日志中的时序证据。")
    if any(token in text for token in ("timeouterror", "timed out", "deadline expired", "超时")):
        explanations.append("有操作超时或超过截止时间：请结合下方原始报错和运行日志定位具体环节。")
    if "queue overflow" in text or "capture ipc overflow" in text or "control tick gap" in text:
        explanations.append("采集队列溢出或控制帧不连续：本条数据可能不完整，请查看采集状态和日志。")
    if not explanations:
        explanations.append("暂未识别此错误的具体含义；请复制下方原始报错，并结合运行日志排查。")
    return explanations
