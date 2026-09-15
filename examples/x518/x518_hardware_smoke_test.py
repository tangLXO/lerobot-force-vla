#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a read-only hardware smoke test against an X518 sensor.

中文说明：连接真实 X518，读取设备参数与连续双通道样本，全程只使用 Modbus 功能码 03。
"""

import argparse
import json
import queue
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.sensors.x518 import X518ChannelConfig, X518Sensor, X518SensorConfig
from lerobot.sensors.x518.diagnostics import summarize_capture

_LEFT_FORCE = "left.normal_force"
_RIGHT_FORCE = "right.normal_force"


def _positive_int(value: str) -> int:
    """Parse a command-line integer that must be greater than zero.

    中文说明：把命令行文本转换为正整数，用于样本数量和读取超时等参数。
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于零的整数")
    return parsed


def _parse_args() -> argparse.Namespace:
    """Parse X518 connection and smoke-test options.

    中文说明：解析设备地址、端口、从站号、期望单位、样本数和单次读取超时。
    """
    parser = argparse.ArgumentParser(description="X518 只读真机冒烟测试")
    parser.add_argument("--host", default="192.168.1.100", help="X518 IP 地址")
    parser.add_argument("--port", type=int, default=502, help="Modbus-TCP 端口")
    parser.add_argument("--unit-id", type=int, default=1, help="Modbus 从站号")
    parser.add_argument("--samples", type=_positive_int, default=10, help="连续读取的样本数")
    parser.add_argument("--duration-s", type=float, help="Measure every acquisition over this duration")
    parser.add_argument("--acquisition-mode", choices=("process", "thread"), default="process")
    parser.add_argument("--sample-rate-hz", type=float, default=400)
    parser.add_argument("--output", type=Path, help="New JSON diagnostic report path")
    parser.add_argument(
        "--timeout-ms",
        type=_positive_int,
        default=1000,
        help="等待每个新样本的超时时间（毫秒）",
    )
    parser.add_argument(
        "--expected-unit",
        choices=("t", "kg", "g", "kN", "N", "lb", "any"),
        default="kg",
        help="期望设备单位；any 表示接受任一已知单位",
    )
    return parser.parse_args()


def _print_connection_hint(host: str) -> None:
    """Print practical checks after a hardware connection failure.

    中文说明：连接失败时提示检查供电、网线、网卡状态和设备所在 IPv4 网段。
    """
    print("\n排查建议：")
    print("  1. 确认 X518 已通电，设备与电脑两端的网口指示灯亮起。")
    print(f"  2. 确认电脑有一块已连接网卡与 {host} 位于同一 IPv4 网段。")
    print("  3. 设备为 192.168.1.100 时，电脑网卡可配置为 192.168.1.110/24。")
    print("  4. 确认 TCP 502 端口、Modbus 从站号和设备实际设置一致。")


def main() -> int:
    """Connect, validate settings, read consecutive samples, and disconnect.

    中文说明：建立只读连接，验证序号、时间戳、有效标志和双通道值，最后可靠释放资源。
    """
    args = _parse_args()
    if args.duration_s is not None and (not np.isfinite(args.duration_s) or args.duration_s <= 0):
        raise ValueError("--duration-s must be positive and finite")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    expected_unit = None if args.expected_unit == "any" else args.expected_unit
    config = X518SensorConfig(
        host=args.host,
        port=args.port,
        unit_id=args.unit_id,
        expected_unit=expected_unit,
        sample_rate_hz=args.sample_rate_hz,
        expected_sample_rate_hz=args.sample_rate_hz,
        acquisition_mode=args.acquisition_mode,
        channels={
            _LEFT_FORCE: X518ChannelConfig(channel=1),
            _RIGHT_FORCE: X518ChannelConfig(channel=2),
        },
    )
    sensor = X518Sensor(config)
    connected = False

    print(f"连接 X518：{args.host}:{args.port}，unit_id={args.unit_id}")
    print("测试方式：只读 Modbus-TCP 功能码 03，不写入设备寄存器。")

    try:
        sensor.connect()
        connected = True
        settings = sensor._device_settings
        if settings is None:
            raise RuntimeError("连接成功，但没有取得设备参数")

        print(
            "连接成功："
            f"设备单位={settings.unit}，小数位={settings.decimal}，"
            f"设备采样率={settings.sample_rate_hz:g} Hz，字交换={settings.word_swap}"
        )
        if args.duration_s is not None:
            subscription = sensor.subscribe(config.resolve_recorder_queue_capacity())
            rows = []
            start = time.perf_counter_ns()
            deadline = time.perf_counter() + args.duration_s
            try:
                while time.perf_counter() < deadline:
                    rows.append(asdict(subscription.get(timeout=args.timeout_ms / 1000)))
                end = time.perf_counter_ns()
            finally:
                sensor.unsubscribe(subscription)
            while True:
                try:
                    rows.append(asdict(subscription.get_nowait()))
                except queue.Empty:
                    break
            result = summarize_capture(rows, target_hz=args.sample_rate_hz, start_ns=start, end_ns=end)
            sensor.disconnect()
            connected = False
            result["acquisition"] = sensor.diagnostics
            result["queue_overflow_count"] = subscription.overflow_count
            result["resources_released"] = not sensor.has_resources
            result["recorder_queue"] = subscription.diagnostics.snapshot(
                subscription.queue, overflow_count=subscription.overflow_count
            )
            result["passed"] &= subscription.overflow_count == 0 and sensor.diagnostics["fault"] is None
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
            return 0 if result["passed"] else 1
        print("\n序号 | 左通道/N | 右通道/N | timestamp_ns")
        print("-" * 66)

        previous_sequence = -1
        previous_timestamp_ns = -1
        last_sample = None
        for _ in range(args.samples):
            sample = sensor.async_read(timeout_ms=args.timeout_ms)
            if not sample.is_valid:
                raise RuntimeError(f"收到无效样本：sequence={sample.sequence}")
            if sample.sequence <= previous_sequence:
                raise RuntimeError("样本序号没有严格递增")
            if sample.timestamp_ns < previous_timestamp_ns:
                raise RuntimeError("样本时间戳发生倒退")
            if set(sample.values) != {_LEFT_FORCE, _RIGHT_FORCE}:
                raise RuntimeError(f"样本特征与配置不一致：{sorted(sample.values)}")

            print(
                f"{sample.sequence:4d} | "
                f"{sample.values[_LEFT_FORCE]:10.6f} | "
                f"{sample.values[_RIGHT_FORCE]:10.6f} | "
                f"{sample.timestamp_ns}"
            )
            previous_sequence = sample.sequence
            previous_timestamp_ns = sample.timestamp_ns
            last_sample = sample

        latest = sensor.read_latest(max_age_ms=args.timeout_ms)
        if last_sample is not None and latest.sequence < last_sample.sequence:
            raise RuntimeError("read_latest() 返回了更旧的样本")

        print(f"\n[通过] 已连续读取 {args.samples} 个有效样本，双通道输出单位均为 N。")
        return 0
    except Exception as exc:
        print(f"\n[失败] {type(exc).__name__}: {exc}")
        _print_connection_hint(args.host)
        return 1
    finally:
        if connected or sensor.has_resources:
            sensor.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
