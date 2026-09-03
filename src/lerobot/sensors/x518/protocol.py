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

"""Private Modbus-TCP transport and register decoding for X518 sensors.

中文说明：封装 X518 使用的只读 Modbus-TCP 通信、响应校验和寄存器数据解码。
"""

from __future__ import annotations

import contextlib
import socket
import struct
import threading
from dataclasses import dataclass

_READ_HOLDING_REGISTERS = 0x03
_REALTIME_DATA_ADDRESS = 0x0A00

_UNIT_CODE_ADDRESS = 0x0614
_DECIMAL_ADDRESS = 0x0616
_SAMPLE_RATE_CODE_ADDRESS = 0x061A
_CHANNEL_SELECT_ADDRESS = 0x061C
_DATA_FORMAT_ADDRESS = 0x0638
_ETHERNET_PROTOCOL_ADDRESS = 0x0644

# The fields needed at connect time live in two compact, contiguous ranges.
_MEASUREMENT_CONFIG_BLOCK = (_UNIT_CODE_ADDRESS, 10)  # 0x0614..0x061D
_TRANSPORT_CONFIG_BLOCK = (_DATA_FORMAT_ADDRESS, 14)  # 0x0638..0x0645

_UNIT_MAP = {
    1: "t",
    2: "kg",
    3: "g",
    4: "kN",
    5: "N",
    6: "lb",
}

_SAMPLE_RATE_MAP = {
    0: 6.25,
    1: 12.5,
    2: 25.0,
    3: 50.0,
    4: 100.0,
    5: 200.0,
    6: 400.0,
    7: 800.0,
    8: 1600.0,
}

_UNIT_TO_NEWTONS = {
    "t": 9806.65,
    "kg": 9.80665,
    "g": 0.00980665,
    "kN": 1000.0,
    "N": 1.0,
    "lb": 4.4482216152605,
}


@dataclass(frozen=True)
class _X518DeviceSettings:
    """Decoded measurement and transport settings reported by the device.

    中文说明：保存连接时从设备寄存器读取并解码出的单位、小数位、采样率和传输格式。
    """

    unit_code: int
    unit: str
    decimal: int
    scale: float
    sample_rate_code: int
    sample_rate_hz: float
    channel_select: int
    data_format: int
    word_swap: bool
    ethernet_protocol: int

    @property
    def newtons_per_device_unit(self) -> float:
        """Return the multiplier that converts one configured device unit to newtons.

        中文说明：返回设备当前力单位换算为牛顿时使用的倍率。
        """
        return _UNIT_TO_NEWTONS[self.unit]


class _ModbusTCPClient:
    """Minimal X518 transport supporting only Modbus function code 03.

    中文说明：这是 X518 的私有只读客户端，仅实现读取保持寄存器功能码 03。
    """

    def __init__(self, host: str, port: int, unit_id: int, timeout_s: float) -> None:
        """Store connection parameters without opening a socket.

        中文说明：保存 IP、端口、从站号和超时参数；构造对象时不会连接设备。
        """
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._transaction_id = 0
        self._lock = threading.RLock()

    @property
    def socket(self) -> socket.socket | None:
        """Expose the active socket for connection-state checks.

        中文说明：返回当前 TCP socket；未连接或连接已关闭时返回 ``None``。
        """
        return self._socket

    def connect(self) -> None:
        """Open and configure a fresh TCP connection to the X518.

        中文说明：先关闭旧连接，再创建新的 TCP 连接并设置超时、低延迟和保活选项。
        """
        with self._lock:
            self.close()
            sock: socket.socket | None = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                sock.settimeout(self.timeout_s)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self._socket = sock
            except Exception:
                if sock is not None:
                    with contextlib.suppress(OSError):
                        sock.close()
                raise

    def close(self) -> None:
        """Interrupt pending socket I/O and release the TCP connection.

        中文说明：主动 shutdown 后关闭 socket，可打断另一个线程中阻塞的 ``recv()``。
        """
        # Deliberately do not acquire _lock: shutdown must be able to interrupt
        # another thread blocked in recv() while holding the request lock.
        sock = self._socket
        self._socket = None
        if sock is None:
            return

        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            sock.close()

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        """Receive exactly ``size`` bytes or fail if the peer closes early.

        中文说明：循环接收直到凑满指定字节数；设备提前断开时抛出连接错误。
        """
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("X518 closed the TCP connection.")
            data.extend(chunk)
        return bytes(data)

    def request(self, pdu: bytes) -> bytes:
        """Send one Modbus PDU and return its validated response body.

        中文说明：组装并发送 MBAP 报文，随后校验事务号、协议号、从站号和响应长度。
        """
        with self._lock:
            sock = self._socket
            if sock is None:
                raise ConnectionError("X518 TCP connection is not open.")

            self._transaction_id = (self._transaction_id + 1) & 0xFFFF
            transaction_id = self._transaction_id
            mbap = struct.pack(">HHHB", transaction_id, 0, len(pdu) + 1, self.unit_id)
            sock.sendall(mbap + pdu)

            header = self._recv_exact(sock, 7)
            response_id, protocol_id, length, response_unit = struct.unpack(">HHHB", header)
            if response_id != transaction_id:
                raise RuntimeError(f"X518 Modbus transaction ID mismatch: {response_id} != {transaction_id}.")
            if protocol_id != 0:
                raise RuntimeError(f"Invalid X518 Modbus protocol ID: {protocol_id}.")
            if response_unit != self.unit_id:
                raise RuntimeError(f"X518 Modbus unit ID mismatch: {response_unit} != {self.unit_id}.")
            if not 2 <= length <= 254:
                raise RuntimeError(f"Invalid X518 Modbus-TCP response length: {length}.")

            body = self._recv_exact(sock, length - 1)
            if body[0] & 0x80:
                exception_code = body[1] if len(body) >= 2 else None
                raise RuntimeError(f"X518 Modbus exception response: code={exception_code}.")
            return body

    def read_holding_registers(self, address: int, count: int) -> tuple[int, ...]:
        """Read and decode a contiguous range of holding registers.

        中文说明：使用功能码 03 读取连续寄存器，并严格检查功能码、字节数和载荷长度。
        """
        if not 0 <= address <= 0xFFFF:
            raise ValueError(f"Invalid Modbus register address: {address}.")
        if not 1 <= count <= 125:
            raise ValueError(f"Modbus register count must be in 1..125, got {count}.")
        if address + count - 1 > 0xFFFF:
            raise ValueError("Modbus register range exceeds 0xFFFF.")

        body = self.request(struct.pack(">BHH", _READ_HOLDING_REGISTERS, address, count))
        if len(body) < 2:
            raise RuntimeError("X518 Modbus response is missing its byte-count field.")
        if body[0] != _READ_HOLDING_REGISTERS:
            raise RuntimeError(f"Unexpected X518 Modbus function code: 0x{body[0]:02X}.")

        expected_bytes = count * 2
        if body[1] != expected_bytes:
            raise RuntimeError(f"X518 Modbus byte count mismatch: got {body[1]}, expected {expected_bytes}.")
        payload = body[2:]
        if len(payload) != expected_bytes:
            raise RuntimeError(
                f"X518 Modbus payload length mismatch: got {len(payload)}, expected {expected_bytes}."
            )
        return struct.unpack(f">{count}H", payload)

    def read_device_settings(self) -> _X518DeviceSettings:
        """Read and decode all device settings required for acquisition.

        中文说明：连接后批量读取测量与传输配置，并解析单位、小数位、采样率和字交换规则。
        """
        measurement_start, measurement_count = _MEASUREMENT_CONFIG_BLOCK
        transport_start, transport_count = _TRANSPORT_CONFIG_BLOCK
        measurement = self.read_holding_registers(measurement_start, measurement_count)
        transport = self.read_holding_registers(transport_start, transport_count)

        def decode(block: tuple[int, ...], block_start: int, address: int) -> int:
            """Decode one signed 32-bit setting from a previously read register block.

            中文说明：按目标地址计算块内偏移，将两个 16 位寄存器还原为有符号 32 位配置值。
            """
            offset = address - block_start
            return self._u32_to_i32(self._registers_to_u32(block[offset], block[offset + 1]))

        unit_code = decode(measurement, measurement_start, _UNIT_CODE_ADDRESS)
        decimal = decode(measurement, measurement_start, _DECIMAL_ADDRESS)
        sample_rate_code = decode(measurement, measurement_start, _SAMPLE_RATE_CODE_ADDRESS)
        channel_select = decode(measurement, measurement_start, _CHANNEL_SELECT_ADDRESS)
        data_format = decode(transport, transport_start, _DATA_FORMAT_ADDRESS)
        ethernet_protocol = decode(transport, transport_start, _ETHERNET_PROTOCOL_ADDRESS)

        unit = _UNIT_MAP.get(unit_code)
        if unit is None:
            raise RuntimeError(f"Unknown X518 force unit code: {unit_code}.")
        if not 0 <= decimal <= 5:
            raise RuntimeError(f"Invalid X518 decimal setting: {decimal}.")
        sample_rate_hz = _SAMPLE_RATE_MAP.get(sample_rate_code)
        if sample_rate_hz is None:
            raise RuntimeError(f"Unknown X518 sample-rate code: {sample_rate_code}.")

        return _X518DeviceSettings(
            unit_code=unit_code,
            unit=unit,
            decimal=decimal,
            scale=10.0 ** (-decimal),
            sample_rate_code=sample_rate_code,
            sample_rate_hz=sample_rate_hz,
            channel_select=channel_select,
            data_format=data_format,
            word_swap=(data_format // 10) == 1,
            ethernet_protocol=ethernet_protocol,
        )

    def read_raw_channels(self, *, word_swap: bool) -> tuple[int, int]:
        """Read both realtime channels as signed 32-bit raw integers.

        中文说明：从实时数据区一次读取四个寄存器，并按设备格式解码两路有符号原始值。
        """
        registers = self.read_holding_registers(_REALTIME_DATA_ADDRESS, 4)
        channel_1 = self._registers_to_u32(registers[0], registers[1], word_swap=word_swap)
        channel_2 = self._registers_to_u32(registers[2], registers[3], word_swap=word_swap)
        return self._u32_to_i32(channel_1), self._u32_to_i32(channel_2)

    @staticmethod
    def _registers_to_u32(high: int, low: int, *, word_swap: bool = False) -> int:
        """Combine two 16-bit registers into one unsigned 32-bit integer.

        中文说明：必要时先交换高低字，再合并成无符号 32 位整数。
        """
        if word_swap:
            high, low = low, high
        return ((high & 0xFFFF) << 16) | (low & 0xFFFF)

    @staticmethod
    def _u32_to_i32(value: int) -> int:
        """Interpret an unsigned 32-bit value as a two's-complement signed integer.

        中文说明：按二进制补码规则把无符号 32 位数转换为有符号整数，保留负力值。
        """
        value &= 0xFFFFFFFF
        return value - 0x100000000 if value & 0x80000000 else value
