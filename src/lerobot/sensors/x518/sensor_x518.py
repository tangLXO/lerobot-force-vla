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

"""LeRobot sensor driver for the X518 dual-channel force acquisition device.

中文说明：把 X518 的 Modbus-TCP 双通道读数转换为统一的 ``SensorSample`` 牛顿值。
"""

import logging
import math
import threading
import time

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..sensor import Sensor, SensorSample
from .configuration_x518 import X518SensorConfig
from .protocol import _ModbusTCPClient, _X518DeviceSettings

logger = logging.getLogger(__name__)


class X518Sensor(Sensor):
    """Acquire calibrated X518 channels in newtons over Modbus-TCP.

    中文说明：管理连接、后台轮询、断线重连和三种读取接口，最终始终输出牛顿值。
    """

    _PRECISE_WAIT_WINDOW_S = 0.020
    _ERROR_LOG_INTERVAL_NS = 1_000_000_000

    def __init__(self, config: X518SensorConfig):
        """Initialize the driver without opening a network connection.

        中文说明：保存配置并初始化线程同步状态和协议客户端；此阶段不会访问硬件。
        """
        super().__init__(config)
        self.config = config
        self._client = _ModbusTCPClient(
            host=config.host,
            port=config.port,
            unit_id=config.unit_id,
            timeout_s=config.request_timeout_s,
        )

        self._lifecycle_lock = threading.RLock()
        self._sample_ready = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_active = False

        self._device_settings: _X518DeviceSettings | None = None
        self._latest_sample: SensorSample | None = None
        self._next_sequence = 0
        self._last_consumed_sequence = -1
        self._last_error_log_ns = 0
        self._suppressed_error_logs = 0

    @property
    def features(self) -> dict[str, type]:
        """Describe configured semantic force outputs, each expressed in newtons.

        中文说明：返回配置中声明的语义特征，每个特征的数据类型都是 ``float``。
        """
        return dict.fromkeys(self.config.channels, float)

    @property
    def is_connected(self) -> bool:
        """Return whether acquisition is active and its background reader is alive.

        中文说明：只有生命周期已激活且后台采样线程仍存活时才认为传感器已连接。
        """
        thread = self._thread
        return self._lifecycle_active and thread is not None and thread.is_alive()

    def connect(self) -> None:
        """Connect, validate device settings, and start background acquisition.

        中文说明：按配置重试连接，读取并校验设备参数，然后启动后台轮询线程。
        """
        with self._lifecycle_lock:
            if self._lifecycle_active or self.is_connected:
                raise DeviceAlreadyConnectedError(f"{self.__class__.__name__} is already connected.")

            self._stop_event.clear()
            with self._sample_ready:
                self._latest_sample = None
                self._next_sequence = 0
                self._last_consumed_sequence = -1

            try:
                settings = self._connect_initially()
                self._apply_device_settings(settings)
                self._lifecycle_active = True
                self._thread = threading.Thread(
                    target=self._reader_loop,
                    name=f"X518Reader-{self.config.host}:{self.config.port}",
                    daemon=True,
                )
                self._thread.start()
            except Exception:
                self._lifecycle_active = False
                self._stop_event.set()
                self._client.close()
                self._thread = None
                self._device_settings = None
                self.sample_rate_hz = self.config.sample_rate_hz
                raise

        logger.info(
            "%s connected at %.3g Hz; device unit=%s, outputs converted to N.",
            self,
            self.sample_rate_hz,
            settings.unit,
        )

    def _connect_initially(self) -> _X518DeviceSettings:
        """Establish the initial connection with bounded exponential-backoff retries.

        中文说明：首次连接最多尝试 ``connect_retries + 1`` 次，并按指数退避等待后重试。
        """
        last_error: Exception | None = None
        for attempt in range(self.config.connect_retries + 1):
            try:
                self._client.connect()
                return self._client.read_device_settings()
            except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                last_error = exc
                self._client.close()
                if attempt >= self.config.connect_retries:
                    break
                delay_s = self.config.connect_backoff_s * (2**attempt)
                if self._stop_event.wait(delay_s):
                    raise ConnectionError("X518 connection attempt was stopped.") from exc

        if last_error is None:  # pragma: no cover - loop always makes at least one attempt
            raise ConnectionError("X518 initial connection failed.")
        raise last_error

    def _apply_device_settings(self, settings: _X518DeviceSettings) -> None:
        """Validate reported device settings and select the effective polling rate.

        中文说明：确认设备处于 Modbus-TCP 模式、单位符合预期，并确定实际本地轮询频率。
        """
        if settings.ethernet_protocol != 1:
            raise RuntimeError(
                f"X518 Ethernet protocol must be Modbus-TCP (1), got {settings.ethernet_protocol}."
            )
        if self.config.expected_unit is not None and settings.unit != self.config.expected_unit:
            raise RuntimeError(
                f"X518 force unit is {settings.unit!r}, expected {self.config.expected_unit!r}."
            )

        requested_rate = self.config.sample_rate_hz
        if requested_rate is not None and requested_rate > settings.sample_rate_hz + 1e-9:
            raise ValueError(
                f"Requested X518 sample_rate_hz={requested_rate:g} exceeds the device rate "
                f"of {settings.sample_rate_hz:g} Hz."
            )

        self._device_settings = settings
        self.sample_rate_hz = requested_rate if requested_rate is not None else settings.sample_rate_hz

    def _reader_loop(self) -> None:
        """Continuously poll, reconnect after failures, and publish valid samples.

        中文说明：后台线程的主循环；按采样周期读取数据，失败时关闭连接并自动重连。
        """
        next_deadline = time.perf_counter()

        while not self._stop_event.is_set():
            if not self._wait_until(next_deadline):
                break

            try:
                if self._client.socket is None and not self._reconnect_until_ready():
                    break
                values, timestamp_ns = self._read_values()
                if self._stop_event.is_set():
                    break
                self._publish_sample(values, timestamp_ns)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._client.close()
                self._record_read_error(exc)

            sample_rate_hz = self.sample_rate_hz
            if sample_rate_hz is None:  # pragma: no cover - settings validation establishes a rate
                self._record_read_error(RuntimeError("X518 polling rate is unavailable."))
                break

            period_s = 1.0 / sample_rate_hz
            next_deadline += period_s
            now = time.perf_counter()
            if now - next_deadline > period_s:
                next_deadline = now + period_s

        with self._sample_ready:
            self._sample_ready.notify_all()

    def _wait_until(self, deadline: float) -> bool:
        """Wait until an absolute polling deadline or until shutdown is requested.

        中文说明：等待到下一次采样时刻，同时允许 ``disconnect()`` 通过停止事件立即打断等待。
        """
        while not self._stop_event.is_set():
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0:
                return True
            if remaining_s > self._PRECISE_WAIT_WINDOW_S:
                if self._stop_event.wait(remaining_s - self._PRECISE_WAIT_WINDOW_S):
                    return False
            else:
                time.sleep(remaining_s)
        return False

    def _reconnect_until_ready(self) -> bool:
        """Reconnect repeatedly and refresh device settings until ready or stopped.

        中文说明：断线后持续尝试恢复连接；每次成功后重新读取设备参数，停止时返回 ``False``。
        """
        while not self._stop_event.is_set():
            try:
                self._client.connect()
                if self._stop_event.is_set():
                    self._client.close()
                    return False
                settings = self._client.read_device_settings()
                if self._stop_event.is_set():
                    self._client.close()
                    return False
                self._apply_device_settings(settings)
                return True
            except Exception as exc:
                self._client.close()
                self._record_read_error(exc)
                if self._stop_event.wait(self.config.reconnect_delay_s):
                    return False
        return False

    def _read_values(self) -> tuple[dict[str, float], int]:
        """Read raw channels and map scaled values to semantic features in newtons.

        中文说明：读取两路原始值，使用高分辨率 ``perf_counter_ns()`` 记录响应接收时刻，
        应用小数位与单位换算，再按显式 ``channels`` 映射生成结果。
        """
        settings = self._device_settings
        if settings is None:
            raise RuntimeError("X518 device settings are unavailable.")

        channel_values = self._client.read_raw_channels(word_swap=settings.word_swap)
        timestamp_ns = time.perf_counter_ns()
        device_to_newtons = settings.scale * settings.newtons_per_device_unit
        values = {
            feature_name: channel_values[channel_config.channel - 1] * device_to_newtons
            for feature_name, channel_config in self.config.channels.items()
        }
        if not all(math.isfinite(value) for value in values.values()):  # pragma: no cover - int32 is finite
            raise RuntimeError("X518 returned a non-finite force value.")
        return values, timestamp_ns

    def _publish_sample(self, values: dict[str, float], timestamp_ns: int) -> None:
        """Publish one valid sample and wake readers waiting for new data.

        中文说明：为成功读数分配递增序号、保存为最新样本，并唤醒等待中的读取调用。
        """
        with self._sample_ready:
            sample = SensorSample(
                timestamp_ns=timestamp_ns,
                sequence=self._next_sequence,
                values=values,
                is_valid=True,
            )
            self._next_sequence += 1
            self._latest_sample = sample
            self._sample_ready.notify_all()

    def _record_read_error(self, exc: Exception) -> None:
        """Rate-limit repeated acquisition warnings while retaining their count.

        中文说明：限制连续采样错误的日志频率，避免断线期间刷屏，同时记录被抑制的错误数量。
        """
        now_ns = time.perf_counter_ns()
        with self._sample_ready:
            if now_ns - self._last_error_log_ns >= self._ERROR_LOG_INTERVAL_NS:
                suppressed = self._suppressed_error_logs
                self._suppressed_error_logs = 0
                self._last_error_log_ns = now_ns
            else:
                self._suppressed_error_logs += 1
                return

        suffix = f" ({suppressed} similar errors suppressed)" if suppressed else ""
        logger.warning("X518 acquisition failed: %s%s", exc, suffix)

    def read(self) -> SensorSample:
        """Wait indefinitely for a sample produced after this call begins.

        中文说明：忽略调用前已有的样本，一直阻塞到后台线程发布一个更新的样本。
        """
        with self._sample_ready:
            self._require_active()
            baseline_sequence = self._latest_sample.sequence if self._latest_sample is not None else -1
            while self._latest_sample is None or self._latest_sample.sequence <= baseline_sequence:
                self._sample_ready.wait()
                self._require_active()
            sample = self._latest_sample
            self._last_consumed_sequence = sample.sequence
            return sample

    def async_read(self, timeout_ms: float = 200) -> SensorSample:
        """Return the newest unconsumed sample, waiting up to ``timeout_ms``.

        中文说明：返回尚未被消费的最新样本；没有新样本时最多等待指定毫秒数。
        """
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int | float):
            raise ValueError(f"timeout_ms must be a finite non-negative number, got {timeout_ms!r}.")
        if not math.isfinite(timeout_ms) or timeout_ms < 0:
            raise ValueError(f"timeout_ms must be a finite non-negative number, got {timeout_ms!r}.")

        deadline = time.perf_counter() + timeout_ms / 1000.0
        with self._sample_ready:
            self._require_active()
            while self._latest_sample is None or self._latest_sample.sequence <= self._last_consumed_sequence:
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    raise TimeoutError(f"Timed out waiting for a new X518 sample after {timeout_ms:g} ms.")
                self._sample_ready.wait(timeout=remaining_s)
                self._require_active()

            sample = self._latest_sample
            self._last_consumed_sequence = sample.sequence
            return sample

    def read_latest(self, max_age_ms: int = 500) -> SensorSample:
        """Peek at the latest sample, rejecting samples older than ``max_age_ms``.

        中文说明：非消费式查看最新样本；没有样本或样本超过允许时效时会报错。
        """
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int | float):
            raise ValueError(f"max_age_ms must be a finite number, got {max_age_ms!r}.")
        if not math.isfinite(max_age_ms):
            raise ValueError(f"max_age_ms must be a finite number, got {max_age_ms!r}.")

        with self._sample_ready:
            self._require_active()
            sample = self._latest_sample

        if sample is None:
            raise RuntimeError("X518 has not produced a sample yet.")
        age_ms = (time.perf_counter_ns() - sample.timestamp_ns) / 1e6
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"Latest X518 sample is {age_ms:.1f} ms old (maximum allowed: {max_age_ms:g} ms)."
            )
        return sample

    def _require_active(self) -> None:
        """Raise when the sensor lifecycle or background reader is not active.

        中文说明：统一检查连接状态和后台线程状态，供三个读取接口在访问样本前调用。
        """
        if not self._lifecycle_active:
            raise DeviceNotConnectedError(f"{self.__class__.__name__} is not connected.")
        thread = self._thread
        if thread is None or not thread.is_alive():
            raise RuntimeError("X518 background reader is not running.")

    def disconnect(self) -> None:
        """Stop acquisition, interrupt pending network I/O, and release resources.

        中文说明：设置停止事件、关闭 socket 以打断阻塞读取、等待线程退出并清理设备状态。
        """
        with self._lifecycle_lock:
            thread = self._thread
            if not self._lifecycle_active and (thread is None or not thread.is_alive()):
                raise DeviceNotConnectedError(f"{self.__class__.__name__} is not connected.")

            self._lifecycle_active = False
            self._stop_event.set()
            with self._sample_ready:
                self._sample_ready.notify_all()

            # Closing before joining interrupts a blocking recv().
            self._client.close()
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(2.0, self.config.request_timeout_s * 2))
                if thread.is_alive():  # pragma: no cover - depends on operating-system socket behavior
                    logger.warning("X518 background reader did not stop within the timeout.")

            # A reconnecting thread can finish socket.create_connection() after
            # the pre-join close. Close once more so that race cannot leak a socket.
            self._client.close()

            if thread is None or not thread.is_alive():
                self._thread = None
            self._device_settings = None
            self.sample_rate_hz = self.config.sample_rate_hz

        logger.info("%s disconnected.", self)

    def __str__(self) -> str:
        """Return a concise device identifier for logs and errors.

        中文说明：使用设备地址和端口生成便于识别的日志名称。
        """
        return f"X518Sensor({self.config.host}:{self.config.port})"
