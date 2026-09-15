# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Private X518 polling engine shared by thread and spawn transports."""

import logging
import struct
import time

from .protocol import _ModbusTCPClient

logger = logging.getLogger(__name__)


def advance_deadline(deadline_ns: int, now_ns: int, period_ns: int) -> tuple[int, int]:
    """Choose the next future slot, counting expired slots without catch-up reads."""
    next_ns = deadline_ns + period_ns
    skipped = max(0, (now_ns - next_ns) // period_ns + 1)
    return next_ns + skipped * period_ns, skipped


class X518Acquisition:
    """Own the socket and acquisition algorithm without framework or Dataset state."""

    def __init__(self, config, stop, publish, settings_changed, reconnect_required, service=lambda: None):
        """Keep acquisition state and the small callbacks supplied by either transport."""
        self.config = config
        self.stop = stop
        self.publish = publish
        self.settings_changed = settings_changed
        self.reconnect_required = reconnect_required
        self.service = service
        self.client = _ModbusTCPClient(config.host, config.port, config.unit_id, config.request_timeout_s)
        self.settings = None
        self.sample_rate_hz = config.sample_rate_hz
        self.missed_deadlines = 0
        self.reconnecting = False
        self._last_warning = float("-inf")

    def apply_settings(self, settings):
        """Validate the read-only device configuration and resolve the polling rate."""
        if settings.ethernet_protocol != 1:
            raise RuntimeError(
                f"X518 Ethernet protocol must be Modbus-TCP (1), got {settings.ethernet_protocol}."
            )
        if self.config.expected_unit is not None and settings.unit != self.config.expected_unit:
            raise RuntimeError(
                f"X518 force unit is {settings.unit!r}, expected {self.config.expected_unit!r}."
            )
        requested = self.config.sample_rate_hz
        if requested is not None and requested > settings.sample_rate_hz + 1e-9:
            raise ValueError(
                f"Requested X518 sample_rate_hz={requested:g} exceeds the device rate "
                f"of {settings.sample_rate_hz:g} Hz."
            )
        self.settings = settings
        self.sample_rate_hz = requested if requested is not None else settings.sample_rate_hz
        self.settings_changed(settings, self.sample_rate_hz)

    def wait_until(self, deadline_ns):
        """Sleep with the GIL released, servicing control requests between short waits."""
        while not self.stop.is_set():
            self.service()
            if self.stop.is_set():
                break
            remaining = (deadline_ns - time.perf_counter_ns()) / 1e9
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.005))
        return False

    def connect_initially(self):
        """Connect with the configured bounded retry budget."""
        for attempt in range(self.config.connect_retries + 1):
            if self.stop.is_set():
                raise ConnectionError("X518 connection attempt was stopped.")
            try:
                self.client.connect()
                settings = self.client.read_device_settings()
                self.apply_settings(settings)
                return settings
            except ValueError:
                self.client.close()
                raise
            except (TimeoutError, ConnectionError, OSError, RuntimeError):
                self.client.close()
                if attempt == self.config.connect_retries:
                    raise
                delay = self.config.connect_backoff_s * 2**attempt
                if not self.wait_until(time.perf_counter_ns() + int(delay * 1e9)):
                    raise ConnectionError("X518 connection attempt was stopped.") from None

    def read_components(self):
        """Timestamp the complete response before converting registers to newtons."""
        settings = self.settings
        if settings is None:
            raise RuntimeError("X518 device settings are unavailable.")
        channels = self.client.read_raw_channels(word_swap=settings.word_swap)
        timestamp_ns = time.perf_counter_ns()
        factor = settings.scale * settings.newtons_per_device_unit
        values = {
            name: channels[mapping.channel - 1] * factor for name, mapping in self.config.channels.items()
        }
        native = {
            f"channel_{mapping.channel}.register": channels[mapping.channel - 1]
            for mapping in self.config.channels.values()
        }
        payload = struct.pack(">ii", *channels) if self.config.record_native_payload else None
        return values, native, payload, timestamp_ns

    def reconnect(self):
        """Refresh device settings outside protected required-stream recordings."""
        self.reconnecting = True
        while not self.stop.is_set():
            self.service()
            if self.stop.is_set() or self.reconnect_required(ConnectionError("X518 connection lost")):
                return False
            try:
                self.client.connect()
                settings = self.client.read_device_settings()
            except Exception as exc:
                self.client.close()
                self.warn(exc)
                if not self.wait_until(time.perf_counter_ns() + int(self.config.reconnect_delay_s * 1e9)):
                    return False
                continue
            if self.stop.is_set():
                return False
            if self.reconnect_required(ConnectionError("X518 reconnected during recorder startup")):
                return False
            # Invalid settings and transport failures must remain fatal rather
            # than being swallowed as another recoverable socket read error.
            self.apply_settings(settings)
            self.reconnecting = False
            return True
        return False

    def warn(self, error):
        """Bound acquisition error logging independently of sample evidence."""
        now = time.perf_counter()
        if now - self._last_warning >= 1:
            logger.warning("X518 acquisition failed: %s", error)
            self._last_warning = now

    def run(self):
        """Perform one real request per slot and preserve invalid acquisition attempts."""
        deadline = time.perf_counter_ns()
        try:
            while self.wait_until(deadline):
                if self.client.socket is None and not self.reconnect():
                    break
                period = round(1e9 / self.sample_rate_hz)
                # A late wake can skip several slots, but never starts a burst of reads.
                late = max(0, (time.perf_counter_ns() - deadline) // period)
                deadline += late * period
                self.missed_deadlines += late
                error = None
                try:
                    values, native, payload, timestamp = self.read_components()
                    sample = {
                        "values": values,
                        "timestamp_ns": timestamp,
                        "native_values": native if self.config.record_native_values else None,
                        "native_payload": payload,
                    }
                except Exception as exc:
                    if self.stop.is_set():
                        break
                    error = exc
                    sample = {
                        "values": None,
                        "timestamp_ns": time.perf_counter_ns(),
                        "is_valid": False,
                        "status": "read_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self.client.close()
                    self.warn(exc)
                # Transport/publication failures must not be mistaken for device read failures.
                self.publish(sample, self.missed_deadlines)
                if error is not None and self.reconnect_required(error):
                    break
                deadline, skipped = advance_deadline(deadline, time.perf_counter_ns(), period)
                self.missed_deadlines += skipped
        finally:
            self.client.close()
