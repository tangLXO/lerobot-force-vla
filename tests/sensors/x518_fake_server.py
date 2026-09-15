# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Local Modbus fixture exercised through real sockets and spawned acquisition."""

import contextlib
import socket
import struct
import threading
import time


class ModbusServer:
    """Serve real X518 reads; expose explicit failure and response-delay controls."""

    def __init__(self):
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.port = self.socket.getsockname()[1]
        self.socket.listen()
        self.socket.settimeout(0.05)
        self.stop = threading.Event()
        self.fail = threading.Event()
        self.hold = threading.Event()
        self.hold_settings = threading.Event()
        self.settings_started = threading.Event()
        self.read_started = threading.Event()
        self.connections = 0
        self.responses = 0
        self.unit = 5
        self.delay = 0
        self.client = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.socket.close()
        if self.client is not None:
            with contextlib.suppress(OSError):
                self.client.shutdown(socket.SHUT_RDWR)
                self.client.close()
        self.thread.join(2)
        assert not self.thread.is_alive()

    def receive(self, client, size):
        data = b""
        while len(data) < size and not self.stop.is_set():
            chunk = client.recv(size - len(data))
            if not chunk:
                raise ConnectionError("closed")
            data += chunk
        return data

    def run(self):
        while not self.stop.is_set():
            try:
                client, _ = self.socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            self.client = client
            client.settimeout(1)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with client:
                try:
                    while not self.stop.is_set():
                        header = self.receive(client, 7)
                        transaction, protocol, length, unit = struct.unpack(">HHHB", header)
                        body = self.receive(client, length - 1)
                        function, address, count = struct.unpack(">BHH", body)
                        assert protocol == 0 and function == 3
                        if address == 0x0A00:
                            self.read_started.set()
                            while self.hold.is_set() and not self.stop.wait(0.005):
                                pass
                            if self.fail.is_set():
                                break
                            if self.delay:
                                time.sleep(self.delay)
                            self.responses += 1
                            payload = struct.pack(">ii", self.responses, -self.responses)
                        else:
                            self.settings_started.set()
                            while self.hold_settings.is_set() and not self.stop.wait(0.005):
                                pass
                            settings = {
                                0x0614: self.unit,
                                0x0616: 0,
                                0x061A: 6,
                                0x061C: 0,
                                0x0638: 0,
                                0x0644: 1,
                            }
                            registers = []
                            for current in range(address, address + count):
                                registers.append(settings.get(current - 1, 0) if current % 2 else 0)
                            payload = struct.pack(f">{count}H", *registers)
                        response = struct.pack(">BB", 3, len(payload)) + payload
                        client.sendall(
                            struct.pack(">HHHB", transaction, 0, len(response) + 1, unit) + response
                        )
                except (OSError, ConnectionError, struct.error):
                    pass
            self.client = None
