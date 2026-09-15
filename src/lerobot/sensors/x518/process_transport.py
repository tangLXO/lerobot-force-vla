# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Bounded spawn transport for X518; no Robot or Dataset objects cross processes."""

import contextlib
import multiprocessing
import os
import queue
import threading
from multiprocessing.connection import wait

from .acquisition import X518Acquisition


def acquisition_worker(config, samples, commands, status, stop, protected):
    """Own acquisition in a spawned process and serialize barriers with sample messages."""
    sequence = 0
    peak = 0
    overflow = 0
    parent = multiprocessing.parent_process()

    def report(kind, payload):
        status.send((kind, payload))

    def enqueue(message):
        nonlocal peak, overflow
        try:
            samples.put_nowait(message)
            with contextlib.suppress(NotImplementedError):
                peak = max(peak, samples.qsize())
        except queue.Full as exc:
            overflow += 1
            raise RuntimeError("X518 acquisition IPC queue overflow; samples are incomplete.") from exc

    def publish(sample, missed):
        nonlocal sequence
        attempt = sequence
        sequence += 1
        enqueue(("sample", (attempt, sample, missed, peak)))

    def settings_changed(settings, rate):
        enqueue(("settings", (settings, rate)))

    def reconnect_required(error):
        if protected.is_set():
            raise RuntimeError(f"Required Sensor needs reconnect: {error}")
        return False

    def service():
        if engine.reconnecting and protected.is_set():
            reconnect_required(ConnectionError("X518 reconnect overlaps recorder startup"))
        while commands.poll():
            token = commands.recv()
            enqueue(("barrier", token))

    engine = X518Acquisition(config, stop, publish, settings_changed, reconnect_required, service)
    watch_done = threading.Event()

    def watch_owner():
        # The parent sentinel also works after an abrupt os._exit()/TerminateProcess.
        while not watch_done.wait(0.05):
            if parent is not None and wait([parent.sentinel], timeout=0):
                engine.client.close()
                os._exit(1)
            if stop.is_set():
                engine.client.close()  # interrupt an outstanding recv during normal shutdown

    watchdog = threading.Thread(target=watch_owner, name="X518OwnerWatch", daemon=True)
    watchdog.start()
    clean = False
    try:
        settings = engine.connect_initially()
        report("ready", (settings, engine.sample_rate_hz))
        engine.run()
        enqueue(("stopped", (sequence, engine.missed_deadlines)))
        clean = True
    except BaseException as exc:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            report("error", (type(exc).__name__, str(exc), overflow))
    finally:
        stop.set()
        engine.client.close()
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            report(
                "finished",
                {
                    "ipc_peak_size": peak,
                    "ipc_overflow_count": overflow,
                    "acquisition_attempts": sequence,
                    "missed_deadlines": engine.missed_deadlines,
                },
            )
        commands.close()
        status.close()
        if not clean:
            # A failed episode cannot commit. Do not hang in Queue's feeder finalizer
            # when its reader has died, or try to reuse the damaged channel.
            samples.cancel_join_thread()
        samples.close()
        if clean:
            samples.join_thread()
        watch_done.set()
        watchdog.join(timeout=1)
