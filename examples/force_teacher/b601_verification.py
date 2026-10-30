"""Run read-only demonstration verification in a cancellable child process."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


class VerificationWorker:
    """Drain child output without blocking the desktop supervisor or Qt thread."""

    def __init__(self, project: Path, root: Path):
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.output: list[str] = []
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = os.pathsep.join((str(project / "src"), str(project), env.get("PYTHONPATH", "")))
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(project / "examples/force_teacher/validate_demonstration.py"),
                str(root),
                "--progress-jsonl",
            ],
            cwd=project,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.thread = threading.Thread(target=self._read, name="teacher-verification-output", daemon=False)
        try:
            self.thread.start()
        except BaseException:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            finally:
                self.process.stdout.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    self.output = (self.output + [line.strip()])[-8:]
                    continue
                if isinstance(event, dict) and event.get("event") in {"episode", "video", "result"}:
                    self.events.put(event)
        finally:
            self.process.stdout.close()

    def poll(self):
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                break
        return events

    @property
    def finished(self):
        return self.process.poll() is not None and not self.thread.is_alive()

    def cancel(self):
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except OSError:
                if self.process.poll() is None:
                    raise
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        if not self.finished:
            raise RuntimeError("核验进程尚未退出")

    def close(self):
        if not self.finished:
            self.cancel()
        self.thread.join(timeout=0)
