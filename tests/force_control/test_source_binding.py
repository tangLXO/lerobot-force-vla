"""The shared venv and Windows-spawn child must resolve this checkout's source."""

import multiprocessing
from pathlib import Path

import pytest

from lerobot.force_control.source_binding import require_source_tree


def child_source(queue, root):
    queue.put(require_source_tree(root))


def test_parent_and_spawn_child_import_the_capture_worktree():
    root = Path(__file__).resolve().parents[2]
    expected = require_source_tree(root)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue(1)
    process = context.Process(target=child_source, args=(queue, root))
    try:
        process.start()
        assert queue.get(timeout=30) == expected
        process.join(10)
        assert process.exitcode == 0
    finally:
        queue.close()
        queue.cancel_join_thread()
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_wrong_checkout_is_rejected_before_io(tmp_path):
    with pytest.raises(RuntimeError, match="Source checkout mismatch"):
        require_source_tree(tmp_path)
