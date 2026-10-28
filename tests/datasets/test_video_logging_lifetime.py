"""Video workers must not replace process-global logging or retain native stdio."""

import logging
import os
import queue
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

av = pytest.importorskip("av")

from lerobot.configs.video import RGBEncoderConfig  # noqa: E402
from lerobot.datasets import video_utils  # noqa: E402
from lerobot.datasets.video_utils import (  # noqa: E402
    StreamingVideoEncoder,
    _CameraEncoderThread,
    _prepare_native_svt_logging,
    encode_video_frames,
    get_video_info,
    reencode_video,
)


@pytest.fixture
def caller_logging(monkeypatch, request):
    """Fail deterministically if a helper overwrites the caller's PyAV policy."""
    libav_logger = logging.getLogger("libav")
    previous = libav_logger.level
    libav_logger.setLevel(logging.DEBUG)
    previous_av_level = av.logging.get_level()
    set_level = av.logging.set_level
    level = getattr(request, "param", previous_av_level)
    set_level(level)

    def forbidden(*args, **kwargs):
        raise AssertionError("Video helper replaced the process-global PyAV callback")

    monkeypatch.setattr(av.logging, "restore_default_callback", forbidden)
    monkeypatch.setattr(av.logging, "set_level", forbidden)
    try:
        yield libav_logger, level
        assert libav_logger.level == logging.DEBUG
        assert av.logging.get_level() == level
    finally:
        libav_logger.setLevel(previous)
        set_level(previous_av_level)


def encoder():
    return StreamingVideoEncoder(
        30,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="veryfast", g=2),
        encoder_threads=2,
        strict_integrity=True,
    )


def test_workers_and_probes_preserve_logging_across_repeated_codec_teardown(tmp_path, caller_logging):
    worker = encoder()
    keys = ["front", "wrist"]
    try:
        for _ in range(2):
            worker.start_episode(keys, tmp_path, frame_sizes=dict.fromkeys(keys, (16, 16)))
            worker.wait_until_ready()
            for _ in range(3):
                for key in keys:
                    worker.feed_frame(key, np.zeros((16, 16, 3), np.uint8))
            results = worker.finish_episode(expected_frames=3)
            assert not worker._threads
            for path, _ in results.values():
                info = get_video_info(path)
                assert info["video.codec"] == "h264" and info["has_audio"] is False
                with av.open(str(path)) as source:
                    assert sum(1 for _ in source.decode(video=0)) == 3
    finally:
        worker.close()


@pytest.mark.parametrize("cancel_at_open", [False, True])
def test_cancelled_initialization_never_starts_a_native_codec(tmp_path, monkeypatch, cancel_at_open):
    stopped = threading.Event()
    worker = _CameraEncoderThread(
        tmp_path / "cancelled.mp4",
        30,
        RGBEncoderConfig(vcodec="h264"),
        queue.Queue(),
        queue.Queue(),
        stopped,
    )
    container = Mock()
    container.add_stream.return_value = SimpleNamespace()

    def open_container(*args, **kwargs):
        stopped.set()
        return container

    opening = Mock(side_effect=open_container)
    monkeypatch.setattr(av, "open", opening)
    if not cancel_at_open:
        stopped.set()
    with pytest.raises(RuntimeError, match="initialization cancelled"):
        worker._open_container(16, 16)
    container.start_encoding.assert_not_called()
    if cancel_at_open:
        container.close.assert_called_once()
    else:
        opening.assert_not_called()


def test_worker_error_remains_explicit_without_changing_logging(tmp_path, caller_logging):
    worker = encoder()
    try:
        worker.start_episode(["front"], tmp_path, frame_sizes={"front": (16, 16)})
        worker.wait_until_ready()
        thread = worker._threads["front"]
        worker.feed_frame("front", np.zeros((18, 16, 3), np.uint8))
        with pytest.raises(RuntimeError, match="Frame dimensions do not match"):
            worker.finish_episode(expected_frames=1)
        assert not thread.is_alive()
        assert isinstance(thread.error, ValueError)
        assert worker._integrity_error is not None
    finally:
        worker.close()


@pytest.mark.parametrize("caller_logging", [av.logging.ERROR], indirect=True)
def test_native_codec_error_retains_callers_captured_diagnostics(tmp_path, caller_logging):
    worker = encoder()
    with av.logging.Capture(local=False) as records:
        worker.start_episode(["front"], tmp_path, frame_sizes={"front": (16, 15)})
        thread = worker._threads["front"]
        try:
            with pytest.raises(RuntimeError, match="Encoder preparation failed") as caught:
                worker.wait_until_ready()
            assert caught.value.__cause__ is thread.error
            assert isinstance(thread.error, av.FFmpegError)
            assert thread.error.errno != 0
        finally:
            worker.close()
    assert any("width" in message and "divisible" in message for _, _, message in records)


@pytest.mark.parametrize("operation", ["encode", "reencode"])
def test_explicit_python_log_level_restores_previous_level_on_failure(
    tmp_path, monkeypatch, caller_logging, operation
):
    def failed_open(*args, **kwargs):
        assert logging.getLogger("libav").level == logging.ERROR
        raise RuntimeError("injected container open failure")

    monkeypatch.setattr(av, "open", failed_open)
    if operation == "encode":
        from PIL import Image

        images = tmp_path / "images"
        images.mkdir()
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(images / "frame-000000.png")
        with pytest.raises(RuntimeError, match="injected container open failure"):
            encode_video_frames(images, tmp_path / "out.mp4", 30, log_level=logging.ERROR)
    else:
        with pytest.raises(RuntimeError, match="injected container open failure"):
            reencode_video(tmp_path / "input.mp4", tmp_path / "out.mp4", log_level=logging.ERROR)


@pytest.mark.parametrize("vcodec", ["h264", "libsvtav1"])
def test_real_codecs_survive_native_descriptor_capture_rotation(tmp_path, vcodec):
    """Exercise the Windows crash in a child so a native failure retains its log."""
    script = tmp_path / "capture_child.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from _pytest.capture import FDCapture, MultiCapture

            capture = MultiCapture(out=FDCapture(1), err=FDCapture(2), in_=None)
            capture.start_capturing()
            import av
            import numpy as np
            from lerobot.configs.video import RGBEncoderConfig
            from lerobot.datasets.video_utils import StreamingVideoEncoder, get_video_info
            capture.suspend_capturing()
            root = Path(sys.argv[1])
            root.mkdir()
            vcodec = sys.argv[2]
            size = (64, 96) if vcodec == "libsvtav1" else (16, 16)
            encoder = StreamingVideoEncoder(
                30, rgb_encoder=RGBEncoderConfig(
                    vcodec=vcodec, preset=12 if vcodec == "libsvtav1" else "veryfast", g=2,
                ),
                encoder_threads=2, strict_integrity=True,
            )
            keys = ["front", "wrist"]
            try:
                for episode in range(40):
                    capture.resume_capturing()
                    encoder.start_episode(keys, root, frame_sizes=dict.fromkeys(keys, size))
                    encoder.wait_until_ready()
                    for index in range(3):
                        for key in keys:
                            encoder.feed_frame(key, np.zeros((*size, 3), np.uint8))
                    results = encoder.finish_episode(expected_frames=3)
                    for path, _ in results.values():
                        assert get_video_info(path)["has_audio"] is False
                        with av.open(str(path)) as source:
                            assert sum(1 for _ in source.decode(video=0)) == 3
                    capture.suspend_capturing()
            finally:
                encoder.close()
                capture.stop_capturing()
            for path in encoder.native_log_paths:
                print("NATIVE_LOG", path, flush=True)
            print("COMPLETED 40 dual-camera episodes", flush=True)
            """
        ),
        encoding="utf-8",
    )
    log = tmp_path / "capture_child.log"
    with log.open("wb") as output:
        child = subprocess.run(
            [sys.executable, "-X", "faulthandler", str(script), str(tmp_path / "videos"), vcodec],
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    contents = log.read_text(encoding="utf-8", errors="replace")
    assert child.returncode == 0, (
        f"Child exited {child.returncode:#x}; retained log {log}:\n{contents[-4000:]}"
    )
    assert "COMPLETED 40 dual-camera episodes" in contents
    if os.name == "nt" and vcodec == "libsvtav1":
        paths = [
            line.removeprefix("NATIVE_LOG ")
            for line in contents.splitlines()
            if line.startswith("NATIVE_LOG ")
        ]
        assert paths
        assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths)


@pytest.mark.skipif(os.name != "nt", reason="Legacy CRT diagnostic ownership is Windows-specific")
def test_native_log_ownership_survives_caller_policy_and_spawn(tmp_path):
    grandchild = textwrap.dedent(
        """
        import json,os
        import numpy as np
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.video_utils import StreamingVideoEncoder, _prepare_native_svt_logging
        path = _prepare_native_svt_logging("libsvtav1")
        encoder = StreamingVideoEncoder(30, rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1",preset=12), encoder_threads=2)
        try:
            encoder.start_episode(["front"], path.parent, frame_sizes={"front":(64,96)})
            encoder.wait_until_ready()
            encoder.feed_frame("front",np.zeros((64,96,3),np.uint8))
            encoder.finish_episode()
        finally:
            encoder.close()
        print(json.dumps({"pid":os.getpid(),"path":str(path)}),flush=True)
        """
    )
    script = textwrap.dedent(
        """
        import json,os,sys,subprocess
        from pathlib import Path
        from lerobot.datasets.video_utils import _prepare_native_svt_logging
        path = _prepare_native_svt_logging("libsvtav1")
        path.write_text("retained parent diagnostic",encoding="utf-8")
        assert _prepare_native_svt_logging("libsvtav1") == path
        child = subprocess.run([sys.executable,"-X","faulthandler","-c",sys.argv[1]],capture_output=True,text=True,timeout=60)
        assert child.returncode == 0,child.stderr
        result = json.loads(child.stdout.splitlines()[-1])
        assert result["pid"] != os.getpid() and result["path"] != str(path)
        assert path.read_text(encoding="utf-8") == "retained parent diagnostic"
        assert Path(result["path"]).stat().st_size > 0
        caller = Path(sys.argv[2])
        os.environ["SVT_LOG_FILE"] = str(caller)
        try:
            _prepare_native_svt_logging("libsvtav1")
        except RuntimeError as error:
            assert "new process" in str(error)
        else:
            raise AssertionError("Cannot advertise a new native log destination mid-process")
        policy = 'import os; from pathlib import Path; from lerobot.datasets.video_utils import _prepare_native_svt_logging; assert _prepare_native_svt_logging("libsvtav1") == Path(os.environ["SVT_LOG_FILE"]).absolute()'
        checked = subprocess.run([sys.executable,"-c",policy],capture_output=True,text=True,timeout=60)
        assert checked.returncode == 0,checked.stderr
        assert path.is_file()
        print("OWNERSHIP_AND_CALLER_POLICY_PASSED",flush=True)
        """
    )
    environment = dict(os.environ)
    environment.pop("SVT_LOG_FILE", None)
    environment.pop("_LEROBOT_SVT_LOG_FILE_OWNER", None)
    log = tmp_path / "ownership_child.log"
    with log.open("wb") as output:
        child = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-c",
                script,
                grandchild,
                str(tmp_path / "caller-svt.log"),
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=120,
            env=environment,
        )
    contents = log.read_text(encoding="utf-8", errors="replace")
    assert child.returncode == 0, contents[-4000:]
    assert "OWNERSHIP_AND_CALLER_POLICY_PASSED" in contents


@pytest.mark.skipif(os.name != "nt", reason="Legacy CRT diagnostic ownership is Windows-specific")
def test_ready_native_log_does_not_mutate_environment_again(monkeypatch):
    path = _prepare_native_svt_logging("libsvtav1")
    original_crt = video_utils.ctypes.CDLL("msvcrt")

    def forbidden(*args):
        raise AssertionError("A matching native environment must not be mutated while codecs can be active")

    proxy = SimpleNamespace(getenv=original_crt.getenv, _putenv_s=forbidden)
    monkeypatch.setattr(video_utils.ctypes, "CDLL", lambda name: proxy)
    assert _prepare_native_svt_logging("libsvtav1") == path
