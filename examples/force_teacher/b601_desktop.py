"""B601 Qt operation workspace. Startup loads configuration without connecting devices."""

import argparse
import gc
import sys
from pathlib import Path

from communication_options import communication_description, resolve_command_tx_gap

from lerobot.utils.import_utils import require_package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Build and close the desktop without device I/O"
    )
    parser.add_argument("--connect", action="store_true", help="Explicitly start query-only monitoring")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--gripper-only", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Synthetic offline UI; never opens hardware")
    parser.add_argument("--legacy-ui", action="store_true", help="Open the original Tk panel")
    parser.add_argument("--recording-config", type=Path)
    parser.add_argument(
        "--command-feedback-mode",
        choices=("sequential", "pipelined", "cached"),
        default="cached",
        help="Startup-only seven-axis communication mode; daily capture defaults to cached",
    )
    parser.add_argument(
        "--command-tx-gap-us",
        type=float,
        default=None,
        help="Minimum asynchronous write-start interval: CLI > MOTORBRIDGE_TX_GAP_US > 120 us",
    )
    args = parser.parse_args(argv)
    try:
        gap, gap_source = resolve_command_tx_gap(args.command_tx_gap_us)
    except ValueError as exc:
        parser.error(str(exc))
    if args.legacy_ui:
        if args.demo:
            parser.error("--demo requires Qt")
        from b601_tuner import main as legacy_main

        forwarded = []
        for flag in ("smoke_test", "connect", "gripper_only"):
            if getattr(args, flag):
                forwarded.append("--" + flag.replace("_", "-"))
        for flag in (
            "calibration_file",
            "settings_file",
            "recording_config",
            "command_feedback_mode",
            "command_tx_gap_us",
        ):
            if getattr(args, flag) is not None:
                forwarded.extend(("--" + flag.replace("_", "-"), str(getattr(args, flag))))
        return legacy_main(forwarded, resolved_tx_gap=(gap, gap_source))
    print(
        communication_description(args.command_feedback_mode, gap, gap_source, gripper_only=args.gripper_only)
    )
    try:
        require_package("PySide6", "force-teacher-ui")
        require_package("pyqtgraph", "force-teacher-ui")
    except ImportError as exc:
        parser.exit(
            2, f"{exc}\n项目内安装：uv sync --extra force-teacher-ui\n可使用 --legacy-ui 打开原面板。\n"
        )
    from b601_session import SessionController
    from desktop.window import DesktopWindow
    from PySide6 import QtCore, QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("B601Desktop")
    application.setOrganizationName("LeRobotLocal")
    controller = SessionController(
        settings_path=args.settings_file,
        calibration_path=args.calibration_file,
        gripper_only=args.gripper_only,
        demo=args.demo,
        command_feedback_mode=args.command_feedback_mode,
        command_tx_gap_us=gap,
        command_tx_gap_source=gap_source,
    )
    if args.recording_config is not None:
        controller.prefill_recording(args.recording_config)
    window = DesktopWindow(controller, demo=args.demo, gripper_only=args.gripper_only)
    application.aboutToQuit.connect(lambda: None if controller.closed else controller.close())
    controller.start()
    window.show()
    if args.smoke_test:
        QtCore.QTimer.singleShot(350, window.close)
    elif args.connect:
        QtCore.QTimer.singleShot(200, lambda target=window: target.dispatch("connect"))
    try:
        exit_code = application.exec()
    finally:
        if not controller.closed:
            controller.close()
        while not controller.closed:
            application.processEvents()
            QtCore.QThread.msleep(10)
        window.close()
        # QApplication must outlive every QWidget. In a function-local entry
        # point, Python may otherwise destroy it before signal-bound widgets.
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        application.processEvents()
        del window
        gc.collect()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
