#!/usr/bin/env python
"""Verify saved demonstration files without opening hardware or a writer."""

import argparse
import json
import time

from lerobot.force_control.demonstration_validation import validate_demonstration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Local demonstration Dataset directory")
    parser.add_argument("--episode-uid")
    parser.add_argument("--skip-video-decode", action="store_true")
    parser.add_argument(
        "--progress-jsonl", action="store_true", help="Emit progress and final result as JSON lines"
    )
    args = parser.parse_args()

    def progress(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    report = validate_demonstration(
        args.root,
        args.episode_uid,
        verify_video=not args.skip_video_decode,
        progress=progress if args.progress_jsonl else None,
    )
    if args.progress_jsonl:
        print(
            json.dumps(
                {"event": "result", "report": report, "elapsed_seconds": time.perf_counter() - started},
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
