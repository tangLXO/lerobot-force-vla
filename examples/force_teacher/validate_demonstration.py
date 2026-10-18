#!/usr/bin/env python
"""Verify saved demonstration files without opening hardware or a writer."""

import argparse
import json

from lerobot.force_control.demonstration_validation import validate_demonstration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Local demonstration Dataset directory")
    parser.add_argument("--episode-uid")
    parser.add_argument("--skip-video-decode", action="store_true")
    args = parser.parse_args()
    report = validate_demonstration(args.root, args.episode_uid, verify_video=not args.skip_video_decode)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
