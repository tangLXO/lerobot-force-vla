#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Read-only verification of an independent Force Teacher bench run."""

import argparse
import json

from lerobot.force_control.trace import validate_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Existing teacher_bench_schema_v1 run directory")
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="Forensic check, not training acceptance"
    )
    parser.add_argument(
        "--band-N", dest="band_n", type=float, help="Explicit calibrated absolute error band in N"
    )
    parser.add_argument("--recovery-dwell-s", type=float, help="Required sampled dwell in the chosen band")
    args = parser.parse_args()
    report = validate_trace(
        args.root,
        require_complete=not args.allow_incomplete,
        band_n=args.band_n,
        recovery_dwell_s=args.recovery_dwell_s,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
