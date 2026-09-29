#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Compile a quantized HAR to a .hef for Hailo-8 (Hailo DFC Python API).

Compiler options (e.g. performance_param(compiler_optimization_level=max),
which searches longer for a faster allocation) must already be in the
model script -- pass them to optimize.py via --extra-line.

Example:
  python compile_hef.py --har classifier_q.har --out efficientnet_lite3_zeus_cropped_v2.hef
"""

import argparse

from hailo_sdk_client import ClientRunner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", required=True, help="quantized HAR from optimize.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runner = ClientRunner(har=args.har)
    hef = runner.compile()
    with open(args.out, "wb") as f:
        f.write(hef)
    runner.save_har(args.har.rsplit(".", 1)[0] + "_compiled.har")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
