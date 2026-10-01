#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Compile quantized HAR(s) to .hef for Hailo-8 (Hailo DFC Python API).

A single compile is serial inside the Hailo compiler, so extra cores only help
when several models are compiled at once. Pass --har/--out multiple times to
compile them in parallel; the worker count is chosen automatically from CPU
count and available RAM (override with --jobs).

Compiler options (e.g. performance_param(compiler_optimization_level=max),
which searches longer for a faster allocation) must already be in the
model script -- pass them to optimize.py via --extra-line.

Example:
  python compile_hef.py --har classifier_q.har --out efficientnet_lite3_zeus_cropped_v2.hef
  python compile_hef.py --har a.har --out a.hef --har b.har --out b.hef
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Peak per-compile footprint observed: ~2.3 GB compiler + ~1.2 GB python.
GB_PER_JOB = 3.5


def available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def auto_jobs(n_tasks):
    by_ram = int(available_gb() // GB_PER_JOB)
    return max(1, min(n_tasks, os.cpu_count() or 1, by_ram))


def compile_one(har, out):
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=har)
    hef = runner.compile()
    with open(out, "wb") as f:
        f.write(hef)
    runner.save_har(har.rsplit(".", 1)[0] + "_compiled.har")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", required=True, action="append", help="quantized HAR from optimize.py (repeatable)")
    ap.add_argument("--out", required=True, action="append", help="output .hef (repeatable, one per --har)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel compiles (default: auto from CPU/RAM)")
    args = ap.parse_args()

    if len(args.har) != len(args.out):
        ap.error("need the same number of --har and --out")

    tasks = list(zip(args.har, args.out))
    jobs = args.jobs or auto_jobs(len(tasks))
    print(f"Compiling {len(tasks)} model(s) with {jobs} job(s) "
          f"({os.cpu_count()} cores, {available_gb():.1f} GB RAM available)")

    if jobs == 1:
        for har, out in tasks:
            print(f"Wrote {compile_one(har, out)}")
        return

    # spawn: don't fork a process that may already hold TF/Hailo state
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(compile_one, har, out): out for har, out in tasks}
        failed = []
        for fut in as_completed(futs):
            try:
                print(f"Wrote {fut.result()}")
            except Exception as e:
                failed.append(futs[fut])
                print(f"FAILED {futs[fut]}: {e!r}")
    if failed:
        raise SystemExit(f"{len(failed)} compile(s) failed: {failed}")


if __name__ == "__main__":
    main()
