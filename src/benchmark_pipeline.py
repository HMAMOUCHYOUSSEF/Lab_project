"""Benchmark: naive row-loop pipeline vs optimized pipeline. Extrapolates to weekly/500 GB."""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline import RAW, LIMITS, run

BENCH_NAIVE = Path("data/bench_naive")
BENCH_OPT = Path("data/bench_opt")


def _valid(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v != v or v < lo or v > hi:   # NaN or out of range
        return None
    return v


def naive_process_day(path, out_path):
    """The 'before': loads everything, loops row by row, default dtypes."""
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        out.append({
            "device_id": int(r["device_id"]),
            "ts": datetime.fromisoformat(r["ts"]),
            "temperature_c": _valid(r["temperature_c"], *LIMITS["temperature_c"]),
            "pressure_kpa": _valid(r["pressure_kpa"], *LIMITS["pressure_kpa"]),
            "vibration_g": _valid(r["vibration_g"], *LIMITS["vibration_g"]),
            "status": r["status"],
        })
    res = pd.DataFrame(out).drop_duplicates(["device_id", "ts"])
    res.to_csv(out_path, index=False)


def memory_demo(path):
    df_default = pd.read_csv(path)
    from pipeline import READ_DTYPES
    df_opt = pd.read_csv(path, dtype=READ_DTYPES)
    a = df_default.memory_usage(deep=True).sum() / 1e6
    b = df_opt.memory_usage(deep=True).sum() / 1e6
    print(f"\nMemory for 1 day: default dtypes {a:.1f} MB -> optimized {b:.1f} MB "
          f"({100*(1-b/a):.0f}% smaller)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2, help="how many raw days to benchmark")
    ap.add_argument("--weekly-gb", type=float, default=25.0, help="assumed new data per week")
    args = ap.parse_args()

    files = sorted(RAW.glob("readings_*.csv"))[: args.days]
    gb = sum(f.stat().st_size for f in files) / 1e9
    BENCH_NAIVE.mkdir(parents=True, exist_ok=True)

    t = time.time()
    for f in files:
        naive_process_day(f, BENCH_NAIVE / f.name)
    naive_s = time.time() - t

    t = time.time()
    run(files, BENCH_OPT, workers=1, force=True)
    opt1_s = time.time() - t

    workers = max(1, (os.cpu_count() or 2) - 1)
    t = time.time()
    run(files, BENCH_OPT, workers=workers, force=True)
    optN_s = time.time() - t

    def hours_for(sec_per_gb, gb_total):
        return sec_per_gb * gb_total / 3600

    n_spg, o1_spg, oN_spg = naive_s / gb, opt1_s / gb, optN_s / gb
    print(f"\nBenchmarked {len(files)} files = {gb*1000:.0f} MB")
    print(f"{'method':<28}{'seconds':>10}{'sec/GB':>10}{'weekly ({:.0f} GB)'.format(args.weekly_gb):>20}{'500 GB total':>16}")
    for name, s, spg in [("naive (iterrows)", naive_s, n_spg),
                         ("optimized, 1 process", opt1_s, o1_spg),
                         (f"optimized, {workers} processes", optN_s, oN_spg)]:
        w = hours_for(spg, args.weekly_gb)
        tot = hours_for(spg, 500)
        print(f"{name:<28}{s:>10.1f}{spg:>10.0f}{w:>17.2f} h{tot:>13.1f} h")

    print(f"\nSpeed-up (naive -> optimized, {workers} procs): {naive_s/optN_s:.1f}x")
    print(f"Reduction: {100*(1-optN_s/naive_s):.1f}%")
    print("NOTE: weekly/500 GB numbers are EXTRAPOLATIONS from this small sample.")
    memory_demo(files[0])


if __name__ == "__main__":
    main()