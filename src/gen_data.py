"""Generate synthetic sensor data: one CSV per day + devices.csv.

Examples:
  python src/gen_data.py --days 7              # quick test (~80 MB)
  python src/gen_data.py --days 90 --devices 20   # full project (~1 GB)
  python src/gen_data.py --days 90 --devices 200  # ~10 GB, if you want more
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

DATA = Path("data")
RAW = DATA / "raw"


def make_devices(n_devices: int, seed: int) -> pd.DataFrame:
    Faker.seed(seed)
    fake = Faker()
    rng = np.random.default_rng(seed)

    sites = []
    while len(sites) < 5:
        city = f"{fake.city()} Lab"
        if city not in sites:
            sites.append(city)

    types = ["thermo-pressure", "vibration-node", "multi-sensor"]
    firmware = ["1.0.3", "1.1.0", "1.2.4", "2.0.1"]

    rows = []
    for i in range(1, n_devices + 1):
        rows.append({
            "device_id": i,
            "site": sites[(i - 1) % len(sites)],
            "device_type": types[int(rng.integers(0, len(types)))],
            "firmware": firmware[int(rng.integers(0, len(firmware)))],
            "install_date": fake.date_between(start_date="-3y", end_date="-60d").isoformat(),
            "base_temp": round(float(rng.uniform(15, 35)), 2),   # used only by the generator
            "phase": round(float(rng.uniform(0, 2 * np.pi)), 3),  # used only by the generator
        })
    return pd.DataFrame(rows)


def add_outliers(arr, rng, p, lo, hi, signed):
    mask = rng.random(arr.size) < p
    k = int(mask.sum())
    mag = rng.uniform(lo, hi, k)
    if signed:
        mag = mag * rng.choice([-1.0, 1.0], k)
    arr[mask] += mag


def make_day(day: str, devs: pd.DataFrame, interval: int, seed: int,
             p_missing=0.02, p_outlier=0.001, p_dup=0.0005, p_err=0.002) -> pd.DataFrame:
    """Return one day of readings for all devices as a DataFrame (dirty on purpose)."""
    rng = np.random.default_rng([seed, int(day.replace("-", ""))])
    n_dev = len(devs)
    steps = 86400 // interval
    n = n_dev * steps

    sec = np.tile(np.arange(steps, dtype=np.int64) * interval, n_dev)
    dev_idx = np.repeat(np.arange(n_dev), steps)
    base = devs["base_temp"].to_numpy()[dev_idx]
    phase = devs["phase"].to_numpy()[dev_idx]
    angle = 2 * np.pi * sec / 86400

    temp = base + 5 * np.sin(angle - phase) + rng.normal(0, 0.3, n)
    press = 101.3 + 1.5 * np.sin(angle / 2 + phase) + rng.normal(0, 0.2, n)
    vib = rng.gamma(2.0, 0.05, n)
    vib = vib * (1 + 0.8 * ((sec > 8 * 3600) & (sec < 18 * 3600)))  # busier in work hours

    # Outliers (impossible values)
    add_outliers(temp, rng, p_outlier, 80, 150, signed=True)
    add_outliers(press, rng, p_outlier, 30, 60, signed=True)
    add_outliers(vib, rng, p_outlier, 3, 8, signed=False)

    # Missing values
    for arr in (temp, press, vib):
        arr[rng.random(n) < p_missing] = np.nan

    status = rng.choice(["OK", "WARN", "FAULT"], n, p=[0.97, 0.025, 0.005])

    ts = (np.datetime64(day) + sec.astype("timedelta64[s]")).astype("datetime64[s]").astype(str)

    temp_s = pd.Series(np.round(temp, 2)).astype(object)
    temp_s.loc[rng.random(n) < p_err] = "ERR"   # text in a numeric column

    df = pd.DataFrame({
        "device_id": dev_idx + 1,
        "ts": ts,
        "temperature_c": temp_s,
        "pressure_kpa": np.round(press, 2),
        "vibration_g": np.round(vib, 3),
        "status": status,
    })

    # Time-major order (like a real logger), then append duplicates at the END of the file
    df = df.iloc[np.argsort(sec, kind="stable")].reset_index(drop=True)
    dup_idx = np.flatnonzero(rng.random(n) < p_dup)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--devices", type=int, default=20)
    ap.add_argument("--interval", type=int, default=10, help="seconds between readings")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    devs = make_devices(args.devices, args.seed)
    devs.to_csv(DATA / "devices.csv", index=False)

    t0 = time.time()
    total_bytes = 0
    total_rows = 0
    for d in pd.date_range(args.start, periods=args.days):
        day = d.strftime("%Y-%m-%d")
        path = RAW / f"readings_{day}.csv"
        if path.exists():
            total_bytes += path.stat().st_size
            continue
        df = make_day(day, devs, args.interval, args.seed)
        df.to_csv(path, index=False)
        total_bytes += path.stat().st_size
        total_rows += len(df)
        print(f"{day}: {len(df):,} rows, {path.stat().st_size/1e6:.1f} MB")

    print(f"\nDone in {time.time()-t0:.0f}s. New rows: {total_rows:,}. "
          f"Total raw size: {total_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()