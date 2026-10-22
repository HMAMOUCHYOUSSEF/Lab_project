"""Chunked, vectorized cleaning pipeline: raw CSV -> clean Parquet (+ quarantined anomalies)."""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
CHUNK = 100_000

STATUS_MAP = {"OK": 1, "WARN": 2, "FAULT": 3}
LIMITS = {"temperature_c": (-20, 80), "pressure_kpa": (90, 110), "vibration_g": (0, 2)}
SENSOR_ID = {"temperature_c": 1, "pressure_kpa": 2, "vibration_g": 3}
VALUE_COLS = list(LIMITS)
FINAL_COLS = ["device_id", "ts", "temperature_c", "pressure_kpa", "vibration_g", "status_id"]

# Memory optimization: smaller dtypes. temperature is read as text because of "ERR".
READ_DTYPES = {
    "device_id": "uint16",
    "temperature_c": "object",
    "pressure_kpa": "float32",
    "vibration_g": "float32",
    "status": "category",
}


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized type cleaning for one chunk."""
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%dT%H:%M:%S")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce").astype("float32")
    df["status_id"] = (
        df["status"].astype("object").map(STATUS_MAP).fillna(1).astype("uint8")
    )
    return df.drop(columns=["status"])


def split_outliers(df: pd.DataFrame):
    """Set out-of-range values to NaN; return the anomalies we removed (long format)."""
    anomalies = []
    for col, (lo, hi) in LIMITS.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if bad.any():
            a = df.loc[bad, ["device_id", "ts", col]].rename(columns={col: "value"})
            a["sensor_type_id"] = SENSOR_ID[col]
            anomalies.append(a)
            df.loc[bad, col] = np.nan
    return df, anomalies


def fill_short_gaps(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Forward-fill at most `limit` missing readings per device."""
    df = df.sort_values(["device_id", "ts"], kind="stable").reset_index(drop=True)
    df[VALUE_COLS] = df.groupby("device_id", sort=False)[VALUE_COLS].ffill(limit=limit)
    return df


def process_day(path, out_dir, chunksize: int = CHUNK) -> dict:
    t0 = time.time()
    path, out_dir = Path(path), Path(out_dir)
    day = path.stem.replace("readings_", "")
    (out_dir / "readings").mkdir(parents=True, exist_ok=True)
    (out_dir / "anomalies").mkdir(parents=True, exist_ok=True)

    frames, anomalies, rows_in = [], [], 0
    for chunk in pd.read_csv(path, dtype=READ_DTYPES, chunksize=chunksize):
        rows_in += len(chunk)
        chunk = clean_chunk(chunk)
        chunk, a = split_outliers(chunk)
        frames.append(chunk)
        anomalies.extend(a)

    df = pd.concat(frames, ignore_index=True)

    # Duplicates can be far apart (end of file), so de-duplicate after combining the day.
    before = len(df)
    df = df.drop_duplicates(subset=["device_id", "ts"], keep="first")
    dupes = before - len(df)

    nulls_before = int(df[VALUE_COLS].isna().sum().sum())
    df = fill_short_gaps(df, limit=3)
    nulls_after = int(df[VALUE_COLS].isna().sum().sum())

    df[FINAL_COLS].to_parquet(out_dir / "readings" / f"{day}.parquet",
                              index=False, compression="zstd")

    n_anom = 0
    if anomalies:
        a = pd.concat(anomalies, ignore_index=True)
        a = a.drop_duplicates(subset=["device_id", "ts", "sensor_type_id"])
        a["sensor_type_id"] = a["sensor_type_id"].astype("uint8")
        n_anom = len(a)
        a[["device_id", "ts", "sensor_type_id", "value"]].to_parquet(
            out_dir / "anomalies" / f"{day}.parquet", index=False, compression="zstd")

    return {
        "day": day, "rows_in": rows_in, "rows_out": len(df), "dupes_removed": dupes,
        "anomalies": n_anom, "nulls_filled": nulls_before - nulls_after,
        "seconds": round(time.time() - t0, 2), "skipped": False,
    }


def _skip_or_run(args):
    path, out_dir, force = args
    day = Path(path).stem.replace("readings_", "")
    if not force and (Path(out_dir) / "readings" / f"{day}.parquet").exists():
        return {"day": day, "skipped": True, "rows_in": 0, "rows_out": 0,
                "dupes_removed": 0, "anomalies": 0, "nulls_filled": 0, "seconds": 0}
    return process_day(path, out_dir)


def run(files, out_dir, workers: int = 1, force: bool = False):
    tasks = [(f, out_dir, force) for f in files]
    if workers <= 1:
        return [_skip_or_run(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_skip_or_run, tasks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first N files (0 = all)")
    args = ap.parse_args()

    files = sorted(RAW.glob("readings_*.csv"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No raw files found. Run Phase 1 first.")

    t0 = time.time()
    stats = run(files, OUT, args.workers, args.force)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_stats.jsonl", "a") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")

    done = [s for s in stats if not s["skipped"]]
    print(f"Processed {len(done)} files ({len(stats)-len(done)} skipped) "
          f"in {time.time()-t0:.1f}s with {args.workers} workers")
    print(f"rows_in={sum(s['rows_in'] for s in done):,}  "
          f"rows_out={sum(s['rows_out'] for s in done):,}  "
          f"dupes_removed={sum(s['dupes_removed'] for s in done):,}  "
          f"anomalies={sum(s['anomalies'] for s in done):,}")


if __name__ == "__main__":
    main()