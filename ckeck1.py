import pandas as pd
df = pd.read_csv("data/raw/readings_2025-01-01.csv")
print("shape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\nnulls per column:\n", df.isna().sum())
print("\n'ERR' strings in temperature_c:", (df.temperature_c == "ERR").sum())
print("\nduplicate (device_id, ts) rows:", df.duplicated(["device_id","ts"]).sum())
print("\ntemperature_c min/max (non-ERR):",
      pd.to_numeric(df.temperature_c, errors="coerce").min(),
      pd.to_numeric(df.temperature_c, errors="coerce").max())