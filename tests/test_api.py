import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/processed/actuals.csv")

# Read original streaming source
source = pd.read_csv(ROOT/"data/raw/amprion.csv")

# Read actuals stored by predictor
actuals = pd.read_csv(ROOT/"data/processed/actuals.csv")


def parse_time(s):
    s = s.astype("string").str.strip()

    t = pd.to_datetime(
        s,
        format="%d-%m-%Y %H:%M",
        errors="coerce",
        utc=True
    )

    mask = t.isna() & s.notna()

    if mask.any():
        t.loc[mask] = pd.to_datetime(
            s.loc[mask],
            errors="coerce",
            utc=True
        )

    return t


source["utc_timestamp"] = parse_time(source["utc_timestamp"])
actuals["utc_timestamp"] = parse_time(actuals["utc_timestamp"])

source["demand"] = pd.to_numeric(
    source["demand"], errors="coerce"
)

actuals["actual_demand"] = pd.to_numeric(
    actuals["actual_demand"], errors="coerce"
)

# Keep valid records
source = source.dropna(
    subset=["utc_timestamp", "demand"]
)

actuals = actuals.dropna(
    subset=["utc_timestamp", "actual_demand"]
)

# Compare at identical timestamps
comparison = source.merge(
    actuals[["utc_timestamp", "actual_demand"]],
    on="utc_timestamp",
    how="inner"
)

comparison["difference"] = (
    comparison["demand"] - comparison["actual_demand"]
).abs()

print("Matched observations:", len(comparison))

print("\nDemand comparison:")
print(comparison["difference"].describe())

print("\nLargest differences:")
print(
    comparison.nlargest(10, "difference")[
        ["utc_timestamp", "demand",
         "actual_demand", "difference"]
    ].to_string(index=False)
)

print(
    "\nExact matches:",
    (comparison["difference"] < 0.01).sum()
)