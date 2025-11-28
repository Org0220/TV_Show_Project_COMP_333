"""
Data Transformation for TV Show project

This script loads the CLEANED dataset created by data preparation
(`data/processed/cleaned_tv_shows.csv`) and performs:

1. Data normalization / scaling for numeric features:
   - Min-max scaling to [0, 1]
   - Z-score normalization
   - Decimal scaling
   (Applied differently per feature.)

2. Date encoding:
   - Extract year, month, day-of-week, weekend flag, and decade
     from `premiere_date` (if available).

3. Keep all cleaned columns and append engineered features; nothing is dropped.

Outputs:
- data/processed/transformed_tv_shows.csv
- data/processed/data_transformation_report.txt
- PostgreSQL table: integrated_tv_shows_transformed

Run:
    python scripts/data_transformation_final.py
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "processed" / "cleaned_tv_shows.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "transformed_tv_shows.csv"
REPORT_TXT = ROOT / "data" / "processed" / "data_transformation_report.txt"

# Import DB config
sys.path.insert(0, str(ROOT))
from config.db_config import SQLALCHEMY_URL  # type: ignore


# ---------------------------------------------------
# Scaling functions
# ---------------------------------------------------
def min_max_scale(series: pd.Series, feature_range=(0.0, 1.0)):
    s = pd.to_numeric(series, errors="coerce")
    new_min, new_max = feature_range
    old_min = s.min(skipna=True)
    old_max = s.max(skipna=True)

    if pd.isna(old_min) or pd.isna(old_max) or old_min == old_max:
        scaled = pd.Series(np.nan, index=series.index, dtype="float64")
        return scaled, {"old_min": old_min, "old_max": old_max, "note": "constant_or_empty_column"}

    scaled = (s - old_min) / (old_max - old_min)
    scaled = scaled * (new_max - new_min) + new_min

    return scaled, {
        "old_min": float(old_min),
        "old_max": float(old_max),
        "new_min": float(scaled.min(skipna=True)),
        "new_max": float(scaled.max(skipna=True)),
        "mean": float(scaled.mean(skipna=True)),
        "std": float(scaled.std(skipna=True)),
    }


def z_score_scale(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)

    if pd.isna(mean) or pd.isna(std) or std == 0:
        scaled = pd.Series(np.nan, index=series.index, dtype="float64")
        return scaled, {"mean": mean, "std": std, "note": "zero_or_undefined_std"}

    scaled = (s - mean) / std
    return scaled, {
        "mean": float(mean),
        "std": float(std),
        "new_min": float(scaled.min(skipna=True)),
        "new_max": float(scaled.max(skipna=True)),
    }


def decimal_scale(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    max_abs = s.abs().max(skipna=True)

    if pd.isna(max_abs) or max_abs == 0:
        scaled = s.copy().astype("float64")
        return scaled, {"max_abs": max_abs, "j": 0, "note": "max_abs_zero_or_nan"}

    j = int(np.ceil(np.log10(max_abs + 1e-12)))
    if j < 0:
        j = 0

    factor = 10**j
    scaled = s / factor
    return scaled, {
        "max_abs_original": float(max_abs),
        "j": j,
        "max_abs_scaled": float(scaled.abs().max(skipna=True)),
    }


# ---------------------------------------------------
# Save to DB
# ---------------------------------------------------
def save_to_database(df: pd.DataFrame, report_lines: list[str]):
    table_name = "integrated_tv_shows_transformed"
    report_lines.append("\nDATABASE SAVE")
    report_lines.append("-" * 80)

    try:
        engine = create_engine(SQLALCHEMY_URL)

        with engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=500, method="multi")

            if "id" in df.columns:
                conn.execute(text(f"ALTER TABLE {table_name} ALTER COLUMN id SET NOT NULL;"))
                conn.execute(text(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_pkey;"))
                conn.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY (id);"))

        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()

        report_lines.append(f"  - Saved transformed dataset to PostgreSQL table '{table_name}'.")
        report_lines.append(f"  - Row count in database: {count:,}")

    except Exception as e:
        report_lines.append(f"  - ERROR saving to database: {e}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    report_lines = []
    report_lines.append(f"Data transformation run: {datetime.now().isoformat()}")
    report_lines.append("=" * 80 + "\n")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    report_lines.append(f"Loaded cleaned data: {INPUT_CSV}")
    report_lines.append(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]:,}\n")

    # Convert types
    if "premiere_date" in df.columns:
        df["premiere_date"] = pd.to_datetime(df["premiere_date"], errors="coerce")

    for col in ["runtime_minutes", "rating_avg", "release_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------------
    # DATE ENCODING
    # -------------------------------------------------------------
    report_lines.append("DATE ENCODING")
    report_lines.append("-" * 80)

    if "premiere_date" in df.columns:
        df["premiere_year"] = df["premiere_date"].dt.year
        df["premiere_month"] = df["premiere_date"].dt.month
        df["premiere_dayofweek"] = df["premiere_date"].dt.dayofweek
        df["premiere_is_weekend"] = df["premiere_dayofweek"].isin([5, 6]).astype("Int64")
        df["premiere_decade"] = (df["premiere_year"] // 10) * 10

        report_lines.append(
            "Created: premiere_year, premiere_month, premiere_dayofweek, "
            "premiere_is_weekend, premiere_decade"
        )

    report_lines.append("")

    # -------------------------------------------------------------
    # SCALING (different method per feature)
    # -------------------------------------------------------------
    report_lines.append("DATA NORMALIZATION / SCALING")
    report_lines.append("-" * 80)

    # runtime_minutes -> Z-score
    if "runtime_minutes" in df.columns:
        df["runtime_minutes_zscore"], stats_rt = z_score_scale(df["runtime_minutes"])
        report_lines.append("Applied Z-score scaling to 'runtime_minutes' -> 'runtime_minutes_zscore'")
        for k, v in stats_rt.items():
            report_lines.append(f"  runtime_minutes_zscore {k}: {v}")

    # rating_avg -> Min-max [0, 1]
    if "rating_avg" in df.columns:
        df["rating_avg_minmax_01"], stats_ra = min_max_scale(df["rating_avg"])
        report_lines.append("Applied Min-max [0,1] scaling to 'rating_avg' -> 'rating_avg_minmax_01'")
        for k, v in stats_ra.items():
            report_lines.append(f"  rating_avg_minmax_01 {k}: {v}")

    # release_year -> Decimal scaling
    if "release_year" in df.columns:
        df["release_year_decscale"], stats_ry = decimal_scale(df["release_year"])
        report_lines.append("Applied Decimal scaling to 'release_year' -> 'release_year_decscale'")
        for k, v in stats_ry.items():
            report_lines.append(f"  release_year_decscale {k}: {v}")

    # -------------------------------------------------------------
    # FEATURE RETENTION
    # -------------------------------------------------------------
    report_lines.append("\nFEATURE RETENTION")
    report_lines.append("-" * 80)
    report_lines.append(
        "Kept all original cleaned columns and appended engineered features "
        "(date parts + scaled numeric columns). No columns were removed."
    )

    # -------------------------------------------------------------
    # SAVE TO DB
    # -------------------------------------------------------------
    save_to_database(df, report_lines)

    # -------------------------------------------------------------
    # SAVE FILES
    # -------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("OUTPUT")
    report_lines.append(f"Wrote transformed CSV: {OUTPUT_CSV}")
    report_lines.append(f"Final shape: rows={df.shape[0]:,}, cols={df.shape[1]:,}")

    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()

