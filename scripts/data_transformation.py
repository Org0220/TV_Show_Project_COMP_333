"""
Data Transformation for TV Show project

This script loads the CLEANED dataset created by data preparation
(`data/processed/cleaned_tv_shows.csv`) and performs:

1. Data normalization / scaling for numeric features:
   - Min–max scaling to [0, 1]
   - Z-score normalization
   - Decimal scaling

2. Date encoding:
   - Extract year, month, day-of-week, weekend flag, and decade
     from `premiere_date` (if available).

Outputs:
- data/processed/transformed_tv_shows.csv
- data/processed/data_transformation_report.txt

Run:
    python scripts/data_transformation_final.py
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "processed" / "cleaned_tv_shows.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "transformed_tv_shows.csv"
REPORT_TXT = ROOT / "data" / "processed" / "data_transformation_report.txt"


def min_max_scale(series: pd.Series, feature_range=(0.0, 1.0)):
    """
    Min–max scaling:
        v' = (v - min) / (max - min) * (new_max - new_min) + new_min

    Returns (scaled_series, stats_dict)
    """
    s = pd.to_numeric(series, errors="coerce")
    new_min, new_max = feature_range
    old_min = s.min(skipna=True)
    old_max = s.max(skipna=True)

    if pd.isna(old_min) or pd.isna(old_max) or old_min == old_max:
        # Constant or empty column – return NaNs (or zeros) but log this
        scaled = pd.Series(np.nan, index=series.index, dtype="float64")
        stats = {
            "old_min": old_min,
            "old_max": old_max,
            "note": "constant_or_empty_column",
        }
        return scaled, stats

    scaled = (s - old_min) / (old_max - old_min)
    scaled = scaled * (new_max - new_min) + new_min

    stats = {
        "old_min": float(old_min),
        "old_max": float(old_max),
        "new_min": float(scaled.min(skipna=True)),
        "new_max": float(scaled.max(skipna=True)),
        "mean": float(scaled.mean(skipna=True)),
        "std": float(scaled.std(skipna=True)),
    }
    return scaled, stats


def z_score_scale(series: pd.Series):
    """
    Z-score normalization:
        v' = (v - mean) / std

    Returns (scaled_series, stats_dict)
    """
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)

    if pd.isna(mean) or pd.isna(std) or std == 0:
        scaled = pd.Series(np.nan, index=series.index, dtype="float64")
        stats = {
            "mean": mean,
            "std": std,
            "note": "zero_or_undefined_std",
        }
        return scaled, stats

    scaled = (s - mean) / std
    stats = {
        "mean": float(mean),
        "std": float(std),
        "new_min": float(scaled.min(skipna=True)),
        "new_max": float(scaled.max(skipna=True)),
    }
    return scaled, stats


def decimal_scale(series: pd.Series):
    """
    Decimal scaling normalization:
        v' = v / 10^j
    where j is the smallest integer such that max(|v'|) < 1.

    Returns (scaled_series, stats_dict)
    """
    s = pd.to_numeric(series, errors="coerce")
    max_abs = s.abs().max(skipna=True)

    if pd.isna(max_abs) or max_abs == 0:
        scaled = s.copy().astype("float64")
        stats = {
            "max_abs": max_abs,
            "j": 0,
            "note": "max_abs_zero_or_nan",
        }
        return scaled, stats

    # smallest j such that max_abs / 10^j < 1
    j = int(np.ceil(np.log10(max_abs + 1e-12)))
    if j < 0:
        # if max_abs < 1, no scaling needed
        j = 0

    factor = 10 ** j
    scaled = s / factor

    stats = {
        "max_abs_original": float(max_abs),
        "j": j,
        "max_abs_scaled": float(scaled.abs().max(skipna=True)),
    }
    return scaled, stats


def main():
    report_lines = []
    report_lines.append(f"Data transformation run: {datetime.now().isoformat()}")
    report_lines.append("=" * 80 + "\n")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    # Load cleaned dataset
    df = pd.read_csv(INPUT_CSV)
    n_rows, n_cols = df.shape
    report_lines.append(f"Loaded cleaned data: {INPUT_CSV}")
    report_lines.append(f"Rows: {n_rows:,}, Columns: {n_cols:,}\n")

    # Ensure correct types for relevant columns
    if "premiere_date" in df.columns:
        df["premiere_date"] = pd.to_datetime(df["premiere_date"], errors="coerce")

    for col in ["runtime_minutes", "rating_avg", "release_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------------------------------------------------
    # 1. DATE ENCODING
    # ---------------------------------------------------------------------
    report_lines.append("DATE ENCODING")
    report_lines.append("-" * 80)

    if "premiere_date" in df.columns:
        df["premiere_year"] = df["premiere_date"].dt.year
        df["premiere_month"] = df["premiere_date"].dt.month
        df["premiere_dayofweek"] = df["premiere_date"].dt.dayofweek  # 0=Mon,...,6=Sun
        df["premiere_is_weekend"] = df["premiere_dayofweek"].isin([5, 6]).astype("Int64")
        df["premiere_decade"] = (df["premiere_year"] // 10) * 10

        report_lines.append(
            "Created date-encoded features from 'premiere_date': "
            "premiere_year, premiere_month, premiere_dayofweek, "
            "premiere_is_weekend (0/1), premiere_decade"
        )
        n_nat = df["premiere_date"].isna().sum()
        report_lines.append(f"  - premiere_date nulls (NaT): {n_nat:,}")
    else:
        report_lines.append("  - Column 'premiere_date' not found; skipping date encoding.")

    report_lines.append("")

    # ---------------------------------------------------------------------
    # 2. DATA NORMALIZATION / SCALING
    # ---------------------------------------------------------------------
    report_lines.append("DATA NORMALIZATION & DIFFERENT SCALING FUNCTIONS")
    report_lines.append("-" * 80)

    # Decide which numeric columns to scale
    numeric_candidates = []
    for col in ["runtime_minutes", "rating_avg", "release_year"]:
        if col in df.columns:
            numeric_candidates.append(col)

    if not numeric_candidates:
        report_lines.append("No numeric candidates found for scaling. Exiting scaling section.\n")
    else:
        report_lines.append("Numeric features selected for scaling:")
        for col in numeric_candidates:
            non_null = df[col].notna().sum()
            report_lines.append(f"  - {col} (non-null: {non_null:,})")

        report_lines.append("")

        # For each numeric column, create scaled versions with different functions
        for col in numeric_candidates:
            report_lines.append(f"\nFeature: {col}")
            report_lines.append("  -- Min–max scaling to [0, 1] --")
            mm_col = f"{col}_minmax_01"
            df[mm_col], mm_stats = min_max_scale(df[col])
            report_lines.append(f"    Created column: {mm_col}")
            for k, v in mm_stats.items():
                report_lines.append(f"      {k}: {v}")

            report_lines.append("  -- Z-score scaling --")
            zs_col = f"{col}_zscore"
            df[zs_col], zs_stats = z_score_scale(df[col])
            report_lines.append(f"    Created column: {zs_col}")
            for k, v in zs_stats.items():
                report_lines.append(f"      {k}: {v}")

            report_lines.append("  -- Decimal scaling --")
            ds_col = f"{col}_decscale"
            df[ds_col], ds_stats = decimal_scale(df[col])
            report_lines.append(f"    Created column: {ds_col}")
            for k, v in ds_stats.items():
                report_lines.append(f"      {k}: {v}")

    # ---------------------------------------------------------------------
    # 3. FINALIZE & SAVE
    # ---------------------------------------------------------------------
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    report_lines.append("\n" + "=" * 80)
    report_lines.append("OUTPUT")
    report_lines.append("-" * 80)
    report_lines.append(f"Wrote transformed CSV: {OUTPUT_CSV}")
    report_lines.append(f"Final shape: rows={df.shape[0]:,}, cols={df.shape[1]:,}")

    # Save report
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nReport written to: {REPORT_TXT}")
    print(f"Transformed CSV:   {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
