"""
Data Preparation for TV Show integration

This script loads the integrated dataset created by the integration step
(`data/processed/integrated_tv_shows.csv`), performs data cleaning and
transformations, detects/removes duplicates (by title + release_year),
detects outliers (IQR method) for `runtime_minutes` and `rating_avg`,
and writes a cleaned CSV plus a plain-text report describing actions taken
and examples of transformations.

Outputs:
- data/processed/cleaned_tv_shows.csv
- data/processed/data_preparation_report.txt

Run:
    python scripts/data_preparation_final.py

"""
from pathlib import Path
import ast
import json
from datetime import datetime
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / 'data' / 'processed' / 'integrated_tv_shows.csv'
OUTPUT_CSV = ROOT / 'data' / 'processed' / 'cleaned_tv_shows.csv'
REPORT_TXT = ROOT / 'data' / 'processed' / 'data_preparation_report.txt'


def normalize_title(title: str) -> str:
    if pd.isna(title):
        return ''
    t = str(title).strip().lower()
    # remove simple country codes like (u.s.), (uk), (us)
    t = re.sub(r"\s*\(u\.s\.\)\s*", '', t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\(uk\)\s*", '', t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\(us\)\s*", '', t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", ' ', t)
    return t


def parse_rating(rating_cell):
    """Extract numeric average rating from the `rating` column.

    The integrated CSV stores values like `{'average': 7.5}` or `{'average': None}`
    as literal Python dict strings. This function tries to safely parse and
    return a float or NaN.
    """
    if pd.isna(rating_cell):
        return np.nan
    # If already numeric
    if isinstance(rating_cell, (int, float, np.floating, np.integer)):
        return float(rating_cell)

    s = str(rating_cell).strip()
    # Try literal_eval for Python dict strings
    try:
        val = ast.literal_eval(s)
        if isinstance(val, dict) and 'average' in val:
            avg = val.get('average')
            return float(avg) if avg is not None else np.nan
    except Exception:
        pass

    # Fallback: regex to find a floating number
    m = re.search(r"([0-9]+\.?[0-9]*)", s)
    if m:
        return float(m.group(1))

    return np.nan


def iqr_outliers(series: pd.Series, k=1.5):
    """Return boolean mask of outliers using the IQR rule."""
    s = series.dropna()
    if s.empty:
        return pd.Series(False, index=series.index)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def main():
    report_lines = []
    report_lines.append(f"Data preparation run: {datetime.now().isoformat()}\n")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, keep_default_na=True, na_values=['', 'None'])
    report_lines.append(f"Loaded input: {INPUT_CSV} -> rows: {len(df):,}, cols: {len(df.columns)}")

    # --- Initial null summary ---
    null_summary = df.isna().sum().sort_values(ascending=False)
    report_lines.append('\nNULL VALUE SUMMARY (per column):')
    for col, cnt in null_summary.items():
        report_lines.append(f"  - {col}: {cnt:,} nulls")

    # Examples of null cases: sample up to 3 rows where important columns are null
    examples = []
    important_cols = ['runtime_minutes', 'rating', 'language', 'description', 'premiere_date', 'release_year']
    for col in important_cols:
        if col in df.columns:
            null_rows = df[df[col].isna()].head(3)
            if not null_rows.empty:
                examples.append((col, null_rows))

    if examples:
        report_lines.append('\nEXAMPLES OF NULL CASES (first up to 3 each):')
        for col, rows in examples:
            report_lines.append(f"\nColumn: {col} -> {len(rows)} example(s):")
            for _, r in rows.iterrows():
                report_lines.append(f"  - title={r.get('title')!r}, release_year={r.get('release_year')!r}, {col}={r.get(col)!r}")

    # --- Transformations ---
    report_lines.append('\nTRANSFORMATIONS APPLIED:')

    # Normalize title
    df['title_normalized'] = df['title'].apply(normalize_title)
    report_lines.append("  - Created 'title_normalized' (lowercase, remove simple country codes)")

    # Ensure release_year is integer-like
    if 'release_year' in df.columns:
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').astype('Int64')
        report_lines.append("  - Converted 'release_year' to nullable integer (Int64)")

    # Parse premiere_date to datetime
    if 'premiere_date' in df.columns:
        df['premiere_date'] = pd.to_datetime(df['premiere_date'], errors='coerce')
        report_lines.append("  - Parsed 'premiere_date' to datetime (NaT when parsing fails)")

    # Parse rating to numeric average
    df['rating_avg'] = df['rating'].apply(parse_rating)
    report_lines.append("  - Extracted numeric 'rating_avg' from 'rating' column")

    # Parse genres that may be string lists like "['Comedy', 'Music']" into Python lists
    def parse_genres(g):
        if pd.isna(g):
            return []
        if isinstance(g, list):
            return g
        s = str(g).strip()
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return list(val)
        except Exception:
            pass
        # fallback: split on commas
        s2 = re.sub(r"[\[\]'\"]", '', s)
        parts = [p.strip() for p in s2.split(',') if p.strip()]
        return parts

    if 'genres' in df.columns:
        df['genres_parsed'] = df['genres'].apply(parse_genres)
        report_lines.append("  - Parsed 'genres' into 'genres_parsed' (list)")

    # Fill common nulls / impute
    #  - runtime_minutes: fill with median
    if 'runtime_minutes' in df.columns:
        runtime_median = pd.to_numeric(df['runtime_minutes'], errors='coerce').median()
        before_nulls = df['runtime_minutes'].isna().sum()
        df['runtime_minutes'] = pd.to_numeric(df['runtime_minutes'], errors='coerce')
        df['runtime_minutes'] = df['runtime_minutes'].fillna(runtime_median)
        after_nulls = df['runtime_minutes'].isna().sum()
        report_lines.append(f"  - Imputed 'runtime_minutes' nulls: before={before_nulls}, after={after_nulls}, filled with median={runtime_median}")

    #  - rating_avg: leave as NaN but report; optionally fill with global median
    rating_median = df['rating_avg'].median()
    rating_nulls_before = df['rating_avg'].isna().sum()
    # We will not overwrite rating NaNs by default; record strategy
    report_lines.append(f"  - 'rating_avg' nulls: {rating_nulls_before:,} (median={rating_median}) -- retained NaN (no aggressive imputation)")

    #  - language: fill with 'Unknown'
    if 'language' in df.columns:
        lang_nulls_before = df['language'].isna().sum()
        df['language'] = df['language'].fillna('Unknown')
        lang_nulls_after = df['language'].isna().sum()
        report_lines.append(f"  - Filled 'language' nulls: before={lang_nulls_before}, after={lang_nulls_after}, filled with 'Unknown'")

    # --- Duplicate detection and removal ---
    report_lines.append('\nDUPLICATE DETECTION (title_normalized + release_year):')
    df['dup_key'] = df['title_normalized'].astype(str) + '||' + df['release_year'].astype(str)
    dup_counts = df.duplicated(subset=['dup_key'], keep=False).sum()
    report_lines.append(f"  - Duplicate candidate rows (by key): {dup_counts:,}")

    # Show up to 5 examples of duplicates (grouped)
    dup_examples = []
    grouped = df[df.duplicated(subset=['dup_key'], keep=False)].groupby('dup_key')
    for key, g in grouped:
        if len(dup_examples) >= 5:
            break
        dup_examples.append((key, g[['title', 'release_year', 'premiere_date']]))

    if dup_examples:
        report_lines.append('  - Examples of duplicate groups (up to 5):')
        for key, g in dup_examples:
            report_lines.append(f"    * key={key} -> {len(g)} rows")
            for _, r in g.head(3).iterrows():
                report_lines.append(f"      - title={r['title']!r}, release_year={r['release_year']!r}, premiere_date={r['premiere_date']!r}")

    # Remove duplicates keeping the first occurrence
    before = len(df)
    df = df.drop_duplicates(subset=['dup_key'], keep='first').reset_index(drop=True)
    after = len(df)
    removed = before - after
    report_lines.append(f"  - Removed duplicates: {removed:,} rows (kept first occurrence). Resulting rows: {after:,}")

    # Drop helper column
    df = df.drop(columns=['dup_key'])

    # --- Outlier detection ---
    report_lines.append('\nOUTLIER DETECTION (IQR method):')
    outlier_details = {}
    # runtime_minutes
    if 'runtime_minutes' in df.columns:
        mask_rt = iqr_outliers(df['runtime_minutes'])
        n_rt = mask_rt.sum()
        outlier_details['runtime_minutes'] = n_rt
        report_lines.append(f"  - runtime_minutes outliers: {n_rt}")
        if n_rt > 0:
            report_lines.append("    Examples:")
            for _, r in df[mask_rt].head(5).iterrows():
                report_lines.append(f"      - title={r['title']!r}, release_year={r['release_year']!r}, runtime_minutes={r['runtime_minutes']}")

    # rating_avg
    if 'rating_avg' in df.columns:
        mask_rtg = iqr_outliers(df['rating_avg'])
        n_rtg = mask_rtg.sum()
        outlier_details['rating_avg'] = n_rtg
        report_lines.append(f"  - rating_avg outliers: {n_rtg}")
        if n_rtg > 0:
            report_lines.append("    Examples:")
            for _, r in df[mask_rtg].head(5).iterrows():
                report_lines.append(f"      - title={r['title']!r}, release_year={r['release_year']!r}, rating_avg={r['rating_avg']}")

    # Optionally, we could cap outliers — here we only flag and report them.
    report_lines.append("  - Strategy: outliers are flagged and reported; no automatic capping applied.")

    # --- Finalize and save outputs ---
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    report_lines.append(f"\nWrote cleaned CSV: {OUTPUT_CSV} (rows: {len(df):,})")

    # Save the report
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(report_lines))

    print('\n'.join(report_lines))
    print(f"\nReport written to: {REPORT_TXT}\nCleaned CSV: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
